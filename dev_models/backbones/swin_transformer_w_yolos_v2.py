from logging import log
import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision import models

from .mmcv_custom import load_checkpoint

import mmcv
import math

from util.misc import NestedTensor
from typing import Dict, List
from dev_models.backbone import build_position_encoding
from loguru import logger
__all__ = ['swin_nano_yolos_v2', 'swin_tiny_yolos_v2', 'swin_small_yolos_v2', 'swin_base_yolos_v2']

def masked_sin_pos_encoding(x, mask, num_pos_feats, temperature=10000, scale=2 * math.pi):
    """ Masked Sinusoidal Positional Encoding
    Parameters:
        x: [PATCH] tokens
        mask: the padding mask for [PATCH] tokens
        num_pos_feats: the size of channel dimension
        temperature: the temperature value
        scale: the normalization scale
    Returns:
        pos: Sinusoidal positional encodings
    """

    num_pos_feats = num_pos_feats // 2
    not_mask = ~mask

    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t

    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3)

    return pos


class Conv1x1(nn.Module):
    def __init__(self, in_channel, out_channel, force=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(out_channel)
        if not force and in_channel == out_channel:
            self.conv = self.norm = nn.Identity()

    def forward(self, x, H=None, W=None):
        # x: (b, n, c) or (b, c, h, w)
        # ret: (b, n, c)
        if len(x.shape) == 3:
            B = x.shape[0]
            x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Conv3x3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channel)

    def forward(self, x, H=None, W=None):
        # x: (b, n, c) or (b, c, h, w)
        # ret: (b, n, c)
        if len(x.shape) == 3:
            B = x.shape[0]
            x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.conv(x)
        newshape = tuple(x.shape[-2:])
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, newshape


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH


        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, det, mask=None, cross_attn=False, cross_attn_mask=None):
        """ Forward function.
        RAM module receives [Patch] and [DET] tokens and returns their calibrated ones
        Parameters:
            x: [PATCH] tokens
            det: [DET] tokens
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None -> mask for shifted window attention
            "additional inputs for RAM"
            cross_attn: whether to use cross-attention [det x patch] (for selective cross-attention)
            cross_attn_mask: mask for cross-attention
        Returns:
            patch_x: the calibrated [PATCH] tokens
            det_x: the calibrated [DET] tokens
        """

        assert self.window_size[0] == self.window_size[1]
        window_size = self.window_size[0]
        local_map_size = window_size * window_size

        # projection before window partitioning
        if not cross_attn:
            B, H, W, C = x.shape
            N = H * W
            x = x.view(B, N, C)
            x = torch.cat([x, det], dim=1)
            full_qkv = self.qkv(x)
            patch_qkv, det_qkv = full_qkv[:, :N, :], full_qkv[:, N:, :]
        else:
            B, H, W, C = x[0].shape
            N = H * W
            _, ori_H, ori_W, _ = x[1].shape
            ori_N = ori_H * ori_W

            shifted_x = x[0].view(B, N, C)
            cross_x = x[1].view(B, ori_N, C)
            x = torch.cat([shifted_x, cross_x, det], dim=1)
            full_qkv = self.qkv(x)
            patch_qkv, cross_patch_qkv, det_qkv = \
                full_qkv[:, :N, :], full_qkv[:, N:N + ori_N, :], full_qkv[:, N + ori_N:, :]
        patch_qkv = patch_qkv.view(B, H, W, -1)

        # window partitioning for [PATCH] tokens
        patch_qkv = window_partition(patch_qkv, window_size)  # nW*B, window_size, window_size, C
        B_ = patch_qkv.shape[0]
        patch_qkv = patch_qkv.reshape(B_, window_size * window_size, 3, self.num_heads, C // self.num_heads)
        _patch_qkv = patch_qkv.permute(2, 0, 3, 1, 4)
        patch_q, patch_k, patch_v = _patch_qkv[0], _patch_qkv[1], _patch_qkv[2]

        # [PATCH x PATCH] self-attention using window partitions
        patch_q = patch_q * self.scale
        patch_attn = (patch_q @ patch_k.transpose(-2, -1))
        # add relative pos bias for [patch x patch] self-attention
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        patch_attn = patch_attn + relative_position_bias.unsqueeze(0)

        # if shifted window is used, it needs to apply the mask
        if mask is not None:
            nW = mask.shape[0]
            patch_attn = patch_attn.view(B_ // nW, nW, self.num_heads, local_map_size, local_map_size) + \
                         mask.unsqueeze(1).unsqueeze(0)
            patch_attn = patch_attn.view(-1, self.num_heads, local_map_size, local_map_size)

        patch_attn = self.softmax(patch_attn)
        patch_attn = self.attn_drop(patch_attn)
        patch_x = (patch_attn @ patch_v).transpose(1, 2).reshape(B_, window_size, window_size, C)

        # extract qkv for [DET] tokens
        det_qkv = det_qkv.view(B, -1, 3, self.num_heads, C // self.num_heads)
        det_qkv = det_qkv.permute(2, 0, 3, 1, 4)
        det_q, det_k, det_v = det_qkv[0], det_qkv[1], det_qkv[2]

        # if cross-attention is activated
        if cross_attn:

            # reconstruct the spatial form of [PATCH] tokens for global [DET x PATCH] attention
            cross_patch_qkv = cross_patch_qkv.view(B, ori_H, ori_W, 3, self.num_heads, C // self.num_heads)
            patch_kv = cross_patch_qkv[:, :, :, 1:, :, :].permute(3, 0, 4, 1, 2, 5).contiguous()
            patch_kv = patch_kv.view(2, B, self.num_heads, ori_H * ori_W, -1)

            # extract "key and value" of [PATCH] tokens for cross-attention
            cross_patch_k, cross_patch_v = patch_kv[0], patch_kv[1]

            # bind key and value of [PATCH] and [DET] tokens for [DET X [PATCH, DET]] attention
            det_k, det_v = torch.cat([cross_patch_k, det_k], dim=2), torch.cat([cross_patch_v, det_v], dim=2)

        # [DET x DET] self-attention or binded [DET x [PATCH, DET]] attention
        det_q = det_q * self.scale
        det_attn = (det_q @ det_k.transpose(-2, -1))
        # apply cross-attention mask if available
        if cross_attn_mask is not None:
            det_attn = det_attn + cross_attn_mask
        det_attn = self.softmax(det_attn)
        det_attn = self.attn_drop(det_attn)
        det_x = (det_attn @ det_v).transpose(1, 2).reshape(B, -1, C)

        # reverse window for [PATCH] tokens <- the output of [PATCH x PATCH] self attention
        patch_x = window_reverse(patch_x, window_size, H, W)

        # projection for outputs from multi-head
        x = torch.cat([patch_x.view(B, H*W, C), det_x], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)

        # decompose after FFN into [PATCH] and [DET] tokens
        patch_x = x[:, :H * W, :].view(B, H, W, C)
        det_x = x[:, H * W:, :]

        return patch_x, det_x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix, pos, cross_attn, cross_attn_mask):
        """ Forward function.
        Parameters:
            x: Input feature, tensor size (B, H*W + DET, C). i.e., binded [PATCH, DET] tokens
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
            "additional inputs'
            pos: (patch_pos, det_pos)
            cross_attn: whether to use cross attn [det x [det + patch]]
            cross_attn_mask: attention mask for cross-attention
        Returns:
            x: calibrated & binded [PATCH, DET] tokens
        """

        B, L, C = x.shape
        H, W = self.H, self.W

        assert L == H * W + self.det_token_num, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x, det = x[:, :H * W, :], x[:, H * W:, :]
        x = x.view(B, H, W, C)
        orig_x = x

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # projection for det positional encodings: make the channel size suitable for the current layer
        patch_pos, det_pos = pos
        det_pos = self.fuse_det_pos_linear(det_pos)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # prepare cross-attn and add positional encodings
        if cross_attn:
            # patch token (for cross-attention) + Sinusoidal pos encoding
            cross_patch = orig_x + patch_pos
            # det token + learnable pos encoding
            det = det + det_pos
            shifted_x = (shifted_x, cross_patch)
        else:
            # it cross_attn is deativated, only [PATCH] and [DET] self-attention are performed
            det = det + det_pos
            shifted_x = shifted_x

        # W-MSA/SW-MSA
        shifted_x, det = self.attn(shifted_x, mask=attn_mask,
                                   # additional parameters
                                   det=det,
                                   cross_attn=cross_attn,
                                   cross_attn_mask=cross_attn_mask)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = torch.cat([x, det], dim=1)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Parameters:
            x: Input feature, tensor size (B, H*W, C), i.e., binded [PATCH, DET] tokens
            H, W: Spatial resolution of the input feature.
        Returns:
            x: merged [PATCH, DET] tokens;
            only [PATCH] tokens are reduced in spatial dim, while [DET] tokens is fix-scale
        """

        B, L, C = x.shape
        assert L == H * W + self.det_token_num, "input feature has wrong size"

        x, det = x[:, :H * W, :], x[:, H * W:, :]
        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # simply repeating for DET tokens
        det = det.repeat(1, 1, 4)

        x = torch.cat([x, det], dim=1)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 last=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        

    def forward(self, x, H, W, det_pos, input_mask, cross_attn=False):
        """ Forward function.
        Parameters:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            det_pos: pos encoding for det token
            input_mask: padding mask for inputs
            cross_attn: whether to use cross attn [det x [det + patch]]
        """

        B = x.shape[0]

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # mask for cyclic shift
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # compute sinusoidal pos encoding and cross-attn mask here to avoid redundant computation
        if cross_attn:

            _H, _W = input_mask.shape[1:]
            if not (_H == H and _W == W):
                input_mask = F.interpolate(input_mask[None].float(), size=(H, W)).to(torch.bool)[0]

            # sinusoidal pos encoding for [PATCH] tokens used in cross-attention
            patch_pos = masked_sin_pos_encoding(x, input_mask, self.dim)

            # attention padding mask due to the zero padding in inputs
            # the zero (padded) area is masked by 1.0 in 'input_mask'
            cross_attn_mask = input_mask.float()
            cross_attn_mask = cross_attn_mask.masked_fill(cross_attn_mask != 0.0, float(-100.0)). \
                masked_fill(cross_attn_mask == 0.0, float(0.0))

            # pad for detection token (this padding is required to process the binded [PATCH, DET] attention
            cross_attn_mask = cross_attn_mask.view(B, H * W).unsqueeze(1).unsqueeze(2)
            cross_attn_mask = F.pad(cross_attn_mask, (0, self.det_token_num), value=0)

        else:
            patch_pos = None
            cross_attn_mask = None

        # zip pos encodings
        pos = (patch_pos, det_pos)

        for n_blk, blk in enumerate(self.blocks):
            blk.H, blk.W = H, W

            # for selective cross-attention
            if cross_attn:
                _cross_attn = True
                _cross_attn_mask = cross_attn_mask
                _pos = pos # i.e., (patch_pos, det_pos)
            else:
                _cross_attn = False
                _cross_attn_mask = None
                _pos = (None, det_pos)

            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask,
                                          # additional inputs
                                          pos=_pos,
                                          cross_attn=_cross_attn,
                                          cross_attn_mask=_cross_attn_mask)
            else:
                x = blk(x, attn_mask,
                        # additional inputs
                        pos=_pos,
                        cross_attn=_cross_attn,
                        cross_attn_mask=_cross_attn_mask)

        # reduce the number of patch tokens, but maintaining a fixed-scale det tokens
        # meanwhile, the channel dim increases by a factor of 2
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class DWConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PvtAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False, poolsize=7):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(poolsize)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, det_tokens=None):
        Q = []
        # for x
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        Q.append(q)
        # for det_tokens
        det_B, det_N, det_C = det_tokens.shape
        det_q = self.q(det_tokens).reshape(det_B, det_N, self.num_heads, det_C // self.num_heads).permute(0,2,1,3)
        Q.append(det_q)

        K = []
        V = []
        # for x
        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        K.append(k)
        V.append(v)

        # for det_tokens
        # for det_tokens
        if self.linear:
            after_act_det_tokens = self.act(det_tokens)
        det_kv = self.kv(det_tokens).reshape(det_B, -1, 2, self.num_heads, det_C // self.num_heads).permute(2,0,3,1,4)
        det_k, det_v = det_kv[0], det_kv[1]
        K.append(det_k)
        V.append(det_v)

        Q = torch.cat(Q, dim=2)
        K = torch.cat(K, dim=2)
        V = torch.cat(V, dim=2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        X = (attn @ V).transpose(1, 2).reshape(B, -1, C)
        X = self.proj(X)
        X = self.proj_drop(X)

        scale_out, det_tokens = X[:, :-det_N, :], X[:, -det_N:, :]

        return scale_out, det_tokens

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)

        # return x

class FuseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False, poolsize=7):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            self.sr = nn.ModuleList([ nn.Conv2d(dim, dim, kernel_size=s, stride=s) for s in sr_ratio ])
            self.norm = nn.ModuleList([ nn.LayerNorm(dim) for _ in sr_ratio ])
        else:
            self.pool = nn.AdaptiveAvgPool2d(poolsize)
            self.sr = nn.ModuleList([ nn.Conv2d(dim, dim, kernel_size=1, stride=1) for _ in sr_ratio ])
            self.norm = nn.ModuleList([ nn.LayerNorm(dim) for _ in sr_ratio ])
            self.act = nn.GELU()
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, out, shapes, det_tokens=None):
        Q = []
        # for scale
        for x in out:
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            Q.append(q)

        # for det_tokens
        det_B, det_N, det_C = det_tokens.shape
        det_q = self.q(det_tokens).reshape(det_B, det_N, self.num_heads, det_C // self.num_heads).permute(0, 2, 1, 3)
        Q.append(det_q)

        K = []
        V = []
        # for scale
        assert len(out) == len(shapes) == len(self.sr) == len(self.norm)
        for x, shape, sr, norm in zip(out, shapes, self.sr, self.norm):
            H, W = shape
            x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
            if self.linear:
                x = self.pool(x)
            x = sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = norm(x)
            if self.linear:
                x = self.act(x)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            K.append(k)
            V.append(v)

        # for det_tokens
        if self.linear:
            after_act_det_tokens = self.act(det_tokens)
        det_kv = self.kv(det_tokens).reshape(det_B, -1, 2, self.num_heads, det_C // self.num_heads).permute(2,0,3,1,4)
        det_k, det_v = det_kv[0], det_kv[1]
        K.append(det_k)
        V.append(det_v)

        Q = torch.cat(Q, dim=2)
        K = torch.cat(K, dim=2)
        V = torch.cat(V, dim=2)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        X = (attn @ V).transpose(1, 2).reshape(B, -1, C)
        X = self.proj(X)
        X = self.proj_drop(X)

        scale_out, det_tokens = X[:, :-det_N, :], X[:, -det_N:, :]

        return scale_out, det_tokens

    def forward_extra(self, out_q, shapes_q, out_kv, shapes_kv, det_tokens=None):
        Q = []
        # for query scale
        for x in out_q:
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            Q.append(q)
        
        # for query det_tokens
        det_B, det_N, det_C = det_tokens.shape
        det_q = self.q(det_tokens).reshape(det_B, det_N, self.num_heads, det_C // self.num_heads).permute(0,2,1,3)
        Q.append(det_q)

        K = []
        V = []
        # for kv scale
        assert len(out_kv) == len(shapes_kv) == len(self.sr) == len(self.norm)
        for x, shape, sr, norm in zip(out_kv, shapes_kv, self.sr, self.norm):
            H, W = shape
            x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
            if self.linear:
                x = self.pool(x)
            x = sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = norm(x)
            if self.linear:
                x = self.act(x)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            K.append(k)
            V.append(v)
        
        # for kv det_tokens
        if self.linear:
            after_act_det_tokens = self.act(det_tokens)
        det_kv = self.kv(after_act_det_tokens).reshape(det_B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        det_k, det_v = det_kv[0], det_kv[1]
        K.append(det_k)
        V.append(det_v)


        Q = torch.cat(Q, dim=2)
        K = torch.cat(K, dim=2)
        V = torch.cat(V, dim=2)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        X = (attn @ V).transpose(1, 2).reshape(B, -1, C)
        X = self.proj(X)
        X = self.proj_drop(X)

        scale_out, det_tokens = X[:, :-det_N, :], X[:, -det_N:, :]

        return scale_out, det_tokens



class FuseBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=4, linear=False, poolsize=7,
                 fuse=False, fuse_extra=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.sr_ratio = tuple(sr_ratio) if isinstance(sr_ratio, (list, tuple)) else sr_ratio
        self.fuse_extra = fuse_extra
        self.fuse = fuse
        if not fuse:
            self.attn = PvtAttention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear, poolsize=poolsize)
        else:
            self.attn = FuseAttention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear, poolsize=poolsize)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DWConvMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, shapes=None, extra_kv=None, extra_kv_shapes=None, det_tokens=None):
        if self.fuse:
            if self.fuse_extra:
                sizes = [H * W for H, W in shapes]
                
                x = self.norm1(x)
                det_tokens = self.norm1(det_tokens)

                q = list(x.split(sizes, dim=1))
                q_shapes = list(shapes)
                kv = extra_kv + q
                kv_shapes = extra_kv_shapes + q_shapes

                after_attn_q, after_attn_det_tokens = self.attn.forward_extra(q, q_shapes, kv, kv_shapes, det_tokens=det_tokens)
                # for x
                x = x + self.drop_path(after_attn_q)

                x = self.norm2(x)

                shortcut_x = [self.mlp(x_, H, W) for x_, (H, W) in zip(x.split(sizes, dim=1), shapes)]
                shortcut_x = torch.cat(shortcut_x, dim=1)

                x = x + self.drop_path(shortcut_x)

                # for det_tokens
                det_tokens = det_tokens + self.drop_path(after_attn_det_tokens)
                det_tokens = self.norm2(det_tokens)
                det_tokens = det_tokens + self.drop_path(self.det_mlp(det_tokens)) # TODO?


            else:
                sizes = [H * W for H, W in shapes]
                
                x = self.norm1(x)
                det_tokens = self.norm1(det_tokens)

                after_attn_x, after_attn_det_tokens = self.attn(x.split(sizes, dim=1), shapes, det_tokens=det_tokens)
                # for x
                x = x + self.drop_path(after_attn_x)

                x = self.norm2(x)

                shortcut_x = [self.mlp(x_, H, W) for x_, (H, W) in zip(x.split(sizes, dim=1), shapes)]
                shortcut_x = torch.cat(shortcut_x, dim=1)

                x = x + self.drop_path(shortcut_x)

                # for det_tokens
                det_tokens = det_tokens + self.drop_path(after_attn_det_tokens)
                det_tokens = self.norm2(det_tokens)
                det_tokens = det_tokens + self.drop_path(self.det_mlp(det_tokens)) # TODO?
        else:
            after_attn_x, after_attn_det_tokens = self.attn(self.norm1(x), H, W, det_tokens=self.norm1(det_tokens))
            # for x
            x = x + self.drop_path(after_attn_x)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            # for det_tokens
            det_tokens = det_tokens + self.drop_path(after_attn_det_tokens)
            det_tokens = det_tokens + self.drop_path(self.det_mlp(self.norm2(det_tokens))) 

        return x, det_tokens


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 # fusing module
                 fuse_dim=256, 
                 fuse_num_heads=8, fuse_mlp_ratios=4, fuse_depth=3, fuse_linear=False, 
                 fuse_start_lvl=1, fuse_num_addition=1,
                 fuse_dense_lookback=False, fuse_lookback_extra_depth=1, fuse_single_scale=False,
                 # yolos det token param
                 det_token_num=100,
                 cross_indices=[3]
                ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # build cross scale fusion
        self._apply_cross_scale_fusion(drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate,
                                       norm_layer=norm_layer, qkv_bias=qkv_bias,
                                       qk_scale=qk_scale, drop_rate=drop_rate,
                                       fuse_dim=fuse_dim, 
                                       fuse_num_heads=fuse_num_heads, fuse_mlp_ratios=fuse_mlp_ratios, 
                                       fuse_depth=fuse_depth, fuse_linear=fuse_linear, 
                                       fuse_start_lvl=fuse_start_lvl, fuse_num_addition=fuse_num_addition,
                                       fuse_dense_lookback=fuse_dense_lookback, fuse_lookback_extra_depth=fuse_lookback_extra_depth, 
                                       fuse_single_scale=fuse_single_scale,
                                       det_token_num=det_token_num, cross_indices=cross_indices)

        # # add a norm layer for each output
        # for i_layer in out_indices:
        #     layer = norm_layer(num_features[i_layer])
        #     layer_name = f'norm{i_layer}'
        #     self.add_module(layer_name, layer)

        self._freeze_stages()

    def _apply_cross_scale_fusion(self, drop_path_rate=0.2, attn_drop_rate=0.,
                                  norm_layer=nn.LayerNorm, qkv_bias=True,
                                  qk_scale=None, drop_rate=0.,
                                  sr_ratios=[8, 4, 2, 1],
                                  fuse_dim=256, 
                                  fuse_num_heads=8, fuse_mlp_ratios=4, fuse_depth=3, 
                                  fuse_linear=False, fuse_start_lvl=1, fuse_num_addition=1,
                                  fuse_dense_lookback=False, fuse_lookback_extra_depth=1, fuse_single_scale=False,
                                  det_token_num=100, cross_indices=[3]):
        # init det tokens
        self.det_token_num = det_token_num
        self.fuse_det_tokens = nn.Parameter(torch.zeros(1, det_token_num,  self.num_features[0]))
        self.fuse_det_tokens = trunc_normal_(self.fuse_det_tokens, std=.02)
        # learnable positional encoding for detection tokens
        det_pos_embed = torch.zeros(1, det_token_num, fuse_dim)
        det_pos_embed = trunc_normal_(det_pos_embed, std=.02)
        self.fuse_det_pos_embed = nn.Parameter(det_pos_embed)

        self.num_channels = [self.num_features[i+1] for i in range(len(self.num_features)-1)]
        self.cross_indices = cross_indices
        # divisor to reduce the spatial size of the mask
        self.mask_divisor = 2 ** (len(self.layers) - len(self.cross_indices))
        # projection matrix for det pos encoding in each Swin layer (there are 4 blocks)
        for layer in self.layers:
            layer.det_token_num = det_token_num
            if layer.downsample is not None:
                layer.downsample.det_token_num = det_token_num
            for block in layer.blocks:
                block.det_token_num = det_token_num
                block.fuse_det_pos_linear = nn.Linear(fuse_dim, block.dim)
        # self.layers[-1].downsample = None
        self.fuse_det_linear = nn.Linear(self.num_features[-1], fuse_dim)

        # init CECA
        self.fuse_dense_lookback = fuse_dense_lookback
        self.fuse_lookback_extra_depth = fuse_lookback_extra_depth
        self.fuse_start_lvl = fuse_start_lvl
        self.fuse_num_addition = fuse_num_addition
        self.fuse_single_scale = fuse_single_scale
        # proj
        self.fuse_proj = nn.ModuleList([ Conv1x1(d, fuse_dim) for d in self.num_features[fuse_start_lvl:] ])

        # addition scale
        in_channels = self.num_features[-1]
        self.fuse_addin_proj = []
        for _ in range(fuse_num_addition):
            self.fuse_addin_proj.append(Conv3x3(in_channels, fuse_dim))
            in_channels = fuse_dim
        self.fuse_addin_proj = nn.ModuleList(self.fuse_addin_proj)

        # dense fusing
        fuse_num_scale = self.num_layers - fuse_start_lvl + fuse_num_addition

        if isinstance(fuse_depth, int):
            assert fuse_depth % (fuse_num_scale - 1) == 0
            fuse_depth_per_scale = [fuse_depth // (fuse_num_scale - 1)] * (fuse_num_scale - 1)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, fuse_depth)]
        else:
            assert isinstance(fuse_depth, (list, tuple))
            assert len(fuse_depth) == fuse_num_scale - 1
            fuse_depth_per_scale = fuse_depth
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(fuse_depth))]
        sr_ratios = list(sr_ratios + [sr_ratios[-1]] * fuse_num_addition)
        for k in range(fuse_num_scale - 1):  # no need to fuse if you have only one scale
            srr = sr_ratios[fuse_start_lvl : fuse_start_lvl + k + 2]  # not nearby or nearby_extra kv: all exist scales
            fuse_block = nn.ModuleList([FuseBlock(
                dim=fuse_dim, num_heads=fuse_num_heads, mlp_ratio=fuse_mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                sr_ratio=srr, linear=fuse_linear, fuse=True, fuse_extra=fuse_dense_lookback)
                for i in range(fuse_depth_per_scale[k])])
            # for det_tokens
            for blk in fuse_block:
                blk.det_mlp = Mlp(in_features=fuse_dim, hidden_features=fuse_dim, drop=drop_rate)
            fuse_norm = nn.LayerNorm(fuse_dim)

            self.add_module(f'fuse_layer{k+2}', fuse_block)
            self.add_module(f'fuse_norm{k+2}', fuse_norm)

        if fuse_dense_lookback and fuse_lookback_extra_depth > 0:
            fuse_block = nn.ModuleList([FuseBlock(
                dim=fuse_dim, num_heads=fuse_num_heads, mlp_ratio=fuse_mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
                sr_ratio=sr_ratios[fuse_start_lvl], linear=fuse_linear)
                for i in range(fuse_lookback_extra_depth)])
            # for det_tokens
            for blk in fuse_block:
                blk.det_mlp = Mlp(in_features=fuse_dim, hidden_features=fuse_dim, drop=drop_rate)
            fuse_norm = nn.LayerNorm(fuse_dim)
            logger.info('lookback srr:', fuse_block[0].sr_ratio)

            self.add_module(f'fuse_layer1', fuse_block)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            logger.error('pretrained must be a str or None')
            raise TypeError('pretrained must be a str or None')

    def forward_dense(self, x, mask):
        """Forward function."""
        # original input shape
        B, ori_H, ori_W = x.shape[0], x.shape[2], x.shape[3]
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        det_tokens = self.fuse_det_tokens.expand(B,-1,-1)
        det_pos = self.fuse_det_pos_embed
        # prepare a mask for cross attention
        mask = F.interpolate(mask[None].float(),
                     size=(Wh // self.mask_divisor, Ww // self.mask_divisor)).to(torch.bool)[0]

        outs = []
        spatial_shapes = []
        for i in range(self.num_layers+self.fuse_num_addition):
            if i < self.num_layers:
                # normal layer
                layer = self.layers[i]
                # whether to use cross-attention
                cross_attn = True if i in self.cross_indices else False

                # concat input
                x = torch.cat([x, det_tokens], dim=1)
                # inference
                x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww,
                                            # additional input for VIDT
                                            input_mask=mask,
                                            det_pos=det_pos,
                                            cross_attn=cross_attn)
                x_out, det_tokens_out = x_out[:, :-self.det_token_num, :], x_out[:, -self.det_token_num:, :]
                x, det_tokens = x[:, :-self.det_token_num, :], x[:, -self.det_token_num:, :]
            else:
                addin_proj = self.fuse_addin_proj[i - self.num_layers]
                x, (H, W) = addin_proj(x, H, W)
                x = x.transpose(1, 2).reshape(B, -1, H, W)
            
            if i >= self.fuse_start_lvl:
                # proj to fuse_dim
                if i < self.num_layers:
                    fuse_proj = self.fuse_proj[i - self.fuse_start_lvl]
                    x_out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                    outs.append(fuse_proj(x_out))
                    spatial_shapes.append((H, W))
                else:
                    outs.append(x.flatten(2).transpose(1, 2))
                    spatial_shapes.append((H, W))
        det_tokens = self.fuse_det_linear(det_tokens)
        det_tokens = det_tokens + det_pos
        for i in range(len(outs)):
            if i == 0 and self.fuse_dense_lookback and self.fuse_lookback_extra_depth > 0:
                fuse_block = getattr(self, f"fuse_layer1")
                fuse_norm = getattr(self, f"fuse_norm1")
                outs_0 = outs[0]
                for blk in fuse_block:
                    outs_0, det_tokens = blk(outs_0, *spatial_shapes[0], det_tokens=det_tokens)
                outs_0 = fuse_norm(outs_0)
                det_tokens = fuse_norm(det_tokens)
                outs[0] = outs_0

            if i >= 1:
                fuse_block = getattr(self, f"fuse_layer{i+1}")
                fuse_norm = getattr(self, f"fuse_norm{i+1}")
                if self.fuse_dense_lookback:
                    outs_holdout, spatial_shapes_holdout = outs[:i], spatial_shapes[:i]
                    outs_i, spatial_shapes_i = outs[i], spatial_shapes[i]
                    # outs_i = [outs_i]   
                    spatial_shapes_i = [spatial_shapes_i]
                for blk in fuse_block:
                    if self.fuse_dense_lookback:
                        outs_i, det_tokens = blk(outs_i, None, None, spatial_shapes_i, outs_holdout, spatial_shapes_holdout, det_tokens=det_tokens)
                    else:
                        outs_i, det_tokens = blk(outs_i, None, None, spatial_shapes_i, det_tokens=det_tokens)
                outs_i = fuse_norm(outs_i)
                det_tokens = fuse_norm(det_tokens)
                
                # sizes = [h * w for h, w in spatial_shapes_i]
                # outs_i = list(outs_i.split(sizes, dim=1))

                if self.fuse_dense_lookback:
                    outs[i] = outs_i
                    # spatial_shapes[i] = spatial_shapes_i

        outs = [x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x, (H, W) in zip(outs, spatial_shapes)]
        return outs, det_tokens, det_pos

    def forward(self, tensor_list: NestedTensor):
        xs, det_tokens, det_pos = self.forward_dense(tensor_list.tensors, tensor_list.mask)
        out: Dict[str, NestedTensor] = {}
        for i, x in enumerate(xs):
            if self.fuse_single_scale and i < len(xs) - 1: continue
            name = str(i)
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out, det_tokens, det_pos

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs, det_tokens, det_pos = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos, det_tokens, det_pos

def swin_nano_yolos_v2(pretrained=False, pretrained_path=None, out_indices=(1, 2, 3), **kwargs):
    model = SwinTransformer(embed_dim=48,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.,
                            attn_drop_rate=0.,
                            ape=False,
                            drop_path_rate=0.1,
                            patch_norm=True,
                            out_indices=out_indices,
                            use_checkpoint=False)
    model.init_weights(pretrained=pretrained_path)
    return model


def swin_tiny_yolos_v2(pretrained=False, pretrained_path=None, out_indices=(1, 2, 3), **kwargs):
    model = SwinTransformer(embed_dim=96,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.,
                            attn_drop_rate=0.,
                            ape=False,
                            drop_path_rate=0.1,
                            patch_norm=True,
                            out_indices=out_indices,
                            use_checkpoint=False)
    model.init_weights(pretrained=pretrained_path)
    return model

def swin_small_yolos_v2(pretrained=False, pretrained_path=None, out_indices=(1, 2, 3), **kwargs):
    model = SwinTransformer(embed_dim=96,
                            depths=[2, 2, 18, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.,
                            attn_drop_rate=0.,
                            ape=False,
                            drop_path_rate=0.2,
                            patch_norm=True,
                            out_indices=out_indices,
                            use_checkpoint=False)
    model.init_weights(pretrained=pretrained_path)
    return model


def swin_base_yolos_v2(pretrained=False, pretrained_path=None, out_indices=(1, 2, 3), **kwargs):
    model = SwinTransformer(embed_dim=128,
                            depths=[2, 2, 18, 2],
                            num_heads=[4, 8, 16, 32],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.,
                            attn_drop_rate=0.,
                            ape=False,
                            drop_path_rate=0.3,
                            patch_norm=True,
                            out_indices=out_indices,
                            use_checkpoint=False)
    model.init_weights(pretrained=pretrained_path)
    return model

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    fuse_single_scale = args.num_feature_levels == 1
    if args.vit_backbone == 'swin_nano_yolos_v2':
        logger.info(f'build backbone {args.vit_backbone}')
        backbone = SwinTransformer(embed_dim=48,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.,
                            attn_drop_rate=0.,
                            ape=False,
                            drop_path_rate=0.1,
                            patch_norm=True,
                            use_checkpoint=False,
                            # fuse param
                            fuse_mlp_ratios=4, 
                            fuse_depth=[3,3] if fuse_single_scale else [3,3,3],
                            fuse_linear=True,
                            fuse_dense_lookback=True, fuse_lookback_extra_depth=0,
                            fuse_num_addition=0 if fuse_single_scale else 1,
                            fuse_single_scale=fuse_single_scale,
                            # yolos det_token num
                            det_token_num=args.num_queries,
                            cross_indices=args.cross_indices
                            )
    else:
        logger.error(f"{args.vit_backbone} not supported")
    if fuse_single_scale:
        backbone.strides=[32]
        backbone.num_channels=[256]
    else:
        backbone.strides = [8, 16, 32, 64]
        backbone.num_channels = [256,256,256,256]
    backbone.non_backbone_names = sorted(list(set(['backbone.0.' + name 
                                        for name, param in backbone.named_parameters() if param.requires_grad and 'fuse' in name])))
    backbone.backbone_names     = sorted(list(set(['backbone.0.' + name 
                                        for name, param in backbone.named_parameters() if param.requires_grad and 'fuse' not in name])))
    backbone.init_weights(args.pretrained_path)
    model = Joiner(backbone, position_embedding)
    return model

if __name__ == '__main__':
    model = swin_tiny(pretrained_path='/home/lbc/pretrained_model/swin_tiny_patch4_window7_224.pth')
    import pdb;pdb.set_trace()
    inputs = torch.randn([1,3,320,320])
    model = model.cuda()
    inputs = inputs.cuda()
    with torch.no_grad():
        outputs = model(inputs)
    import pdb;pdb.set_trace()
