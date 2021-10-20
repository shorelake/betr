# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import dev_models.backbones as backbones


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            out_indices=(1,2,3)
            self.names = ["layer2","layer3","layer4"]
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [int(backbone.embed_dim*2**(i+1)) for i in range(len(out_indices))]
        else:
            out_indices=(3)
            self.names = ["layer4"]
            # return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [int(backbone.embed_dim*8)]
        # self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.body = backbone

    def forward(self, tensor_list: NestedTensor):
        xs, det_tokens = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in zip(self.names, xs):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out, det_tokens


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 pretrained_path: str):
        # norm_layer = FrozenBatchNorm2d
        # backbone = getattr(torchvision.models, name)(
        #     replace_stride_with_dilation=[False, False, dilation],
        #     pretrained=is_main_process(), norm_layer=norm_layer)
        # assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        assert 'ram' in name, "only support swin with ram"
        if return_interm_layers:
            backbone = getattr(backbones, name)(pretrained_path=pretrained_path, out_indices=(1,2,3))
        else:
            backbone = getattr(backbones, name)(pretrained_path=pretrained_path, out_indices=(3,))
        
        # get swin backbone
        super().__init__(backbone, train_backbone, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs, det_tokens = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos, det_tokens


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.vit_backbone, train_backbone, return_interm_layers, args.pretrained_path) # No dilation ,now support swin with ram only
    model = Joiner(backbone, position_embedding)
    return model
