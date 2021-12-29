# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
from .proposal_network import build_proposal_network

from .attention import MultiheadAttention
from loguru import logger

class DeformableTransformer(nn.Module):
    def __init__(self, proposal, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 args=None, num_classes=91,
                 # msi_sso_encoder param
                 msi_sso_encoder=None,
                 only_selfattn_dec=False, neck_decoder='def_decoder', normalize_before=False):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        self.eff_query_init = args.eff_query_init
        self.eff_specific_head = args.eff_specific_head
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers) if num_encoder_layers !=0 else None
        self.neck_decoder = neck_decoder
        if neck_decoder == 'def_decoder':
            decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, dec_n_points, only_selfattn_dec)
            self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec) if num_decoder_layers !=0 else None
        elif neck_decoder == 'cond_decoder':
            assert only_selfattn_dec == False, 'cond_decoder not support --selfattn_dec'
            assert args.with_box_refine == False, 'cond_decoder not support --with_box_refine'
            decoder_layer = ConditionalTransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = ConditionalTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, topk_kv=args.topk_kv, sample_topk_ratio=args.sample_topk_ratio)
        elif neck_decoder == 'detr_decoder':
            assert only_selfattn_dec == False, 'detr_decoder not support --selfattn_dec'
            assert args.with_box_refine == False, 'detr_decoder not support --with_box_refine'
            # assert args.topk_kv == False, 'detr_decoder not support --with_box_refine'
            decoder_layer = DETRTransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = DETRTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec,
                                            d_model=d_model, topk_kv=args.topk_kv, sample_topk_ratio=args.sample_topk_ratio)
        else:
            logger.error(f'WRONG --neck_decoder {neck_decoder}')
            raise ValueError(f'WRONG --neck_decoder {neck_decoder}')


        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.msi_sso_encoder = None
        if num_feature_levels > 1:
            if msi_sso_encoder is not None:
                logger.info(f'build --msi_sso_encoder {msi_sso_encoder}')
                msi_sso_encoder_layer = MsiSsoDeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
                self.msi_sso_encoder = MsiSsoDeformableTransformerEncoder(msi_sso_encoder_layer, 1)
                if msi_sso_encoder == 'msi_sso_p4':
                    self.sso_index = 1
                elif msi_sso_encoder == 'msi_sso_p3':
                    self.sso_index = 0
                elif msi_sso_encoder == 'msi_sso_p5':
                    self.sso_index = 2
                else:
                    logger.error(f'ERROR, WRONG --msi_sso_encoder {msi_sso_encoder}')
                    raise ValueError(f'ERROR, WRONG --msi_sso_encoder {msi_sso_encoder}')
                if self.sso_index > num_feature_levels -1:
                    logger.error(f'WRONG, --msi_sso_encoder {msi_sso_encoder}, --num_feature_levels {num_feature_levels}')
                    raise ValueError(f'WRONG, --msi_sso_encoder {msi_sso_encoder}, --num_feature_levels {num_feature_levels}')

        if two_stage:
            self.proposal = proposal
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self.init_query_from_backbone = args.init_query_from_backbone

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
            # if isinstance(m, ProposalNet):
            #     m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        if self.two_stage:
            self.proposal._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, tgt=None, targets=None):
        assert self.two_stage or query_embed is not None
        # assert self.training and targets is not None
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # # encoder
        # memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        if self.encoder is not None:
            memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        else:
            memory = src_flatten
        
        # # msi_sso_encoder
        if self.msi_sso_encoder is not None:
            memory, spatial_shapes, level_start_index, valid_ratios,lvl_pos_embed_flatten, mask_flatten = \
                self.msi_sso_encoder(memory, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, self.sso_index)
        # prepare input for decoder
        bs, _, c = memory.shape
        enc_outputs_class = None
        if self.two_stage:
            enc_outputs_class, enc_outputs_coord, reference_points, query_embed, tgt, enc_outputs_fg_class, enc_loss = \
                self.proposal(memory,mask_flatten,spatial_shapes,level_start_index,valid_ratios, targets=targets)
            init_reference_out = reference_points
        else:
            if not self.init_query_from_backbone:
                query_embed, tgt = torch.split(query_embed, c, dim=1)
                query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
                tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            else:
                query_embed = query_embed.expand(bs, -1, -1)
                tgt = tgt
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points
        # decoder
        if self.decoder is not None:
            hs, inter_references = self.decoder(tgt, reference_points, memory,
                                                spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten, lvl_pos_embed_flatten, enc_outputs_fg_class)

            inter_references_out = inter_references
        else:
            hs = None
            inter_references_out = None

        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord, enc_loss
        return hs, init_reference_out, inter_references_out, None, None, None

class MsiSsoDeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None,tgt=None):
        # self attention
        tgt2 = self.self_attn(self.with_pos_embed(tgt, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class MsiSsoDeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None, sso_index=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        sso_start = level_start_index[sso_index]
        sso_end = level_start_index[sso_index+1] if sso_index < level_start_index.shape[0] else -1
        tgt = output[:,sso_start:sso_end,:]
        ret_pos = pos[:, sso_start:sso_end, :]
        ret_spatial_shapes = spatial_shapes[sso_index].unsqueeze(0)
        ret_level_start_index = level_start_index[0].unsqueeze(0)
        ret_valid_ratios = valid_ratios[:,sso_index,:].unsqueeze(1)
        ret_padding_mask = padding_mask[:, sso_start:sso_end]
        reference_points = reference_points[:,sso_start:sso_end,:,:]
        for _, layer in enumerate(self.layers):
            tgt = layer(output, ret_pos, reference_points, spatial_shapes, level_start_index, padding_mask,tgt)
            # output[:, sso_start:sso_end,:] = tgt
        return tgt, ret_spatial_shapes,ret_level_start_index,ret_valid_ratios,ret_pos,ret_padding_mask



class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 only_selfattn_dec=False):
        super().__init__()
        self.only_selfattn_dec = only_selfattn_dec
        if not only_selfattn_dec:
            # cross attention
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # # self attention
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        # tgt2 = tgt + self.dropout2(tgt2)
        # tgt2 = self.norm2(tgt2)

        # # cross attention
        # tgt3 = self.cross_attn(tgt,
        #                        reference_points,
        #                        src, src_spatial_shapes, level_start_index, src_padding_mask)
        # tgt = tgt2 + self.dropout1(tgt3)
        # tgt = self.norm1(tgt)

        # # ffn
        # tgt = self.forward_ffn(tgt)

        # return tgt
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if not self.only_selfattn_dec:
            # cross attention
            tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                                reference_points,
                                src, src_spatial_shapes, level_start_index, src_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, src_pos_embed=None, enc_outputs_fg_class=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class ConditionalTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):
                     
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,
                is_first = False):
        if self.normalize_before:
            raise NotImplementedError
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, query_sine_embed, is_first)

class ConditionalTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=256,
                 topk_kv=False, sample_topk_ratio=1/2.):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        # self.ref_point_head = MLP(d_model, d_model, 2, 2)
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None

        # token sampler
        if topk_kv:
            self.sampler = SortSampler(sample_topk_ratio, d_model)
        else:
            self.sampler = None


        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    # def forward(self, tgt, memory,
    #             tgt_mask: Optional[Tensor] = None,
    #             memory_mask: Optional[Tensor] = None,
    #             tgt_key_padding_mask: Optional[Tensor] = None,
    #             memory_key_padding_mask: Optional[Tensor] = None,
    #             pos: Optional[Tensor] = None,
    #             query_pos: Optional[Tensor] = None):

    def forward(self, tgt, reference_points, memory, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, memory_key_padding_mask=None, pos=None, enc_outputs_fg_class=None,
                tgt_mask=None, tgt_key_padding_mask=None,memory_mask=None):
        tgt = tgt.transpose(0,1)
        memory = memory.transpose(0,1)
        pos = pos.transpose(0,1)
        query_pos = query_pos.transpose(0,1)
        # import pdb;pdb.set_trace()
        if self.sampler is not None:
            memory, sample_reg_loss, sort_confidence_topk, memory_key_padding_mask, pos = self.sampler(memory, memory_key_padding_mask, pos, enc_outputs_fg_class)

        # import pdb;pdb.set_trace()
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        # reference_points_before_sigmoid = self.ref_point_head(query_pos)    # [num_queries, batch_size, 2]
        # reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :2].transpose(0, 1)      # [num_queries, batch_size, 2]

            # For the first decoder layer, we do not apply transformation over p_s
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(output)

            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)     
            # apply transformation
            query_sine_embed = query_sine_embed * pos_transformation
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_id](self.norm(output))
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_reference_points.append(reference_points)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate).transpose(1, 2), torch.stack(intermediate_reference_points)

        return output.transpose(0,1), reference_points


class DETRTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class DETRTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=256,
                 topk_kv=False, sample_topk_ratio=1/2.):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

        # token sampler
        if topk_kv:
            self.sampler = SortSampler(sample_topk_ratio, d_model)
        else:
            self.sampler = None

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    # def forward(self, tgt, memory,
    #             tgt_mask: Optional[Tensor] = None,
    #             memory_mask: Optional[Tensor] = None,
    #             tgt_key_padding_mask: Optional[Tensor] = None,
    #             memory_key_padding_mask: Optional[Tensor] = None,
    #             pos: Optional[Tensor] = None,
    #             query_pos: Optional[Tensor] = None):
    def forward(self, tgt, reference_points, memory, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, memory_key_padding_mask=None, pos=None, enc_outputs_fg_class=None,
                tgt_mask=None, tgt_key_padding_mask=None,memory_mask=None):

        tgt = tgt.transpose(0,1)
        memory = memory.transpose(0,1)
        pos = pos.transpose(0,1)
        query_pos = query_pos.transpose(0,1)

        if self.sampler is not None:
            memory, sample_reg_loss, sort_confidence_topk, memory_key_padding_mask, pos = self.sampler(memory, memory_key_padding_mask, pos, enc_outputs_fg_class)


        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for layer_id, layer in enumerate(self.layers):
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_id](self.norm(output))
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_reference_points.append(reference_points)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate).transpose(1, 2), torch.stack(intermediate_reference_points)

        return output.transpose(0,1), reference_points



def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SortSampler(nn.Module):

    def __init__(self, topk_ratio, input_dim, kproj_net='2layer-fc', unsample_abstract_number=30,pos_embed_kproj=False):
        super().__init__()
        self.topk_ratio = topk_ratio

        self.norm_feature = nn.LayerNorm(input_dim,elementwise_affine=False)
        self.unsample_abstract_number = unsample_abstract_number
        if kproj_net == '2layer-fc':
            self.k_proj = nn.Sequential(nn.Linear(input_dim, input_dim),
                                                nn.ReLU(),
                                                nn.Linear(input_dim, unsample_abstract_number))
        elif kproj_net == '1layer-fc':
            self.k_proj = nn.Linear(input_dim, unsample_abstract_number)
        else:
            raise ValueError
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.pos_embed_kproj = pos_embed_kproj

    def forward(self, src, mask, pos_embed, enc_outputs_fg_class, sample_ratio=None):
        # bs,c ,h, w  = src.shape
        _, bs, c = src.shape
        # import pdb;pdb.set_trace()
        # sample_weight = self.score_pred_net(src.transpose(0,1)).sigmoid().view(bs,-1)
        sample_weight = enc_outputs_fg_class.sigmoid().view(bs,-1)
        # sample_weight[mask] = sample_weight[mask].clone() * 0.
        # sample_weight.data[mask] = 0.
        sample_weight_clone = sample_weight.clone().detach()
        sample_weight_clone[mask] = -1.

        if sample_ratio==None:
            sample_ratio = self.topk_ratio
        ##max sample number:
        sample_lens = ((~mask).sum(1)*sample_ratio).int()
        max_sample_num = sample_lens.max()
        mask_topk = torch.arange(max_sample_num).expand(len(sample_lens), max_sample_num).to(sample_lens.device) > (sample_lens-1).unsqueeze(1)

        ## for sampling remaining unsampled points
        min_sample_num = sample_lens.min()

        sort_order = sample_weight_clone.sort(descending=True,dim=1)[1]
        sort_confidence_topk = sort_order[:,:max_sample_num]
        sort_confidence_topk_remaining = sort_order[:,min_sample_num:]
        ## flatten for gathering
        # src = src.flatten(2).permute(2, 0, 1)
        src = self.norm_feature(src)

        src_sample_remaining = src.gather(0, sort_confidence_topk_remaining.permute(1, 0)[..., None].expand(-1, -1, c))

        ## this will maskout the padding and sampled points
        mask_unsampled = torch.arange(mask.size(1)).expand(len(sample_lens), mask.size(1)).to(sample_lens.device) < (sample_lens).unsqueeze(1)
        mask_unsampled = mask_unsampled | mask.gather(1, sort_order)
        mask_unsampled = mask_unsampled[:,min_sample_num:]

        ## abstract the unsampled points with attention
        if self.pos_embed_kproj:
            pos_embed_sample_remaining = pos_embed.gather(0, sort_confidence_topk_remaining.permute(1, 0)[..., None].expand(-1, -1, c))
            kproj = self.k_proj(src_sample_remaining+pos_embed_sample_remaining)
        else:
            kproj = self.k_proj(src_sample_remaining)
        kproj = kproj.masked_fill(
            mask_unsampled.permute(1,0).unsqueeze(2),
            float('-inf'),
        ).permute(1,2,0).softmax(-1)
        abs_unsampled_points = torch.bmm(kproj, self.v_proj(src_sample_remaining).permute(1,0,2)).permute(1,0,2)
        abs_unsampled_pos_embed = torch.bmm(kproj, pos_embed.gather(0,sort_confidence_topk_remaining.
                                                                          permute(1,0)[...,None].expand(-1,-1,c)).permute(1,0,2)).permute(1,0,2)
        abs_unsampled_mask = mask.new_zeros(mask.size(0),abs_unsampled_points.size(0))

        ## reg sample weight to be sparse with l1 loss
        sample_reg_loss = sample_weight.gather(1,sort_confidence_topk).mean()
        src_sampled = src.gather(0,sort_confidence_topk.permute(1,0)[...,None].expand(-1,-1,c)) *sample_weight.gather(1,sort_confidence_topk).permute(1,0).unsqueeze(-1)
        pos_embed_sampled = pos_embed.gather(0,sort_confidence_topk.permute(1,0)[...,None].expand(-1,-1,c))
        mask_sampled = mask_topk

        src = torch.cat([src_sampled, abs_unsampled_points])
        pos_embed = torch.cat([pos_embed_sampled,abs_unsampled_pos_embed])
        mask = torch.cat([mask_sampled, abs_unsampled_mask],dim=1)
        assert ((~mask).sum(1)==sample_lens+self.unsample_abstract_number).all()
        return src, sample_reg_loss, sort_confidence_topk, mask, pos_embed



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    logger.error(F"activation should be relu/gelu, not {activation}.")
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer_wo_encoder(args):
    if args.dataset_file == 'coco':
        num_classes = 90
    elif args.dataset_file == 'coco_panoptic':
        num_classes = 250
    else:
        num_classes = 20
    num_classes += 1
    if args.two_stage:
        logger.info(f'build proposal network')
        proposal = build_proposal_network(args)
    else:
        proposal = None
    return DeformableTransformer(
        proposal,
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage, # transformer neck w/o encoder cannot support original two stage
        two_stage_num_proposals=args.num_queries,
        args=args,
        num_classes= 1 if args.agn_proposal else num_classes,
        msi_sso_encoder=args.msi_sso_encoder,
        only_selfattn_dec=args.only_selfattn_dec,
        neck_decoder=args.neck_decoder,
        normalize_before=args.pre_norm)


