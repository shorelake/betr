import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
from loguru import logger

def _broadcast_params(params, num_features, name):
    """
    If one size (or aspect ratio) is specified and there are multiple feature
    maps, we "broadcast" anchors of that single size (or aspect ratio)
    over all feature maps.

    If params is list[float], or list[list[float]] with len(params) == 1, repeat
    it num_features time.

    Returns:
        list[list[float]]: param for each feature
    """
    assert isinstance(
        params, (list, tuple)
    ), f"{name} in anchor generator has to be a list! Got {params}."
    assert len(params), f"{name} in anchor generator cannot be empty!"
    if not isinstance(params[0], (list, tuple)):  # list[float]
        return [params] * num_features
    if len(params) == 1:
        return list(params) * num_features
    assert len(params) == num_features, (
        f"Got {name} of length {len(params)} in anchor generator, "
        f"but the number of input features is {num_features}!"
    )
    return params

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers):
        super(BufferList, self).__init__()
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(i), buffer)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

class DefaultProposalNet(nn.Module):
    def __init__(self, d_model=256, num_classes=91, num_proposals=300, 
                 eff_query_init=False, eff_specific_head=False,
                 has_dec=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.eff_query_init = eff_query_init
        self.eff_specific_head = eff_specific_head
        
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.pos_trans = nn.Linear(d_model * 2, d_model * (1 if self.eff_query_init else 2))
        self.pos_trans_norm = nn.LayerNorm(d_model * (1 if self.eff_query_init else 2))

        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        
        self.has_dec = has_dec
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], 0.0)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals)) # inverse sigmoid
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        return output_memory, output_proposals

    def forward(self, memory, mask_flatten, spatial_shapes,level_start_index,valid_ratios):
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.class_embed(output_memory)
        enc_outputs_coord_unact = self.bbox_embed(output_memory) + output_proposals
        if self.has_dec:
            topk = self.num_proposals
            if self.eff_specific_head:
                # take the best score for judging objectness with class specific head
                enc_outputs_fg_class = enc_outputs_class.topk(1, dim=2).values[... , 0]
            else:
                # take the score from the binary(fore/background) classfier 
                # though outputs have 91 output dim, the 1st dim. alone will be used for the loss computation.
                enc_outputs_fg_class = enc_outputs_class[..., 0]

            topk_proposals = torch.topk(enc_outputs_fg_class, topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))

            if self.eff_query_init:
                # Efficient-DETR uses top-k memory as the initialization of `tgt` (query vectors)
                tgt = torch.gather(memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, memory.size(-1)))
                query_embed = pos_trans_out
            else:
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            return enc_outputs_class, enc_outputs_coord_unact, reference_points, query_embed, tgt
        else:
            return enc_outputs_class, enc_outputs_coord_unact, None, None, None

class RpnDefaultProposalNet(nn.Module):
    def __init__(self, d_model=256, num_classes=91, num_proposals=300, 
                 eff_query_init=False, eff_specific_head=False,
                 has_dec=True):
        super().__init__()
        self.has_dec=has_dec
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.eff_query_init = eff_query_init
        self.eff_specific_head = eff_specific_head
        
        # self.enc_output = nn.Linear(d_model, d_model)
        # self.enc_output_norm = nn.LayerNorm(d_model)

        self.rpn_tower = nn.Sequential(
            nn.Conv2d(d_model,d_model,kernel_size=3,stride=1,padding=1,bias=True),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True)
        )


        self.pos_trans = nn.Linear(d_model * 2, d_model * (1 if self.eff_query_init else 2))
        self.pos_trans_norm = nn.LayerNorm(d_model * (1 if self.eff_query_init else 2))

        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.01)
                torch.nn.init.constant_(m.bias, 0)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], 0.0)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        output_memory = []
        outputs_coord_unact = []
        outputs_class = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            memory_lvl = memory[:, _cur:(_cur + H_ * W_), :]#.view(N_,H_,W_,C_).permute(0,3,1,2) # N C H W

            memory_lvl = memory_lvl.view(N_,H_,W_,C_).permute(0,3,1,2) # N C H W
            memory_lvl = self.rpn_tower(memory_lvl) # N C H W

            output_memory.append(memory_lvl.flatten(2).permute(0,2,1))

            _cur += (H_ * W_)

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals)) # inverse sigmoid
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        output_memory = torch.cat(output_memory,1)
        return output_memory, output_proposals

    def forward(self, memory, mask_flatten, spatial_shapes,level_start_index,valid_ratios):
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.class_embed(output_memory)
        enc_outputs_coord_unact = self.bbox_embed(output_memory) + output_proposals
        if self.has_dec:
            topk = self.num_proposals
            if self.eff_specific_head:
                # take the best score for judging objectness with class specific head
                enc_outputs_fg_class = enc_outputs_class.topk(1, dim=2).values[... , 0]
            else:
                # take the score from the binary(fore/background) classfier 
                # though outputs have 91 output dim, the 1st dim. alone will be used for the loss computation.
                enc_outputs_fg_class = enc_outputs_class[..., 0]

            topk_proposals = torch.topk(enc_outputs_fg_class, topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))

            if self.eff_query_init:
                # Efficient-DETR uses top-k memory as the initialization of `tgt` (query vectors)
                tgt = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.size(-1)))
                query_embed = pos_trans_out
            else:
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            return enc_outputs_class, enc_outputs_coord_unact, reference_points, query_embed, tgt
        else:
            return enc_outputs_class, enc_outputs_coord_unact, None,None,None


class RetinaProposalNet(nn.Module):
    def __init__(self, d_model=256, num_classes=91, num_proposals=300,
                 eff_query_init=False, eff_specific_head=False,
                 has_dec=True,
                 num_feature_levels=4, strides = [8, 16, 32, 64],
                 # anchor param
                 ):
        super().__init__()
        self.has_dec = has_dec
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.eff_query_init = eff_query_init
        self.eff_specific_head = eff_specific_head
        
        num_conv = 2
        share_tower = []
        for idx in range(num_conv):
            share_tower.append(nn.Conv2d(d_model,d_model,kernel_size=3,stride=1,padding=1,bias=True))
            share_tower.append(nn.GroupNorm(32, d_model))
            share_tower.append(nn.ReLU(inplace=True))
        self.share_tower = nn.Sequential(*share_tower)



        # Init parameters.
        prior_prob = 0.01
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)

        # anchor parameters
        anchor_sizes = [[x, x * 2**(1.0/3), x * 2**(2.0/3) ] for x in [32, 64, 128, 256]]
        aspect_ratios= [[0.5,1.0,2.0]]
        anchor_sizes = _broadcast_params(anchor_sizes,num_feature_levels, "sizes")
        aspect_ratios = _broadcast_params(aspect_ratios, num_feature_levels, "aspect_ratios")

        self.cell_anchors = self._calculate_anchors(anchor_sizes, aspect_ratios) 
        num_anchors = [len(cell_anchors) for cell_anchors in self.cell_anchors]

        self.num_anchors = num_anchors[0]
        self.cls_score = nn.Conv2d(d_model, self.num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(d_model, self.num_anchors * 4, kernel_size=3, stride=1, padding=1)

        self.pos_trans = nn.Linear(d_model * 2, d_model * (1 if self.eff_query_init else 2))
        self.pos_trans_norm = nn.LayerNorm(d_model * (1 if self.eff_query_init else 2))
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.01)
                torch.nn.init.constant_(m.bias, 0)

        # initialize the bias for focal loss.
        nn.init.constant_(self.cls_score.bias, self.bias_value)
        # nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        # nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], 0.0)

    def _calculate_anchors(self, sizes, aspect_ratios):
        cell_anchors = [
            self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
        ]
        return BufferList(cell_anchors)

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).

        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 2) storing anchor boxes
                in WH format.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227

        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                # x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([w,h])
        return torch.tensor(anchors)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos  

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        memorys = []
        outputs_class = []
        outputs_coord_unact = []
        _cur = 0
        buffers: List[torch.Tensor] = [x[1] for x in self.cell_anchors.named_buffers()]
        for lvl, (anchor_wh,(H_, W_)) in enumerate(zip(buffers,spatial_shapes)):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            # import pdb;pdb.set_trace()
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1).unsqueeze(3) + 0.5) / scale
            anchor_wh = anchor_wh.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(N_,-1,-1,-1,-1) / scale
            num_anchor = anchor_wh.shape[3]
            n, h, w, _, _ = grid.shape
            grid = grid.expand(-1,-1,-1,num_anchor,-1)
            anchor_wh = anchor_wh.expand(n,h,w,-1,-1)
            proposal = torch.cat((grid, anchor_wh), -1).view(N_, -1, 4) # N_, HWA, 4

            proposal_valid = ((proposal > 0.01) & (proposal < 0.99)).all(-1, keepdim=True)

            proposal = torch.log(proposal / (1 - proposal)) # inverse sigmoid
            mask_flatten_ = mask_flatten_.unsqueeze(3).expand(-1,-1,-1,num_anchor,-1)
            proposal = proposal.masked_fill(mask_flatten_.flatten(1).unsqueeze(-1), float('inf'))
            proposal = proposal.masked_fill(~proposal_valid, float('inf'))
            proposals.append(proposal)

            memory_lvl = memory[:, _cur:(_cur + H_ * W_), :].view(N_,H_,W_,C_).permute(0,3,1,2) # N C H W
            feat1 = self.share_tower(memory_lvl)

            class_logits = self.cls_score(feat1) # N AxC H W
            pred_offset = self.bbox_pred(feat1) # N Ax4 H W

            class_logits = class_logits.permute(0,2,3,1).reshape(N_,-1,self.num_classes) # N HWA C
            pred_offset = pred_offset.permute(0,2,3,1).reshape(N_,-1,4) # N HWA 4

            # class_logits = class_logits.masked_fill(mask_flatten_.flatten(1).unsqueeze(-1), float('-inf'))
            # class_logits = class_logits.masked_fill(~proposal_valid, float('-inf'))

            # pred_offset = pred_offset.masked_fill(mask_flatten_.flatten(1).unsqueeze(-1), float('inf'))
            # pred_offset = pred_offset.masked_fill(~proposal_valid, float('inf'))
            pred_unact_coord = pred_offset + proposal

            outputs_coord_unact.append(pred_unact_coord)
            outputs_class.append(class_logits)
            
            _cur += (H_ * W_)

        output_proposals = torch.cat(proposals, 1)
        outputs_class = torch.cat(outputs_class, 1)
        outputs_coord_unact = torch.cat(outputs_coord_unact, 1)
        return outputs_class, outputs_coord_unact

    def forward(self, memory, mask_flatten, spatial_shapes,level_start_index,valid_ratios):
        bs, _, c = memory.shape
        enc_outputs_class, enc_outputs_coord_unact = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        if self.has_dec:
            topk = self.num_proposals
            if self.eff_specific_head:
                # take the best score for judging objectness with class specific head
                enc_outputs_fg_class = enc_outputs_class.topk(1, dim=2).values[... , 0]
            else:
                # take the score from the binary(fore/background) classfier 
                # though outputs have 91 output dim, the 1st dim. alone will be used for the loss computation.
                enc_outputs_fg_class = enc_outputs_class[..., 0]

            topk_proposals = torch.topk(enc_outputs_fg_class, topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))

            if self.eff_query_init:
                # Efficient-DETR uses top-k memory as the initialization of `tgt` (query vectors)
                # import pdb;pdb.set_trace() # TODO back to resolution
                memory_topk = topk_proposals // self.num_anchors
                tgt = torch.gather(memory, 1, memory_topk.unsqueeze(-1).repeat(1, 1, memory.size(-1)))
                query_embed = pos_trans_out
            else:
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            return enc_outputs_class, enc_outputs_coord_unact, reference_points, query_embed, tgt
        else:
            return enc_outputs_class, enc_outputs_coord_unact, None, None, None

class FcosProposalNet(nn.Module):
    def __init__(self, d_model=256, num_classes=91, num_proposals=300,
                 eff_query_init=False, eff_specific_head=False,
                 has_dec=True,
                 num_feature_levels=4, strides = [8, 16, 32, 64],
                 # anchor param
                 ):
        super().__init__()
        self.has_dec=has_dec
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.eff_query_init = eff_query_init
        self.eff_specific_head = eff_specific_head
        
        num_conv = 2
        share_tower = []
        for idx in range(num_conv):
            share_tower.append(nn.Conv2d(d_model,d_model,kernel_size=3,stride=1,padding=1,bias=True))
            share_tower.append(nn.GroupNorm(32, d_model))
            share_tower.append(nn.ReLU(inplace=True))
        self.share_tower = nn.Sequential(*share_tower)



        # Init parameters.
        prior_prob = 0.01
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)

        self.cls_score = nn.Conv2d(d_model, num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(d_model, 4, kernel_size=3, stride=1, padding=1)

        self.pos_trans = nn.Linear(d_model * 2, d_model * (1 if self.eff_query_init else 2))
        self.pos_trans_norm = nn.LayerNorm(d_model * (1 if self.eff_query_init else 2))
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.01)
                torch.nn.init.constant_(m.bias, 0)

        # initialize the bias for focal loss.
        nn.init.constant_(self.cls_score.bias, self.bias_value)
        # nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        # nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], 0.0)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos  

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        memorys = []
        outputs_class = []
        outputs_coord_unact = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            # import pdb;pdb.set_trace()
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            # import pdb;pdb.set_trace()
            n, h, w, _ = grid.shape

            locations = grid.view(N_, -1, 2) # N_, HWA, 4

            locations_valid = ((locations > 0.01) & (locations < 0.99)).all(-1, keepdim=True)

            locations = torch.log(locations / (1 - locations)) # inverse sigmoid
            locations = locations.masked_fill(mask_flatten_.flatten(1).unsqueeze(-1), float('inf'))
            locations = locations.masked_fill(~locations_valid, float('inf'))
            proposals.append(locations)

            memory_lvl = memory[:, _cur:(_cur + H_ * W_), :].view(N_,H_,W_,C_).permute(0,3,1,2) # N C H W
            feat1 = self.share_tower(memory_lvl)

            class_logits = self.cls_score(feat1) # N C H W
            pred_offset = self.bbox_pred(feat1) # N 4 H W

            class_logits = class_logits.permute(0,2,3,1).reshape(N_,-1,self.num_classes) # N HW C
            pred_offset = pred_offset.permute(0,2,3,1).reshape(N_,-1,4) # N HW 4

            # class_logits = class_logits.masked_fill(mask_flatten_.flatten(1).unsqueeze(-1), float('-inf'))
            # class_logits = class_logits.masked_fill(~locations_valid, float('-inf'))
            pred_offset = pred_offset.masked_fill(mask_flatten_.flatten(1).unsqueeze(-1), float('inf'))
            pred_offset = pred_offset.masked_fill(~locations_valid, float('inf'))
            # import pdb;pdb.set_trace()
            pred_offset[...,:2] += locations
            pred_unact_coord = pred_offset

            outputs_coord_unact.append(pred_unact_coord)
            outputs_class.append(class_logits)
            
            _cur += (H_ * W_)

        # output_proposals = torch.cat(proposals, 1)
        outputs_class = torch.cat(outputs_class, 1)
        outputs_coord_unact = torch.cat(outputs_coord_unact, 1)
        return outputs_class, outputs_coord_unact

    def forward(self, memory, mask_flatten, spatial_shapes,level_start_index,valid_ratios):
        bs, _, c = memory.shape
        enc_outputs_class, enc_outputs_coord_unact = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        if self.has_dec:
            topk = self.num_proposals
            if self.eff_specific_head:
                # take the best score for judging objectness with class specific head
                enc_outputs_fg_class = enc_outputs_class.topk(1, dim=2).values[... , 0]
            else:
                # take the score from the binary(fore/background) classfier 
                # though outputs have 91 output dim, the 1st dim. alone will be used for the loss computation.
                enc_outputs_fg_class = enc_outputs_class[..., 0]

            topk_proposals = torch.topk(enc_outputs_fg_class, topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))

            if self.eff_query_init:
                # Efficient-DETR uses top-k memory as the initialization of `tgt` (query vectors)
                memory_topk = topk_proposals
                tgt = torch.gather(memory, 1, memory_topk.unsqueeze(-1).repeat(1, 1, memory.size(-1)))
                query_embed = pos_trans_out
            else:
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            return enc_outputs_class, enc_outputs_coord_unact, reference_points, query_embed, tgt
        else:
            return enc_outputs_class, enc_outputs_coord_unact, None, None, None

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


def build_proposal_network(args):
    if args.dataset_file == 'coco':
        num_classes = 90
    elif args.dataset_file == 'coco_panoptic':
        num_classes = 250
    else:
        num_classes = 20
    num_classes += 1
    has_dec = False if args.dec_layers == 0 else True
    if args.proposal_net == 'default':
        logger.info(f'build default proposal net')
        return DefaultProposalNet(
            d_model=args.hidden_dim,
            num_classes= 1 if args.agn_proposal else num_classes,
            num_proposals=args.num_queries,
            eff_query_init=args.eff_query_init,
            eff_specific_head=args.eff_specific_head,
            has_dec=has_dec
        )
    elif args.proposal_net == 'rpn_default':
        logger.info(f'build rpn default proposal net')
        return RpnDefaultProposalNet(
            d_model=args.hidden_dim,
            num_classes= 1 if args.agn_proposal else num_classes,
            num_proposals=args.num_queries,
            eff_query_init=args.eff_query_init,
            eff_specific_head=args.eff_specific_head,
            has_dec=has_dec
        )
    elif args.proposal_net == 'rpn':
        logger.error(f'build rpn default proposal net, not IMPLEMENT YET!')
        raise ValueError(f'build fcos proposal net, NOT IMPLEMENT YET!')
        return RpnProposalNet(
            d_model=args.hidden_dim,
            num_classes= 1 if args.agn_proposal else num_classes,
            num_proposals=args.num_queries,
            eff_query_init=args.eff_query_init,
            eff_specific_head=args.eff_specific_head,
            has_dec=has_dec
        )
    elif args.proposal_net == 'fcos':
        # TODO
        logger.info(f'build fcos proposal net')
        return FcosProposalNet(
            d_model=args.hidden_dim,
            num_classes= 1 if args.agn_proposal else num_classes,
            num_proposals=args.num_queries,
            eff_query_init=args.eff_query_init,
            eff_specific_head=args.eff_specific_head,
            has_dec=has_dec
        )
    elif args.proposal_net == 'retina':
        # TODO
        logger.info(f'build retina proposal net')
        # raise ValueError(f'build retina proposal net, NOT IMPLEMENT YET!')
        return RetinaProposalNet(
            d_model=args.hidden_dim,
            num_classes= 1 if args.agn_proposal else num_classes,
            num_proposals=args.num_queries,
            eff_query_init=args.eff_query_init,
            eff_specific_head=args.eff_specific_head,
            has_dec=has_dec
        )