import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
from dev_models.matcher import AnchorMatcher
from util.box_ops import box_cxcywh_to_xyxy, box_area
from util import box_ops
from loguru import logger

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


# sigmoid_focal_loss_jit = torch.jit.script(
#     sigmoid_focal_loss
# )  # type: torch.jit.ScriptModule
def pairwise_intersection(boxes1, boxes2) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: intersection, sized [N,M].
    """
    # boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection
# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)
    area1 = box_area(boxes1) #[N]
    area2 = box_area(boxes2) #[M]

    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou

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
                 has_dec=True, my_enc_loss=False, has_mask_pred=False):
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

        self.mask_embed = None
        self.has_mask_pred = has_mask_pred
        if has_mask_pred:
            self.mask_embed = MaskPredictor(d_model, d_model)
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

    def forward(self, memory, mask_flatten, spatial_shapes,level_start_index,valid_ratios, targets=None):
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.class_embed(output_memory)
        enc_outputs_coord_unact = self.bbox_embed(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()

        enc_outputs_mask = None
        if self.mask_embed is not None:
            enc_outputs_mask = self.mask_embed(output_memory)
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
            return (enc_outputs_class, enc_outputs_coord, enc_outputs_mask, 
                    reference_points, query_embed, tgt, enc_outputs_fg_class,None, topk_proposals, None)
        else:
            return (enc_outputs_class, enc_outputs_coord,enc_outputs_mask, 
                    None,None,None, None,None, None, None)

class RpnDefaultProposalNet(nn.Module):
    def __init__(self, d_model=256, num_classes=91, num_proposals=300, 
                 eff_query_init=False, eff_specific_head=False,
                 has_dec=True, my_enc_loss=False, has_mask_pred=False,
                 proposal_filter=False):
        super().__init__()
        self.my_enc_loss=my_enc_loss
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
        self.filter = None
        if proposal_filter:
            self.filter = MaskPredictor(d_model, d_model)

        self.mask_embed = None
        self.has_mask_pred = has_mask_pred
        if has_mask_pred:
            self.mask_embed = MaskPredictor(d_model, d_model)
        
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

    def forward(self, memory, mask_flatten, spatial_shapes,level_start_index,valid_ratios, targets=None):
        # assert self.training and targets is not None

        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.class_embed(output_memory)
        enc_outputs_coord_unact = self.bbox_embed(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        enc_outputs_filter = None
        if self.filter is not None:
            enc_outputs_filter = self.filter(output_memory)
        enc_outputs_mask = None
        if self.mask_embed is not None:
            enc_outputs_mask = self.mask_embed(output_memory)

        if self.has_dec:
            topk = self.num_proposals
            if self.eff_specific_head:
                # take the best score for judging objectness with class specific head
                if enc_outputs_filter is not None:
                    filtered_enc_outputs_class = enc_outputs_class.sigmoid() * enc_outputs_filter.sigmoid()
                    # filtered_enc_outputs_class = enc_outputs_filter
                else:
                    filtered_enc_outputs_class = enc_outputs_class
                enc_outputs_fg_class = filtered_enc_outputs_class.topk(1, dim=2).values[... , 0]
            else:
                # take the score from the binary(fore/background) classfier 
                # though outputs have 91 output dim, the 1st dim. alone will be used for the loss computation.
                if enc_outputs_filter is not None:
                    filtered_enc_outputs_class = enc_outputs_class.sigmoid() * enc_outputs_filter.sigmoid()
                    # filtered_enc_outputs_class = enc_outputs_filter
                else:
                    filtered_enc_outputs_class = enc_outputs_class
                enc_outputs_fg_class = filtered_enc_outputs_class[..., 0]

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
            return (enc_outputs_class, enc_outputs_coord, enc_outputs_mask, 
                    reference_points, query_embed, tgt, enc_outputs_fg_class,None, topk_proposals, enc_outputs_filter)
        else:
            return (enc_outputs_class, enc_outputs_coord,enc_outputs_mask,
                    None,None,None, None,None, None, None)

class RpnDefaultAssignProposalNet(nn.Module):
    def __init__(self, d_model=256, num_classes=91, num_proposals=300, 
                 eff_query_init=False, eff_specific_head=False,
                 has_dec=True, my_enc_loss=False):
        super().__init__()
        self.my_enc_loss=my_enc_loss
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
        
        self.anchor_matcher = AnchorMatcher([0.4, 0.5],[0, -1, 1],allow_low_quality_matches=True)
        self.pre_nms_topk=2000

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
        proposals = proposals * scale
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

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]: List of #img tensors. i-th element is a vector of labels whose length is
            the total number of anchors across all feature maps (sum(Hi * Wi * A)).
            Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.

            list[Tensor]: i-th element is a Rx4 tensor, where R is the total number of anchors
            across feature maps. The values are the matched gt boxes for each anchor.
            Values are undefined for those anchors not labeled as foreground.
        """
        # anchors = Boxes.cat(anchors)  # Rx4

        gt_labels = []
        matched_gt_boxes = []
        # import pdb;pdb.set_trace()
        for anchor_per_image, gt_per_image in zip(anchors, gt_instances):
            gt_boxes = gt_per_image["boxes"]
            gt_classes = gt_per_image["labels"]
            match_quality_matrix = pairwise_iou(gt_boxes, anchor_per_image)
            matched_idxs, anchor_labels = self.anchor_matcher(match_quality_matrix)
            del match_quality_matrix

            if len(gt_boxes) > 0:
                matched_gt_boxes_i = gt_boxes[matched_idxs]

                gt_labels_i = gt_classes[matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_labels_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_labels_i[anchor_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(anchor_per_image)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    def _ema_update(self, name: str, value: float, initial_value: float, momentum: float = 0.9):
        """
        Apply EMA update to `self.name` using `value`.

        This is mainly used for loss normalizer. In Detectron1, loss is normalized by number
        of foreground samples in the batch. When batch size is 1 per GPU, #foreground has a
        large variance and using it lead to lower performance. Therefore we maintain an EMA of
        #foreground to stabilize the normalizer.

        Args:
            name: name of the normalizer
            value: the new value to update
            initial_value: the initial value to start with
            momentum: momentum of EMA

        Returns:
            float: the updated EMA value
        """
        if hasattr(self, name):
            old = getattr(self, name)
        else:
            old = initial_value
        new = old * momentum + value * (1 - momentum)
        setattr(self, name, new)
        return new
    # def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes):
    def losses(self, enc_outputs_coord, enc_outputs_class, gt_labels, gt_boxes):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor storing the loss.
                Used during training only. The dict keys are: "loss_cls" and "loss_box_reg"
        """
        loss = {}
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)
        gt_boxes = torch.stack(gt_boxes)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        # get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        normalizer = self._ema_update("loss_normalizer", max(num_pos_anchors, 1), 100)

        # classification and regression loss
        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[
            :, :-1
        ]  # no loss for the last (background) class
        loss_cls = sigmoid_focal_loss(
            enc_outputs_class[valid_mask],
            gt_labels_target.to(enc_outputs_class[0].dtype),
            alpha=0.25,
            gamma=2,
            reduction="sum",
        )
        loss['loss_ce'] = loss_cls/normalizer
        gt_boxes_target = gt_boxes[pos_mask]
        src_boxes = enc_outputs_coord[pos_mask]
        
        loss_bbox = F.l1_loss(src_boxes, gt_boxes_target, reduction='none')
        loss['loss_bbox'] = loss_bbox.sum() / normalizer

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(gt_boxes_target)))
        loss['loss_giou'] = loss_giou.sum() / normalizer

        return loss

    def predict_instances(self, enc_outputs_class, enc_outputs_coord, spatial_shapes,level_start_index, output_memory):
        # import pdb;pdb.set_trace()
        enc_outputs_class = enc_outputs_class.sigmoid()
        num_images,_,_ = enc_outputs_class.shape
        device = enc_outputs_class.device
        # if self.eff_specific_head:
        #     # take the best score for judging objectness with class specific head
        #     enc_outputs_fg_class = enc_outputs_class.topk(1, dim=2).values[... , 0]
        # else:
        #     # take the score from the binary(fore/background) classfier 
        #     # though outputs have 91 output dim, the 1st dim. alone will be used for the loss computation.
        #     enc_outputs_fg_class = enc_outputs_class[..., 0]
        _cur = 0
        pre_nms_sampled_boxes = []
        pre_nms_sampled_memory = []

        # with torch.no_grad():

        # 1. Select top-k anchor for every level and every image
        topk_scores = []  # #lvl Tensor, each of shape N x topk
        topk_labels = []
        topk_memory = []
        topk_proposals = []
        level_ids = []  # #lvl Tensor, each of shape (topk,)
        batch_idx = torch.arange(num_images, device=device)
        for lvl , (H_, W_) in enumerate(spatial_shapes):
            lvl_heatmap = enc_outputs_class[:, _cur:(_cur + H_ * W_),:] # N, HW, C
            lvl_box_reg = enc_outputs_coord[:, _cur:(_cur + H_ * W_),:]
            lvl_memory = output_memory[:, _cur:(_cur + H_ * W_),:]
            lvl_scores, lvl_labels = torch.max(lvl_heatmap,dim=2)
            Hi_Wi_A = lvl_scores.shape[1]
            if isinstance(Hi_Wi_A, torch.Tensor):  # it's a tensor in tracing
                num_proposals_i = torch.clamp(Hi_Wi_A, max=self.pre_nms_topk)
            else:
                num_proposals_i = min(Hi_Wi_A, self.pre_nms_topk)

            topk_scores_i, topk_idx = lvl_scores.topk(num_proposals_i, dim=1)

            # each is N x topk
            topk_proposals_i = lvl_box_reg[batch_idx[:, None], topk_idx]  # N x topk x 4

            topk_memory_i = lvl_memory[batch_idx[:, None], topk_idx]

            topk_labels_i = lvl_labels[batch_idx[:, None], topk_idx]

            topk_proposals.append(topk_proposals_i)
            topk_scores.append(topk_scores_i)
            topk_memory.append(topk_memory_i)
            topk_labels.append(topk_labels_i)
            level_ids.append(torch.full((num_proposals_i,), lvl, dtype=torch.int64, device=device))

        # 2. Concat all levels together
        topk_scores = torch.cat(topk_scores, dim=1)
        topk_labels = torch.cat(topk_labels, dim=1)
        topk_proposals = torch.cat(topk_proposals, dim=1)
        topk_memory = torch.cat(topk_memory, dim=1)
        level_ids = torch.cat(level_ids, dim=0)

        reference_points=[]
        tgt=[]
        # import pdb;pdb.set_trace()
        for n in range(num_images):
            boxes = topk_proposals[n]
            scores = topk_scores[n]
            labels = topk_labels[n]
            memory = topk_memory[n]
            xyxyboxes = box_ops.box_cxcywh_to_xyxy(boxes)

            keep = box_ops.batched_nms(xyxyboxes, 
                    scores, 
                    labels, 
                    0.5)
            keep = keep[:self.num_proposals]
            reference_points.append(boxes[keep])
            tgt.append(memory[keep])
        
        reference_points = torch.stack(reference_points)
        tgt = torch.stack(tgt)

        reference_points = reference_points.detach()
        query_embed = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(reference_points)))

        return reference_points, query_embed, tgt 


    def forward(self, memory, mask_flatten, spatial_shapes,level_start_index,valid_ratios, targets=None):
        # assert self.training and targets is not None

        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.class_embed(output_memory)
        enc_outputs_coord_unact = self.bbox_embed(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()

        # hack implementation for anchor matcher
        anchors = output_proposals.sigmoid()
        # import pdb;pdb.set_trace()
        enc_loss = None
        if targets is not None:
            gt_labels, gt_boxes = self.label_anchors(anchors, targets)
            # import pdb;pdb.set_trace()
            enc_loss = self.losses(enc_outputs_coord, enc_outputs_class, gt_labels, gt_boxes)
        if self.has_dec:
            if self.eff_specific_head:
                # take the best score for judging objectness with class specific head
                enc_outputs_fg_class = enc_outputs_class.topk(1, dim=2).values[... , 0]
            else:
                # take the score from the binary(fore/background) classfier 
                # though outputs have 91 output dim, the 1st dim. alone will be used for the loss computation.
                enc_outputs_fg_class = enc_outputs_class[..., 0]
            # import pdb;pdb.set_trace()
            reference_points, query_embed, tgt = self.predict_instances(enc_outputs_class, enc_outputs_coord,
                                                                        spatial_shapes,level_start_index, output_memory)
            return enc_outputs_class, enc_outputs_coord, reference_points, query_embed, tgt, enc_outputs_fg_class, enc_loss
        else:
            return enc_outputs_class, enc_outputs_coord, None,None,None, None, enc_loss


class RpnDefaultProposalNetV2(nn.Module):
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
        nn.init.normal_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        # nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], 0.0)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def bbox_transform_inv(self, proposal, bbox_deltas): # TODO
        dx = bbox_deltas[:,:,0]
        dy = bbox_deltas[:,:,1]
        dw = bbox_deltas[:,:,2]
        dh = bbox_deltas[:,:,3]

        pred_ctr_x = proposal[:,:,0] + dx
        pred_ctr_y = proposal[:,:,1] + dy
        pred_w = torch.exp(dw) * proposal[:,:,2]
        pred_h = torch.exp(dh) * proposal[:,:,3]

        # Prevent sending too large values into torch.exp()
        pred_w = torch.clamp(pred_w, max=1.)
        pred_h = torch.clamp(pred_h, max=1.)

        pred_boxes = torch.stack((pred_ctr_x,pred_ctr_y,pred_w,pred_h),dim=-1)

        return pred_boxes.reshape(bbox_deltas.shape)


    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        output_memory = []
        outputs_coord = []
        outputs_class = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask = memory_padding_mask[:, _cur:(_cur + H_ * W_)]
            mask_flatten_ = mask.view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)

            import pdb;pdb.set_trace()
            proposal_valid = ((proposal > 0.01) & (proposal < 0.99)).all(-1, keepdim=True)
            proposal =  proposal.masked_fill(mask.unsqueeze(-1), float(1.0))
            proposal = proposal.masked_fill(~proposal_valid, float(1.0))


            # proposals.append(proposal)
            memory_lvl = memory[:, _cur:(_cur + H_ * W_), :]#.view(N_,H_,W_,C_).permute(0,3,1,2) # N C H W

            memory_lvl = memory_lvl.view(N_,H_,W_,C_).permute(0,3,1,2) # N C H W
            memory_lvl = self.rpn_tower(memory_lvl) # N C H W

            memory_lvl = memory_lvl.flatten(2).permute(0,2,1) # N HW C

            bbox_deltas = self.bbox_embed(memory_lvl)
            bbox = self.bbox_transform_inv(proposal, bbox_deltas)



            outputs_coord.append(bbox)
            output_memory.append(memory_lvl)

            _cur += (H_ * W_)

        # output_proposals = torch.cat(proposals, 1)
        # output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        # output_proposals = torch.log(output_proposals / (1 - output_proposals)) # inverse sigmoid
        # output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        # output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        output_memory = torch.cat(output_memory,1)
        outputs_coord = torch.cat(outputs_coord,1)
        return output_memory, outputs_coord

    def forward(self, memory, mask_flatten, spatial_shapes,level_start_index,valid_ratios):
        bs, _, c = memory.shape
        output_memory, enc_outputs_coord = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.class_embed(output_memory)
        # enc_outputs_coord_unact = self.bbox_embed(output_memory) + output_proposals
        # enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
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
            topk_coords = torch.gather(enc_outputs_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords = topk_coords.detach()
            reference_points = topk_coords
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords)))

            if self.eff_query_init:
                # Efficient-DETR uses top-k memory as the initialization of `tgt` (query vectors)
                tgt = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.size(-1)))
                query_embed = pos_trans_out
            else:
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            return enc_outputs_class, enc_outputs_coord, reference_points, query_embed, tgt, enc_outputs_fg_class,None
        else:
            return enc_outputs_class, enc_outputs_coord, None,None,None, None,None


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
        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
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
            return enc_outputs_class, enc_outputs_coord, reference_points, query_embed, tgt, enc_outputs_fg_class,None
        else:
            return enc_outputs_class, enc_outputs_coord, None, None, None, None,None

class FcosProposalNet(nn.Module):
    def __init__(self, d_model=256, num_classes=91, num_proposals=300,
                 eff_query_init=False, eff_specific_head=False,
                 has_dec=True,my_enc_loss=False, has_mask_pred=False,
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


        self.mask_embed = None
        self.has_mask_pred = has_mask_pred
        if has_mask_pred:
            self.mask_embed = MaskPredictor(d_model, d_model)
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
        output_memory = []
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

            output_memory.append(feat1.flatten(2).permute(0,2,1))
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
        output_memory = torch.cat(output_memory,1)
        return outputs_class, outputs_coord_unact, output_memory

    def forward(self, memory, mask_flatten, spatial_shapes,level_start_index,valid_ratios, targets=None):
        bs, _, c = memory.shape
        enc_outputs_class, enc_outputs_coord_unact, output_memory = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()

        enc_outputs_mask = None
        if self.mask_embed is not None:
            enc_outputs_mask = self.mask_embed(output_memory)

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
            return (enc_outputs_class, enc_outputs_coord, enc_outputs_mask, 
                    reference_points, query_embed, tgt, enc_outputs_fg_class,None, topk_proposals)
        else:
            return (enc_outputs_class, enc_outputs_coord,enc_outputs_mask, 
                    None,None,None, None,None, None)

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

class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1)
        )
    
    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out

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
            has_dec=has_dec,
            my_enc_loss=args.my_enc_loss,
            has_mask_pred=True if args.dense_aux_loss=='dam' else False,
        )
    elif args.proposal_net == 'rpn_default':
        logger.info(f'build rpn default proposal net')
        return RpnDefaultProposalNet(
            d_model=args.hidden_dim,
            num_classes= 1 if args.agn_proposal else num_classes,
            num_proposals=args.num_queries,
            eff_query_init=args.eff_query_init,
            eff_specific_head=args.eff_specific_head,
            has_dec=has_dec,
            my_enc_loss=args.my_enc_loss,
            has_mask_pred=True if args.dense_aux_loss=='dam' else False,
            proposal_filter=args.proposal_filter
        )
    elif args.proposal_net == 'rpn_default_assign':
        logger.error(f'build rpn_default_assign, not IMPLEMENT YET!')
        raise ValueError(f'build rpn_default_assign, not IMPLEMENT YET!')
        return RpnDefaultAssignProposalNet(
            d_model=args.hidden_dim,
            num_classes= 1 if args.agn_proposal else num_classes,
            num_proposals=args.num_queries,
            eff_query_init=args.eff_query_init,
            eff_specific_head=args.eff_specific_head,
            has_dec=has_dec,
            my_enc_loss=args.my_enc_loss
        )
    elif args.proposal_net == 'rpn_default_v2': # TODO
        logger.error(f'build rpn default proposal v2 net, not IMPLEMENT YET!')
        raise ValueError(f'build rpn default proposal v2 net, not IMPLEMENT YET!')
        return RpnDefaultProposalNetV2(
            d_model=args.hidden_dim,
            num_classes= 1 if args.agn_proposal else num_classes,
            num_proposals=args.num_queries,
            eff_query_init=args.eff_query_init,
            eff_specific_head=args.eff_specific_head,
            has_dec=has_dec,
            my_enc_loss=args.my_enc_loss
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
            has_dec=has_dec,
            my_enc_loss=args.my_enc_loss
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
            has_dec=has_dec,
            my_enc_loss=args.my_enc_loss,
            has_mask_pred=True if args.dense_aux_loss=='dam' else False,
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
            has_dec=has_dec,
            my_enc_loss=args.my_enc_loss,
            has_mask_pred=True if args.dense_aux_loss=='dam' else False,
        )