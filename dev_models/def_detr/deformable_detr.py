# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from util.dam import attn_map_to_flat_grid
from util.distillation import knowledge_distillation_kl_div_loss
# from .backbone import build_backbone as build_swin_backbone
from dev_models.backbone_factory import build_backbone
from dev_models.matcher import (build_matcher, HungarianMatcher, build_dense_matcher, 
                            build_dense_aux_matcher)
from dev_models.segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from .deformable_transformer_wo_encoder import build_deforamble_transformer_wo_encoder
import copy
from models.cnn_necks import build_cnn_encoder
from models import cnn_necks


from typing import List
from loguru import logger

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, args=None, cnn_neck=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # Defdetr use sigmoid+bce, rather than softmax+ce, so, class_embed
        # here is not the same as detr, refers to:
        # https://github.com/fundamentalvision/Deformable-DETR/issues/72#issuecomment-886408142
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.cnn_encoder = None
        if not two_stage and not args.init_query_from_backbone:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        
        self.strides = backbone.strides
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            if cnn_neck is None:
                for _ in range(num_backbone_outs):
                    in_channels = backbone.num_channels[_]
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
            else:
                in_channels_list = []
                for _ in range(num_backbone_outs):
                    in_channels = backbone.num_channels[_]
                    in_channels_list.append(in_channels)
                self.cnn_encoder = build_cnn_encoder(cnn_neck, num_backbone_outs, in_channels_list, hidden_dim)
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
                self.strides.append(self.strides[-1]*2)
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        if transformer.decoder is not None:
            self.has_dec = True
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.class_embed.bias.data = torch.ones(num_classes) * bias_value
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)

            # if two-stage, the last class_embed and bbox_embed is for region proposal generation
            # num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
            num_pred = transformer.decoder.num_layers
            if with_box_refine:
                self.class_embed = _get_clones(self.class_embed, num_pred)
                self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
                nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
                # hack implementation for iterative bounding box refinement
                self.transformer.decoder.bbox_embed = self.bbox_embed
            else:
                nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
                self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
                self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
                self.transformer.decoder.bbox_embed = None
            if two_stage:
                # hack implementation for two-stage
                self.transformer.decoder.class_embed = self.class_embed
                for box_embed in self.bbox_embed:
                    nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        else:
            self.has_dec = False
            assert two_stage
            assert args.agn_proposal == False, "0 dec should with class specific proposal"
        if args.no_input_proj:
            self.input_proj = nn.ModuleList([nn.Identity() for _ in range(len(self.input_proj))])

        self.init_query_from_backbone = args.init_query_from_backbone

    def forward(self, samples):
        """ The forward expects a List, which consists of:
               - samples[0]  NestedTensor
                - samples[0].tensor: batched images, of shape [batch_size x 3 x H x W]
                - samples[0].mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
               - samples[1]=targets Dict

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples[0], NestedTensor):
            samples[0] = nested_tensor_from_tensor_list(samples[0])
        targets = None
        if len(samples) == 2:
            targets = samples[1]
        if not self.init_query_from_backbone:
            features, pos = self.backbone(samples[0])
        else:
            features, pos, det_tokens, det_pos = self.backbone(samples[0])
        if self.cnn_encoder is None:
            srcs = []
            masks = []
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src))
                masks.append(mask)
                assert mask is not None
            if self.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        src = self.input_proj[l](srcs[-1])
                    m = samples[0].mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    pos.append(pos_l)
        else:
            srcs = []
            masks = []
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(src)
                masks.append(mask)
                assert mask is not None
            srcs = self.cnn_encoder(srcs)
            if self.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                input_proj_index = 0
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[input_proj_index](features[-1].tensors)
                    else:
                        src = self.input_proj[input_proj_index](srcs[-1])
                    m = samples[0].mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    pos.append(pos_l)
                    input_proj_index = input_proj_index + 1
        query_embeds = None
        if not self.two_stage and not self.init_query_from_backbone:
            query_embeds = self.query_embed.weight
        if not self.init_query_from_backbone:
            (hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord, enc_outputs_mask, 
             enc_loss, spatial_shapes,
             level_start_index, sampling_locations_dec, attn_weights_dec, mask_flatten, topk_proposal,
             enc_outputs_filter) = \
                self.transformer(srcs, masks, pos, query_embeds, targets=targets)
        else:
            (hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord, enc_outputs_mask,
             enc_loss, spatial_shapes,
             level_start_index, sampling_locations_dec, attn_weights_dec, mask_flatten, topk_proposal,
             enc_outputs_filter) = \
                self.transformer(srcs, masks, pos, query_embed=det_pos, tgt=det_tokens, targets=targets)
        if self.has_dec:
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if self.training:
                    if lvl == 0:
                        reference = init_reference
                    else:
                        reference = inter_references[lvl - 1]
                    reference = inverse_sigmoid(reference)
                    outputs_class = self.class_embed[lvl](hs[lvl])
                    tmp = self.bbox_embed[lvl](hs[lvl])
                    if reference.shape[-1] == 4:
                        tmp += reference
                    else:
                        assert reference.shape[-1] == 2
                        tmp[..., :2] += reference
                    outputs_coord = tmp.sigmoid()
                else:
                    if self.with_box_refine:
                        outputs_class = self.class_embed[lvl](hs[lvl])
                        outputs_coord = inter_references[lvl]
                    else:
                        if lvl == 0:
                            reference = init_reference
                        else:
                            reference = inter_references[lvl - 1]
                        reference = inverse_sigmoid(reference)
                        outputs_class = self.class_embed[lvl](hs[lvl])
                        tmp = self.bbox_embed[lvl](hs[lvl])
                        if reference.shape[-1] == 4:
                            tmp += reference
                        else:
                            assert reference.shape[-1] == 2
                            tmp[..., :2] += reference
                        outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            outputs_class = torch.stack(outputs_classes)
            outputs_coord = torch.stack(outputs_coords)

            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

            if self.two_stage:
                # enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
                out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord, 
                                      'pred_filters': enc_outputs_filter,'spatial_shapes': spatial_shapes,
                                      'strides':self.strides, 'pred_mask':enc_outputs_mask, 'sampling_locations_dec': sampling_locations_dec,
                                      'attn_weights_dec': attn_weights_dec, 'level_start_index':level_start_index,
                                       'mask_flatten': mask_flatten, 'topk_proposal': topk_proposal}
            
        else:
            # enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        
        # if self.training:
        #     loss_dict = self.criterion(out, targets)
        #     # import pdb;pdb.set_trace()
        #     if enc_loss is not None:
        #         enc_loss = {k + f'_enc': v for k, v in enc_loss.items()}
        #         loss_dict.update(enc_loss)
        #     return out, loss_dict
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, enc_matcher, enc_aux_matcher, weight_dict, losses, eff_specific_head=False, 
                 focal_alpha=0.25, my_enc_loss=False, dense_aux_loss=None, kd_from_dec=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            eff_specific_head: two stage enc class specific loss
            focal_alpha: alpha in Focal Loss
            my_enc_loss: if support my own label assign loss for two stage proposal network
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.enc_matcher = enc_matcher
        self.enc_aux_matcher = enc_aux_matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.eff_specific_head = eff_specific_head
        self.my_enc_loss = my_enc_loss
        self.dense_aux_loss = dense_aux_loss
        self.kd_from_dec = kd_from_dec

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, enc_outputs=False, dense_loss=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        if dense_loss:
            filters = outputs['pred_filters']
            src_logits = src_logits.sigmoid() * filters.sigmoid()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        if not enc_outputs:
            num_classes = self.num_classes
        else:
            num_classes = src_logits.shape[-1]
        target_classes = torch.full(src_logits.shape[:2], num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2, dense_loss=dense_loss) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
    # only for two stage dense part
    def loss_mask_prediction(self, outputs):
        assert "pred_mask" in outputs
        assert "sampling_locations_dec" in outputs
        assert "attn_weights_dec" in outputs
        assert "spatial_shapes" in outputs
        assert "level_start_index" in outputs
        assert "mask_flatten" in outputs

        mask_prediction = outputs["pred_mask"].squeeze(-1)
        loss_key = "loss_mask_pred"

        sampling_locations_dec = outputs["sampling_locations_dec"]
        attn_weights_dec = outputs["attn_weights_dec"]
        spatial_shapes = outputs["spatial_shapes"]
        level_start_index = outputs["level_start_index"]

        flat_grid_attn_map_dec = attn_map_to_flat_grid(
            spatial_shapes, level_start_index, sampling_locations_dec, attn_weights_dec).sum(dim=(1,2))

        losses = {}

        if 'mask_flatten' in outputs:
            flat_grid_attn_map_dec = flat_grid_attn_map_dec.masked_fill(
                outputs['mask_flatten'], flat_grid_attn_map_dec.min()-1)
        valid_token_num = (~ outputs['mask_flatten']).sum(axis=-1)
        ratio = 0.2
        sparse_token_nums = (valid_token_num*ratio).int()+1
        # sparse_token_nums = outputs["sparse_token_nums"]
        num_topk = sparse_token_nums.max()

        topk_idx_tgt = torch.topk(flat_grid_attn_map_dec, num_topk)[1]
        target = torch.zeros_like(mask_prediction)
        for i in range(target.shape[0]):
            target[i].scatter_(0, topk_idx_tgt[i][:sparse_token_nums[i]], 1)

        losses.update({loss_key: F.multilabel_soft_margin_loss(mask_prediction, target)})

        return losses

    def loss_kd_from_dec(self, enc_outputs, outputs):
        assert "topk_proposal" in enc_outputs
        assert "pred_logits" in enc_outputs
        assert "pred_filters" in enc_outputs
        topk_proposal = enc_outputs['topk_proposal']
        enc_logits = enc_outputs['pred_logits']
        enc_filters = enc_outputs['pred_filters']
        if enc_filters is not None:
            enc_cls = enc_logits.sigmoid() * enc_filters.sigmoid()
        else:
            enc_cls = enc_logits.sigmoid()

        enc_topk_cls = torch.gather(enc_cls,1,topk_proposal.unsqueeze(-1).repeat(1,1,enc_cls.size(-1)))
        # enc_topk_logits = inverse_sigmoid(enc_topk_cls)

        pred_cls=outputs['pred_logits'].sigmoid().detach()

        loss_module = nn.BCELoss()


        losses = {"loss_dec_kd": loss_module(enc_topk_cls, pred_cls)}

        return losses
    def loss_ious(self, outputs, targets, indices, num_boxes, log=True, enc_outputs=False, dense_loss=False):
        assert outputs['pred_filters'] is not None
        idx = self._get_src_permutation_idx(indices)
        src_ious = outputs['pred_filters'][idx]
        src_ious = src_ious.squeeze(1)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        iou = torch.diag(box_ops.box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))[0])

        losses = {}
        loss_iouaware = F.binary_cross_entropy_with_logits(src_ious, iou, reduction='none')
        losses['loss_iouaware'] = loss_iouaware.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if not self.my_enc_loss:
            if 'enc_outputs' in outputs:
                enc_outputs = outputs['enc_outputs']
                bin_targets = copy.deepcopy(targets)
                if not self.eff_specific_head:
                    for bt in bin_targets:
                        bt['labels'] = torch.zeros_like(bt['labels'])
                indices = self.enc_matcher(enc_outputs, bin_targets)

                # for normal indices loss
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                        kwargs['enc_outputs'] = True
                        if enc_outputs['pred_filters'] is not None:
                            kwargs['dense_loss'] = True
                    l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if self.kd_from_dec:
                    kd_l_dict = self.loss_kd_from_dec(enc_outputs, outputs)
                    kd_l_dict = {k + f'_enc': v for k, v in kd_l_dict.items()}
                    losses.update(kd_l_dict)

                if self.dense_aux_loss is not None:
                    if self.enc_aux_matcher is not None:
                        aux_indices = self.enc_aux_matcher(enc_outputs, bin_targets)
                        # for aux indices loss
                        # Compute the average number of foreground boxes accross all nodes, for normalization purposes
                        aux_num_boxes = sum(len(aux_indice[1]) for aux_indice in aux_indices)
                        aux_num_boxes = torch.as_tensor([aux_num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
                        if is_dist_avail_and_initialized():
                            torch.distributed.all_reduce(aux_num_boxes)
                        aux_num_boxes = torch.clamp(aux_num_boxes / get_world_size(), min=1).item()
                        aux_kwargs = {}
                        aux_kwargs['log'] = False
                        aux_kwargs['enc_outputs'] = True
                        aux_l_dict = self.loss_labels(enc_outputs, bin_targets, aux_indices, aux_num_boxes, **aux_kwargs)
                        aux_l_dict = {k + f'_enc_aux': v for k, v in aux_l_dict.items()}
                        import pdb;pdb.set_trace()
                        aux_l_dict = self.loss_ious(enc_outputs, bin_targets, aux_indices, aux_num_boxes, **aux_kwargs)
                        aux_l_dict = {k + f'_enc_aux': v for k, v in aux_l_dict.items()}
                        losses.update(aux_l_dict)
                    else:
                        aux_l_dict = self.loss_mask_prediction(enc_outputs)
                        aux_l_dict = {k + f'_enc_aux': v for k, v in aux_l_dict.items()}
                        losses.update(aux_l_dict)
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results
    @torch.no_grad()
    def nms_forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_boxes = 100
        # import pdb;pdb.set_trace()
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        results = []
        prob = out_logits.sigmoid()
        for i, (scores_per_image, box_pred_per_image,image_size) in enumerate(zip(prob,out_bbox,target_sizes)):
            boxes = box_ops.box_cxcywh_to_xyxy(box_pred_per_image)
            h,w = image_size
            scale_fct = torch.tensor([w,h,w,h], device=target_sizes.device)
            boxes = boxes * scale_fct

            scores, labels = torch.max(scores_per_image,dim=1)
            keep = box_ops.batched_nms(boxes, 
                    scores, 
                    labels, 
                    0.5)
            i = keep[:num_boxes]
            boxes = boxes[i]
            scores = scores[i]
            labels = labels[i]
            results.append({'scores':scores, 'labels':labels, 'boxes':boxes})

        return results


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


def build(args):
    if args.dataset_file == 'coco':
        num_classes = 90
    elif args.dataset_file == 'coco_panoptic':
        num_classes = 250
    else:
        num_classes = 20
    num_classes += 1
    device = torch.device(args.device)
    logger.info(f"building vit backbone {args.vit_backbone}")
    # backbone = build_swin_backbone(args)
    backbone = build_backbone(args)
    logger.info(f"build tranformer with {args.dec_layers} {args.neck_decoder}")
    if args.enc_layers == 0:
        if args.cross_update:
            logger.info("build tranformer neck without encoder, but decoder query & memory cross update")
            from .deformable_transformer_wo_encoder_cross_update import build_deforamble_transformer_wo_encoder_cross_update
            transformer = build_deforamble_transformer_wo_encoder_cross_update(args)
        else:
            logger.info("build tranformer neck without encoder")
            transformer = build_deforamble_transformer_wo_encoder(args)
    else:
        if args.init_query_from_backbone or 'yolos' in args.vit_backbone:
            logger.error(f'not support with encoder for init_query_from_backbone {args.init_query_from_backbone} or vit backbone {args.vit_backbone}')
            raise ValueError(f'not support with encoder for init_query_from_backbone {args.init_query_from_backbone} or vit backbone {args.vit_backbone}')
        logger.info(f"build tranformer neck with {args.enc_layers} encoder")
        transformer = build_deforamble_transformer_wo_encoder(args)

    neck_encoder = args.neck_encoder
    if not neck_encoder in cnn_necks.__all__:
        logger.warning(f'neck_encoder is {neck_encoder}, NOT using CNN necks')
        neck_encoder = None
    logger.info(f'building cnn neck encoder {neck_encoder}')
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        args=args,
        cnn_neck=neck_encoder,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    enc_matcher = None
    enc_aux_matcher = None
    if args.two_stage:
        # enc_matcher = HungarianMatcher(cost_class=args.set_cost_class,
        #                     cost_bbox=args.set_cost_bbox,
        #                     cost_giou=args.set_cost_giou)
        logger.info('build dense matcher')
        enc_matcher = build_dense_matcher(args)
        if args.dense_aux_loss is None:
            enc_aux_matcher = None
        elif args.dense_aux_loss == 'o2m':
            logger.info('build dense aux matcher for one to many loss')
            enc_aux_matcher = build_dense_aux_matcher(args)
        elif args.dense_aux_loss == 'dam':
            logger.info('build dense aux loss using dam loss')
            enc_aux_matcher = None
        else:
            logger.error(f'WRONG --dense_aux_loss {args.dense_aux_loss}')
            raise ValueError(f'WRONG --dense_aux_loss {args.dense_aux_loss}')
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        dense_weight_dict = {'loss_ce_enc': args.dense_cls_loss_coef, 'loss_bbox_enc': args.dense_bbox_loss_coef,
                             'loss_giou_enc': args.dense_giou_loss_coef}
        # aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        aux_weight_dict.update(dense_weight_dict)
        weight_dict.update(aux_weight_dict)
    
    if args.dense_aux_loss is not None:
        if args.dense_aux_loss == 'o2m':
            weight_dict['loss_ce_enc_aux'] = args.dense_aux_loss_coef
        elif args.dense_aux_loss == 'dam':
            weight_dict['loss_mask_pred_enc_aux'] = args.dense_aux_loss_coef
    if args.kd_from_dec:
        weight_dict['loss_dec_kd_enc'] = args.dense_kd_loss_coef

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, enc_matcher,enc_aux_matcher, weight_dict, losses, eff_specific_head=args.eff_specific_head, 
                             focal_alpha=args.focal_alpha, my_enc_loss=args.my_enc_loss, dense_aux_loss=args.dense_aux_loss,
                             kd_from_dec=args.kd_from_dec)
    
  
    
    criterion.to(device)


    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
