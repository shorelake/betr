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

class ProposalNet(nn.Module):
    def __init__(self, d_model=256, num_classes=91, num_proposals=300, 
                 eff_query_init=False, eff_specific_head=False):
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

class RpnProposalNet(nn.Module):
    def __init__(self, d_model=256, num_classes=91, num_proposals=300, 
                 eff_query_init=False, eff_specific_head=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.eff_query_init = eff_query_init
        self.eff_specific_head = eff_specific_head
        
        # self.enc_output = nn.Linear(d_model, d_model)
        # self.enc_output_norm = nn.LayerNorm(d_model)

        self.rpn_tower = nn.Sequential(
            nn.Conv2d(d_model,d_model,kernel_size=3,stride=1,padding=1,bias=True),
            nn.GroupNorm(32, d_model),
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
        memorys = []
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

            proposal_valid = ((proposal > 0.01) & (proposal < 0.99)).all(-1, keepdim=True)

            proposal = torch.log(proposal / (1 - proposal)) # inverse sigmoid
            proposal = proposal.masked_fill(mask_flatten_.flatten(1).unsqueeze(-1), float('inf'))
            proposal = proposal.masked_fill(~proposal_valid, float('inf'))
            proposals.append(proposal)
            memory_lvl = memory[:, _cur:(_cur + H_ * W_), :]#.view(N_,H_,W_,C_).permute(0,3,1,2) # N C H W
            memory_lvl = memory_lvl.masked_fill(mask_flatten_.flatten(1).unsqueeze(-1), float(0))
            memory_lvl = memory_lvl.masked_fill(~proposal_valid, float(0))
            memory_lvl = memory_lvl.view(N_,H_,W_,C_).permute(0,3,1,2) # N C H W
            memory_lvl = self.rpn_tower(memory_lvl) # N C H W
            memorys.append(memory_lvl.flatten(2).permute(0,2,1))
            # memory[:, _cur:(_cur + H_ * W_), :] = memory_lvl.flatten(2).permute(0,2,1)
            _cur += (H_ * W_)

        output_proposals = torch.cat(proposals, 1)
        output_memory = torch.cat(memorys,1)
        # output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def forward(self, memory, mask_flatten, spatial_shapes,level_start_index,valid_ratios):
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.class_embed(output_memory)
        enc_outputs_coord_unact = self.bbox_embed(output_memory) + output_proposals

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
    if args.proposal_net == 'default':
        logger.info(f'build default proposal net')
        return ProposalNet(
            d_model=args.hidden_dim,
            num_classes= 1 if args.agn_proposal else num_classes,
            num_proposals=args.num_queries,
            eff_query_init=args.eff_query_init,
            eff_specific_head=args.eff_specific_head
        )
    elif args.proposal_net == 'rpn_default':
        logger.info(f'build rpn default proposal net')
        return RpnProposalNet(
            d_model=args.hidden_dim,
            num_classes= 1 if args.agn_proposal else num_classes,
            num_proposals=args.num_queries,
            eff_query_init=args.eff_query_init,
            eff_specific_head=args.eff_specific_head
        )
    elif args.proposal_net == 'fcos':
        # TODO
        logger.error(f'build fcos proposal net, NOT IMPLEMENT YET!')
        raise ValueError(f'build fcos proposal net, NOT IMPLEMENT YET!')
        return FcosProposalNet(
            d_model=args.hidden_dim,
            num_classes= 1 if args.agn_proposal else num_classes,
            num_proposals=args.num_queries,
            eff_query_init=args.eff_query_init,
            eff_specific_head=args.eff_specific_head
        )
    elif args.proposal_net == 'retina':
        # TODO
        logger.error(f'build retina proposal net, NOT IMPLEMENT YET!')
        raise ValueError(f'build retina proposal net, NOT IMPLEMENT YET!')
        return RetinaProposalNet(
            d_model=args.hidden_dim,
            num_classes= 1 if args.agn_proposal else num_classes,
            num_proposals=args.num_queries,
            eff_query_init=args.eff_query_init,
            eff_specific_head=args.eff_specific_head
        )