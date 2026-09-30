"""Checkpoint and validation helpers shared by BETR reproduction tools."""
import argparse
from pathlib import Path

import torch

from main import get_args_parser, get_model


def load_betr(checkpoint_path, device='cuda', enc_layers=None, backbone=None, return_criterion=False):
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    args = get_args_parser().parse_args([])
    saved = checkpoint.get('args')
    if saved is None and (enc_layers is None or backbone is None):
        raise ValueError('Checkpoint has no args: provide --enc-layers and --backbone')
    if saved is not None:
        vars(args).update(vars(saved) if isinstance(saved, argparse.Namespace) else saved)
    if enc_layers is not None:
        args.enc_layers = enc_layers
    if backbone is not None:
        args.vit_backbone = backbone
    args.pretrained_path = None
    args.device = device
    args.distributed = False
    args.with_gt_mask = False
    args.with_dam_mask = False
    model, criterion, postprocessors = get_model(args)
    state = checkpoint['model']
    state = {k.removeprefix('module.'): v for k, v in state.items()
             if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    metadata = {'checkpoint': str(Path(checkpoint_path).resolve()),
                'epoch': checkpoint.get('epoch'), 'args': vars(args),
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'device': torch.cuda.get_device_name(device) if device.startswith('cuda') else device}
    if return_criterion:
        return model, criterion, postprocessors, args, metadata
    return model, postprocessors, args, metadata


def add_checkpoint_options(parser):
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--enc-layers', type=int)
    parser.add_argument('--backbone', choices=['swin_nano', 'swin_tiny', 'swin_small', 'swin_base'])
    parser.add_argument('--coco-path', default='../det_data/coco')


def a2f_maps(enc, ratio):
    from util.dam import attn_map_to_flat_grid
    attention = attn_map_to_flat_grid(enc['spatial_shapes'], enc['level_start_index'],
                                     enc['sampling_locations_dec'], enc['attn_weights_dec']).sum((1, 2))
    valid = ~enc['mask_flatten']
    scores = attention.masked_fill(~valid, -float('inf'))
    target = torch.zeros_like(attention)
    for i in range(len(target)):
        n = int(valid[i].sum())
        k = min(int(n * ratio) + 1, n)
        target[i, scores[i].topk(k).indices] = 1
    return attention, target
