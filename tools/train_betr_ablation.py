"""Launch one controlled BETR experiment; defaults follow chapter 3 COCO protocol."""
import argparse
import json
import os
from pathlib import Path

from main import get_args_parser, main as train


PRESETS = {
    'baseline': dict(enc_layers=0, dense_aux_loss=None),
    'gt-defcn': dict(enc_layers=0, dense_aux_loss='gt-defcn'),
    'gt': dict(enc_layers=0, dense_aux_loss='gt'),
    'a2f': dict(enc_layers=0, dense_aux_loss='dam'),
    'full': dict(enc_layers=1, dense_aux_loss='dam'),
    'dense6': dict(enc_layers=0, dec_layers=6, dense_aux_loss=None, spatial_prior_radius=float('inf')),
    'dense2': dict(enc_layers=0, dense_aux_loss=None, spatial_prior_radius=float('inf')),
    'legacy-o2m': dict(enc_layers=0, dense_aux_loss='o2m'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=PRESETS, required=True)
    parser.add_argument('--dry-run', action='store_true')
    cli, extra = parser.parse_known_args()
    model_parser = get_args_parser()
    defaults = dict(dataset='coco', dataset_file='coco', coco_path='../det_data/coco',
                    vit_backbone='swin_nano', pretrained_path='./swin_nano_patch4_window7_224.pth',
                    dec_layers=2, two_stage=True, eff_query_init=True, eff_specific_head=True,
                    proposal_net='rpn_default', neck_decoder='def_decoder', with_box_refine=True,
                    lr=2e-4, lr_backbone=1e-4, lr_linear_proj_mult=1., lr_scheduler='cosinelr',
                    epochs=50, warmup_epochs=0, weight_decay=1e-4, clip_max_norm=.1,
                    num_queries=300, dense_aux_loss_coef=2., spatial_prior_radius=1.5,
                    a2f_ratio=.2, kd_from_dec=True, print_freq=100,
                    output_dir=f'workdir/ablation/{cli.experiment}')
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if 16 % world_size:
        parser.error('Default global batch 16 requires WORLD_SIZE to divide 16')
    defaults['batch_size'] = 16 // world_size
    defaults.update(PRESETS[cli.experiment])
    model_parser.set_defaults(**defaults)
    args = model_parser.parse_args(extra)
    print(json.dumps(vars(args), indent=2), flush=True)
    if cli.dry_run:
        return
    output = Path(args.output_dir)
    if (output / 'checkpoint.pth').exists() and not args.resume:
        raise ValueError('Output already contains checkpoint.pth; use another --output_dir or --resume')
    output.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get('RANK', '0')) == 0:
        (output / 'experiment.json').write_text(json.dumps(vars(args), indent=2))
    train(args)


if __name__ == '__main__':
    main()
