#!/usr/bin/env bash

set -x

EXP_DIR=exps/swin_nano_ram_vidt
PY_ARGS=${@:1}
#  use PY_ARGS set vit --pretrained_path, --batch_size
#  following lr setting follow bacthsize 2 per device, 8 devices
#  when batch size changes, adapt lr at scale
python -u main.py \
    --output_dir ${EXP_DIR} \
    --vit_backbone swin_nano_ram \
    --detector vidt \
    --enc_layers 0 \
    --num_queries 100 \
    --lr_scheduler cosinelr \
    --lr_backbone 1e-4 \
    --lr 1e-4 \
    --lr_linear_proj_mult 1 \
    --pretrained_path ./pretrained_model/swin_nano_patch4_window7_224.pth \
    --with_box_refine \
    ${PY_ARGS}
