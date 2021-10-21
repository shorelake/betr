#!/usr/bin/env bash

set -x

EXP_DIR=exps/swin_small_deformable_detr
PY_ARGS=${@:1}
# use PY_ARGS set vit --pretrained_path, --batch_size
python -u main.py \
    --output_dir ${EXP_DIR} \
    --vit_backbone swin_small \
    --pretrained_path ./pretrained_model/swin_small_patch4_window7_224.pth \
    ${PY_ARGS}
