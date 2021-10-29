#!/usr/bin/env bash

set -x

EXP_DIR=exps/swin_nano_deformable_detr_wo_encoder_with_two_stage
PY_ARGS=${@:1}
# use PY_ARGS set vit --pretrained_path, --batch_size
python -u main.py \
    --output_dir ${EXP_DIR} \
    --vit_backbone swin_nano \
    --pretrained_path ./pretrained_model/swin_nano_patch4_window7_224.pth \
    --with_box_refine \
    --enc_layers 0 \
    --two_stage \
    ${PY_ARGS}
