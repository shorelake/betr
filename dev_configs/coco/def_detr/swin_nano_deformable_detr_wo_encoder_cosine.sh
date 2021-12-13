#!/usr/bin/env bash

set -x

EXP_DIR=exps/swin_nano_deformable_detr_wo_encoder_cosine
PY_ARGS=${@:1}
# use PY_ARGS set vit --pretrained_path, --batch_size
python -u main.py \
    --output_dir ${EXP_DIR} \
    --vit_backbone swin_nano \
    --pretrained_path ./pretrained_model/swin_nano_patch4_window7_224.pth \
    --with_box_refine \
    --batch_size 2 \
    --enc_layers 0 \
    --lr_backbone 1e-4 \
    --lr 1e-4 \
    --lr_linear_proj_mult 1 \
    --lr_scheduler cosinelr \
    ${PY_ARGS}
