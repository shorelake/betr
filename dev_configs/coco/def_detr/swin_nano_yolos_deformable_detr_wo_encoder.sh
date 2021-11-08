#!/usr/bin/env bash

set -x

EXP_DIR=exps/swin_nano_fuse_deformable_detr_wo_encoder
PY_ARGS=${@:1}
# use PY_ARGS set vit --pretrained_path, --batch_size
python -u main.py \
    --output_dir ${EXP_DIR} \
    --vit_backbone swin_nano_yolos \
    --pretrained_path ./pretrained_model/swin_nano_patch4_window7_224.pth \
    --with_box_refine \
    --enc_layers 0 \
    --no_input_proj \
    --init_query_from_backbone \
    ${PY_ARGS}
