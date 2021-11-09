#!/usr/bin/env bash

set -x

EXP_DIR=exps/swin_nano_yolos_vidt
PY_ARGS=${@:1}
# use PY_ARGS set vit --pretrained_path, --batch_size
python -u main.py \
    --output_dir ${EXP_DIR} \
    --vit_backbone swin_nano_yolos \
    --pretrained_path ./pretrained_model/swin_nano_patch4_window7_224.pth \
    --with_box_refine \
    --enc_layers 0 \
    --no_input_proj \
    --detector vidt \
    ${PY_ARGS}
