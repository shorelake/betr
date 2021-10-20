#!/usr/bin/env bash

set -x

EXP_DIR=exps/swin_tiny_ram_vidt
PY_ARGS=${@:1}
# use PY_ARGS set vit --pretrained_path, --batch_size
python -u main.py \
    --output_dir ${EXP_DIR} \
    --vit_backbone swin_tiny_ram \
    --detector vidt \
    --enc_layers 0 \
    ${PY_ARGS}
