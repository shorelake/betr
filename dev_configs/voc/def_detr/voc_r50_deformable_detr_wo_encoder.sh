#!/usr/bin/env bash

set -x

EXP_DIR=exps/voc_r50_deformable_detr_wo_encoder
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --dataset_file voc \
    --dataset voc \
    --enc_layers 0 \
    ${PY_ARGS}
