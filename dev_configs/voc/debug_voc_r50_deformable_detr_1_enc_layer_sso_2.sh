#!/usr/bin/env bash

set -x

EXP_DIR=exps/debug_voc_r50_deformable_detr_1_enc_layer_sso_2
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --dataset_file voc \
    --dataset voc \
    --enc_layers 1 \
    --msi_sso 2 \
    ${PY_ARGS}
