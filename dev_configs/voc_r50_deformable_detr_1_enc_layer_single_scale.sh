#!/usr/bin/env bash

set -x

EXP_DIR=exps/voc_r50_deformable_detr_1_enc_layer_single_scale
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --dataset_file voc \
    --dataset voc \
    --enc_layers 1 \
    --num_feature_levels 1 \
    ${PY_ARGS}
