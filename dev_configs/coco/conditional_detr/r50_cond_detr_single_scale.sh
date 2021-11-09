#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_cond_detr_single_scale
PY_ARGS=${@:1}

python -u main.py \
    --num_feature_levels 1 \
    --output_dir ${EXP_DIR} \
    --dim_feedforward 2048 \
    --detector conditional_detr \
    --batch_size 2 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    ${PY_ARGS}