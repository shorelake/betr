#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_cond_detr_single_scale_learned_anchors
PY_ARGS=${@:1}

python -u main.py \
    --num_feature_levels 1 \
    --output_dir ${EXP_DIR} \
    --dim_feedforward 2048 \
    --detector conditional_detr \
    --batch_size 2 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --with_anchors \
    --spatial_prior learned \
    ${PY_ARGS}