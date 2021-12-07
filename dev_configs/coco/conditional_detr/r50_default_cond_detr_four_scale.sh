#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_default_cond_detr_four_scale
PY_ARGS=${@:1}

python -u main.py \
    --num_feature_levels 4 \
    --output_dir ${EXP_DIR} \
    --dim_feedforward 2048 \
    --detector default_conditional_detr \
    --batch_size 2 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --enc_layers 0 \
    --neck_encoder deftransformer \
#    --neck_encoder fpn \
#    --neck_encoder panet \
    ${PY_ARGS}