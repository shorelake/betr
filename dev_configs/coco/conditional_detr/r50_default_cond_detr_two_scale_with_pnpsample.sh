#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_default_cond_detr_two_scale_with_pnpsample
PY_ARGS=${@:1}

python -u main.py \
    --num_feature_levels 2 \
    --output_dir ${EXP_DIR} \
    --dim_feedforward 2048 \
    --detector default_conditional_detr \
    --batch_size 2 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --with_pnp_sampler \
    --sample_topk_ratio 0.17 \
    ${PY_ARGS}