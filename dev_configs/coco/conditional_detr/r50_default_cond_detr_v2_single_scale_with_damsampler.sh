#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_default_cond_detr_v2_single_scale_with_damsampler
PY_ARGS=${@:1}

python -u main.py \
    --num_feature_levels 1 \
    --output_dir ${EXP_DIR} \
    --dim_feedforward 2048 \
    --detector default_conditional_detr_v2 \
    --batch_size 2 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --with_pnp_sampler \
    --sample_topk_ratio 0.5 \
    --with_dam_mask \
    ${PY_ARGS}