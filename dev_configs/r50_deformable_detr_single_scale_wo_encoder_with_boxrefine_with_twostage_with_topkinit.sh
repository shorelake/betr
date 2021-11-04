#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr_single_scale_wo_encoder_with_boxrefine_with_twostage_with_init_tgt_withtopk
PY_ARGS=${@:1}

python -u main.py \
    --num_feature_levels 1 \
    --output_dir ${EXP_DIR} \
    --enc_layers 0 \
    --with_box_refine \
    --two_stage \
    --init_tgt_withtopk \
    ${PY_ARGS}
