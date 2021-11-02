#!/usr/bin/env bash

set -x

EXP_DIR=exps/voc_r50_deformable_detr_single_scale
PY_ARGS=${@:1}

python -u main.py \
    --num_feature_levels 1 \
    --output_dir ${EXP_DIR} \
    --dataset_file voc \
    --dataset voc \
    ${PY_ARGS}
