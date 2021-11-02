#!/usr/bin/env bash

set -x

EXP_DIR=exps/coco/anchordetr/r50_anchor_detr_default
PY_ARGS=${@:1}
# batch size 2, corresponding lr set 2e-5,2e-4
python -u main.py \
    --output_dir ${EXP_DIR} \
    --dataset_file voc \
    --dataset voc \
    --detector anchor_detr \
    --num_feature_levels 1 \
    --attention_type nn.MultiheadAttention \
    ${PY_ARGS}
