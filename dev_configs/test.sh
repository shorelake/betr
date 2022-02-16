#!/usr/bin/env bash

set -x
EXP_DIR=exps/test
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --vit_backbone yolox_cspdarknet_s \
    --pretrained_path ./pretrained_model/yolox_cspdarknet_s_ckpt_epoch_295.pth \
    --two_stage \
    --eff_query_init \
    --eff_specific_head \
    --proposal_net rpn_default \
    --dec_layers 2 \
    --with_box_refine \
    --enc_layers 1 \
    --real_time \
    --num_feature_levels 3 \
    --batch_size 2 \
    --mosaic \
    ${PY_ARGS}