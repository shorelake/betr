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
    --lr_backbone 2.5e-5 \
    --lr 2.5e-5 \
    --lr_linear_proj_mult 1 \
    --lr_scheduler cosinelr \
    --batch_size 8 \
    --mosaic \
    ${PY_ARGS}