#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PY_ENV_PREFIX="${PY_ENV_PREFIX:-/data/hubin/miniconda3/envs/detr}"
export PATH="${PY_ENV_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
DATA_ROOT="${DATA_ROOT:-../det_data/coco}"
OUT_DIR="${OUT_DIR:-./workdir/betr_coco/}"

mkdir -p "${OUT_DIR}"

"${PY_ENV_PREFIX}/bin/python" -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE:-2}" --master_port="${MASTER_PORT:-29537}" main.py \
  --dataset coco \
  --dataset_file coco \
  --coco_path "${DATA_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --vit_backbone swin_nano \
  --pretrained_path ./swin_nano_patch4_window7_224.pth \
  --batch_size 4 \
  --enc_layers 1 \
  --dec_layers 2 \
  --lr_backbone 1e-4 \
  --lr 2e-4 \
  --lr_linear_proj_mult 0.1 \
  --lr_scheduler cosinelr \
  --two_stage \
  --eff_query_init \
  --eff_specific_head \
  --proposal_net rpn_default \
  --neck_decoder def_decoder \
  --with_box_refine \
  --dense_aux_loss dam \
  --dense_aux_loss_coef 2 \
  "$@"
