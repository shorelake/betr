# Verified detr environment

Verified on 2026-09-17 with two RTX 5090 GPUs (SM 120).

- Prefix: `/data/hubin/miniconda3/envs/detr`
- Python 3.10.21
- PyTorch 2.11.0+cu128; torchvision 0.26.0+cu128
- CUDA compiler 12.8.61 from detr, not system CUDA 13.1
- NumPy 2.2.6; SciPy 1.15.3; timm 1.0.29
- mmcv-full 1.7.2 (this code requires MMCV 1.x APIs)
- pycocotools 2.0.11; fvcore 0.1.5.post20221221
- MultiScaleDeformableAttention 1.0 built from `models/ops`

The original failure was a missing MultiScaleDeformableAttention extension.
Its old THC header and tensor APIs have been updated for current PyTorch.
The local checkpoint loader allows argparse.Namespace metadata while retaining
weights-only loading. CUDA float/double forward comparisons and a 32-channel
numerical gradient check passed; real two-GPU COCO training passed 3500 steps.

## Run

```bash
conda activate detr
bash train_coco.sh --print_freq 100
```

The launcher selects detr Python explicitly and disables user-site packages.
Override `PY_ENV_PREFIX`, `DATA_ROOT`, `OUT_DIR`, or `NPROC_PER_NODE` as needed.
COCO is at `../det_data/coco`. The pretrained checkpoint is linked from
`../Deformable-DETR/swin_nano_patch4_window7_224.pth`.
Logs and epoch checkpoints go to `workdir/betr_coco` by default.
Do not launch another copy while training is already running on the same port.

## Rebuild the CUDA extension

```bash
conda activate detr
bash models/ops/make.sh
python -m pip check
```

Rebuild after changing PyTorch or CUDA. `dev_models/ops` contains a legacy copy
with the same extension name; install the maintained `models/ops` copy above.
For a different GPU, set `TORCH_CUDA_ARCH_LIST` appropriately.

## Nested conda activation

On this machine, `conda run` from an already active environment can trigger
base auto-activation and fail in compiler deactivation hooks. This works:

```bash
CONDA_AUTO_ACTIVATE_BASE=false conda run -n detr python --version
```

The compiler exists in detr; installing another compiler in base is unnecessary.
The training launcher does not depend on `conda run`.
