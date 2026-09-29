#!/bin/bash
set -u; cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
export KNEE_DATA_ROOT='D:\claude2\main_track\knee\phase2\local_box\raptor'
export KNEE_VRAM_FRAC=0.80 KNEE_DUTY=0.55 KNEE_TEMP_LIMIT=75 KNEE_WORKERS=0
echo "=== DINOv3-LARGE 224 (stronger paradigm-diverse arm) starting $(date +%H:%M) ==="
$PY -u _run_sweep.py finetune_box.py \
  --ckpt timm --norm imagenet --arch vit_large_patch16_dinov3 --res 224 --grad_ckpt \
  --pool attn --k 12 --k_eval 12 --epochs 12 --bs 1 \
  --aug geo,erase,blur,noise --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --labels 'D:\claude2\main_track\knee\labels_blendimp_cd.parquet' --tag DINOV3L_224
echo "=== done $(date +%H:%M) ==="
