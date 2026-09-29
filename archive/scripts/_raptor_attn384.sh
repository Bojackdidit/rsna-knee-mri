#!/bin/bash
# BUILD the Raptor core properly. The first attempt used a stripped head (mean pool @224) and got
# 0.80. This gives the Raptor SSL base (convnext_small, Dread's foundation) the CHAMPION's proven
# recipe: attention-MIL pooling @384, grad-ckpt, the exact fine-tune that takes timm ConvNeXt-Base to
# 0.9257. Same knobs as the shipped ConvNeXt arm, only the backbone weights differ (Raptor SSL vs
# timm ImageNet). If this lands near/above 0.93 gold the Raptor core is real and ships; if not, I
# iterate res/pool/LR from here - not revert to diagnosing.
set -u
cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
export KNEE_DATA_ROOT='D:\claude2\main_track\knee\phase2\local_box\raptor'
export KNEE_VRAM_FRAC=0.65 KNEE_DUTY=0.55 KNEE_TEMP_LIMIT=75 KNEE_WORKERS=0
echo "=== Raptor attn-384 (proven head, SSL base) starting $(date +%H:%M) ==="
$PY -u _run_sweep.py finetune_box.py \
  --ckpt raptor_cnn_ckpt/raptor_ssl_last.pt --arch convnext_small --norm none \
  --pool attn --res 384 \
  --aug geo,erase,blur,noise --k 12 --k_eval 12 --epochs 12 --bs 2 --grad_ckpt \
  --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --labels 'D:\claude2\main_track\knee\labels_blendimp_cd.parquet' \
  --tag RAPTOR_ATTN384
echo "=== done $(date +%H:%M) ==="
