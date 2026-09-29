#!/bin/bash
# Retrain the Raptor base with a real SSL fine-tune LR. The first arm (bb_lr 3e-5, ported from
# CoAtNet where the backbone is meant to barely move) gave only 0.80 gold with weak findings -
# under-adapted backbone. Raise bb_lr 3e-5 -> 2e-4 so the SSL ConvNeXt-Small actually fine-tunes.
set -u
cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
export KNEE_DATA_ROOT='D:\claude2\main_track\knee\phase2\local_box\raptor'
export KNEE_VRAM_FRAC=0.65 KNEE_DUTY=0.55 KNEE_TEMP_LIMIT=75 KNEE_WORKERS=0
echo "=== Raptor mean-224 bb_lr 2e-4 starting $(date +%H:%M) ==="
$PY -u _run_sweep.py finetune_box.py \
  --ckpt raptor_cnn_ckpt/raptor_ssl_last.pt --arch convnext_small --norm none \
  --pool mean --res 224 \
  --aug geo,erase,blur,noise --k 12 --k_eval 12 --epochs 10 --bs 4 \
  --bb_lr 2e-4 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --labels 'D:\claude2\main_track\knee\labels_blendimp_cd.parquet' \
  --tag RAPTOR_LR2e4
echo "=== done $(date +%H:%M) ==="
