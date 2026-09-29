#!/bin/bash
set -u; cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
export KNEE_DATA_ROOT='D:\claude2\main_track\knee\phase2\local_box\raptor'
export KNEE_VRAM_FRAC=0.65 KNEE_DUTY=0.55 KNEE_TEMP_LIMIT=75 KNEE_WORKERS=0
echo "=== EffV2S-288 (leaders' recipe: simple ImageNet net, low res) starting $(date +%H:%M) ==="
$PY -u _run_sweep.py finetune_box.py \
  --ckpt timm --norm imagenet --arch tf_efficientnetv2_s.in21k_ft_in1k --res 288 \
  --pool attn --k 12 --k_eval 12 --epochs 12 --bs 4 \
  --aug geo,erase,blur,noise --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --labels 'D:\claude2\main_track\knee\labels_blendimp_cd.parquet' --tag EFFV2S_288
echo "=== done $(date +%H:%M) ==="
