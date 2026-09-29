#!/bin/bash
# PIVOT off the CoAtNet-384 + attention-MIL architecture (plateaued at 0.945, generalises worse than
# the leaders per the board). This trains the RAPTOR SSL base (our own foundation, +0.043 over public
# bases) with the leaders' shape: SIMPLE MEAN POOLING at 224px, not attention-MIL at 384. Full-data
# single arm for a direct board read of whether the Raptor foundation + a simple head generalises
# better than CoAtNet-384. Local card, throttled, no Vast, no label work.
set -u
cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
export KNEE_DATA_ROOT='D:\claude2\main_track\knee\phase2\local_box\raptor'
export KNEE_VRAM_FRAC=0.65 KNEE_DUTY=0.55 KNEE_TEMP_LIMIT=75 KNEE_WORKERS=0

echo "=== Raptor mean-pool 224 (full-data) starting $(date +%H:%M) ==="
$PY -u _run_sweep.py finetune_box.py \
  --ckpt raptor_cnn_ckpt/raptor_ssl_last.pt --arch convnext_small --norm none \
  --pool mean --res 224 \
  --aug geo,erase,blur,noise --k 12 --k_eval 12 --epochs 10 --bs 4 \
  --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --labels 'D:\claude2\main_track\knee\labels_blendimp_cd.parquet' \
  --tag RAPTOR_MEAN224
echo "=== Raptor mean-224 done $(date +%H:%M) ==="
