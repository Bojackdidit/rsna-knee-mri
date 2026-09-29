#!/bin/bash
# Resume of _folds_chain.sh after the 2026-09-25 power outage. Fold 1 completed 22:47 the night
# before (artifacts verified on disk); this runs the remaining folds 2-4 with the IDENTICAL
# command, so all five folds (CTRL=0, F1..F4) stay comparable for the out-of-fold ensemble.
set -u
cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe

# The scratchpad corpus died with the post-outage cleanup, but a SECOND durable copy already
# existed at phase2\local_box\raptor - byte-verified 2026-09-25 against the Kaggle staging
# (prefix md5 match, exact size, ids identical, masks consistent). Train from it directly.
export KNEE_DATA_ROOT='D:\claude2\main_track\knee\phase2\local_box\raptor'

running () { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'finetune_box' } | Measure-Object).Count" 2>/dev/null | tr -d '\r' | head -1; }

echo "=== folds resume armed $(date +%H:%M), folds 2-4 ==="
for F in 2 3 4; do
  if [ -f "raptor_ft_LR96_F${F}.pt" ]; then echo "=== FOLD $F already exists, skipping ==="; continue; fi
  while [ "$(running)" != "0" ]; do sleep 120; done
  sleep 30
  echo "=== FOLD $F starting $(date +%H:%M) ==="
  $PY -u _run_sweep.py finetune_box.py \
    --ckpt timm --norm imagenet --workers 0 --grad_ckpt \
    --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
    --aug geo,erase,blur,noise --k 12 --k_eval 12 --epochs 10 --bs 2 \
    --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42 \
    --group_csv 'D:\claude2\main_track\knee\study_fingerprint.csv' --group_folds 5 --group_fold $F \
    --labels 'D:\claude2\main_track\knee\labels_blendimp_cd.parquet' \
    --tag LR96_F$F
  echo "=== FOLD $F done $(date +%H:%M) ==="
done
echo "=== all folds done $(date +%H:%M) ==="
