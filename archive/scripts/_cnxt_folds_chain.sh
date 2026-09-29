#!/bin/bash
# Overnight: train ConvNeXt folds 1-4 (fold 0 already done as CNXT_F0), same 5-fold split as the
# CoAtNet arms. Justified by the fold-0 read: cross-arch blend +0.0095 vs CoAtNet at 100% bootstrap
# win-rate on the OOF instrument. Serial, throttled; each ~3.7h. Skips any fold already trained.
set -u
cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
PS=/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe
export KNEE_DATA_ROOT='D:\claude2\main_track\knee\phase2\local_box\raptor'
export KNEE_VRAM_FRAC=0.65 KNEE_DUTY=0.55 KNEE_TEMP_LIMIT=75 KNEE_WORKERS=0

running () { "$PS" -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'finetune_box' } | Measure-Object).Count" 2>/dev/null | tr -d '\r' | head -1; }

for F in 1 2 3 4; do
  if [ -f "raptor_ft_CNXT_F${F}.pt" ]; then echo "CNXT_F${F} exists, skip"; continue; fi
  while [ "$(running)" != "0" ]; do sleep 120; done
  sleep 20
  echo "=== ConvNeXt fold-${F} starting $(date +%H:%M) ==="
  $PY -u _run_sweep.py finetune_box.py \
    --ckpt timm --norm imagenet --workers 0 --grad_ckpt \
    --arch convnext_base.fb_in22k_ft_in1k_384 --res 384 \
    --aug geo,erase,blur,noise --k 12 --k_eval 12 --epochs 10 --bs 2 \
    --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42 \
    --group_csv 'D:\claude2\main_track\knee\study_fingerprint.csv' --group_folds 5 --group_fold $F \
    --labels 'D:\claude2\main_track\knee\labels_blendimp_cd.parquet' \
    --tag CNXT_F${F}
  echo "=== ConvNeXt fold-${F} done $(date +%H:%M) ==="
done
echo "=== all ConvNeXt folds done $(date +%H:%M) ==="
