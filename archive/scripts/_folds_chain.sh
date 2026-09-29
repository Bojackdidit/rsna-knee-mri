#!/bin/bash
# Fold ensemble - the one remaining lever with a competitor's MEASURED board number behind it.
#
# The #1 team reports +0.009 on the board from folds alone, and we have never built one. 0.945 +
# 0.009 is 0.954, which is Scott territory. This is not an offline bet: gold-58 has now predicted
# the wrong sign twice (route3, trio), so nothing here will be selected offline - the ensemble goes
# straight to the board.
#
# Fold 0 already exists as LR96_CTRL (gold-58 0.8949). This trains folds 1, 2 and 3 on the same
# recipe, so all four are comparable and each holds out a different 870 studies.
#
# Runtime caps the ensemble at FOUR arms: the kernel places arm i on cuda:i%2 and one arm per card
# costs ~21 min, so 4 arms (2 per card) is ~42 min and 5 would breach the 1 h scoring limit.
set -u
cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe

running () { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'finetune_box' } | Measure-Object).Count" 2>/dev/null | tr -d '\r' | head -1; }

# Folds 1-4, not 1-3. Four arms is the SHIPPABLE maximum, but the validation pipeline needs all
# five: fold k holds out its own 870 studies, so folds 0-4 together give an out-of-fold prediction
# for every one of the 4,349 labelled studies, from a model that never saw it. That is a validation
# set 75x the size of gold-58, which is the instrument we have never had and the reason every
# offline call this week has been unreliable.
echo "=== folds chain armed $(date +%H:%M), waiting for the GPU ==="
for F in 1 2 3 4; do
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
