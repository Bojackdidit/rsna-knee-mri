#!/bin/bash
# A champion-grade arm with a KNOWN recipe.
#
# SCRATCH_AUG2 is our best arm (gold-58 0.9330) and nobody can reproduce it: it was trained on a
# rented box and its log never came back, so we ship a champion we cannot rebuild. LR96_CTRL proved
# the recipe works honestly (gold-58 0.8949 while surrendering 870 studies to a holdout and running
# only 10 epochs). This is that same recipe at FULL STRENGTH - no holdout, more epochs - so the
# result is directly shippable and, unlike SCRATCH_AUG2, written down.
#
# It validates on gold-58 each epoch, which stays clean because gold is excluded from the soft
# labels for every arm. The 870 fold is NOT held out here, so its 870 number will be memorised -
# read gold-58 only.
set -u
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
# Match python.exe ONLY - bash wrappers and the PowerShell command itself also carry the pattern.
running () { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'finetune_box' } | Measure-Object).Count" 2>/dev/null | tr -d '\r' | head -1; }
while [ "$(running)" != "0" ]; do sleep 120; done
sleep 60
echo "=== GPU free, starting FULL-STRENGTH run at $(date +%H:%M) ==="
$PY -u _run_sweep.py finetune_box.py \
  --ckpt timm --norm imagenet --workers 0 --grad_ckpt \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --aug geo,erase,blur,noise --k 12 --k_eval 12 --epochs 14 --bs 2 \
  --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --labels 'D:\claude2\main_track\knee\labels_blendimp_cd.parquet' \
  --tag FULL96
echo "=== FULL96 done $(date +%H:%M) ==="
