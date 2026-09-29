#!/bin/bash
# Next ablation after slot_emb: --pool gru. A bidirectional GRU over the ORDERED windows, gated to
# zero at init so step 0 is bit-identical to pool=attn. Every arm we ship uses order-agnostic
# pooling, which cannot represent a finding running across consecutive slices at all; this is the
# shape the RSNA-2022 and RSNA-2023 winners used, and being architecturally unlike our stack it
# attacks decorrelation - the only mechanism the board has ever confirmed.
#
# Control is LR96_CTRL (gold-58 0.8949): same corpus, geometry, labels, fold, seed, epochs, aug.
set -u
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
# pgrep cannot see Windows processes from git-bash, so ask Windows - but match python.exe ONLY.
# An unfiltered CommandLine match also hits the bash wrappers AND the PowerShell command itself
# (its own command line contains the pattern), so it counts itself and never reaches 0.
running () { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'finetune_box' } | Measure-Object).Count" 2>/dev/null | tr -d '\r' | head -1; }
while [ "$(running)" != "0" ]; do sleep 120; done
sleep 60
echo "=== GPU free, starting GRU head run at $(date +%H:%M) ==="
$PY -u _run_sweep.py finetune_box.py \
  --ckpt timm --norm imagenet --workers 0 --grad_ckpt \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --pool gru \
  --aug geo,erase,blur,noise --k 12 --k_eval 12 --epochs 10 --bs 2 \
  --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --group_csv 'D:\claude2\main_track\knee\study_fingerprint.csv' --group_folds 5 --group_fold 0 \
  --labels 'D:\claude2\main_track\knee\labels_blendimp_cd.parquet' \
  --tag GRU96
echo "=== gru done $(date +%H:%M) ==="
