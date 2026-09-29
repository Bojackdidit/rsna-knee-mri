#!/bin/bash
# Route-doc defect 3: per-finding routing is starved because every finding reads the same
# globally-pooled per-window vector and the head never knows which anatomical slot a window came
# from. --slot_emb adds a ZERO-INITIALISED slot embedding, so step 0 is bit-identical to the
# control and the optimiser buys plane-selectivity only where it pays.
#
# LR96_CTRL is the matched control: same corpus, geometry, labels, fold, seed, epochs, aug.
# Only --slot_emb differs.
set -u
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
# wait for the scorer to release the GPU
# pgrep from git-bash does NOT see Windows processes, so `pgrep -f <script>` returns nothing and
# the wait falls straight through - that started this run while the scorer still held the GPU and
# put 11.8 GB on a 12.3 GB card. Ask Windows instead.
while powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '_score_all' } | Measure-Object).Count" 2>/dev/null | grep -qv '^0'; do sleep 60; done
sleep 30
$PY -u _run_sweep.py finetune_box.py \
  --ckpt timm --norm imagenet --workers 0 --grad_ckpt \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --aug geo,erase,blur,noise --k 12 --k_eval 12 --epochs 10 --bs 2 \
  --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42 --slot_emb \
  --group_csv 'D:\claude2\main_track\knee\study_fingerprint.csv' --group_folds 5 --group_fold 0 \
  --labels 'D:\claude2\main_track\knee\labels_blendimp_cd.parquet' \
  --tag SLOTEMB96
echo "=== slotemb done ==="
