#!/bin/bash
# Train the public release arm on the ALREADY-PUBLISHED 44-slice corpus.
#
# The point of this arm is that it costs us nothing to give away. The public corpus is 44 slices
# and our own geometry table marks that depth "arms cap out at 0.928"; the last public weights
# scored 0.924. Everything we have learned since - augmentation, the correct backbone LR, the full
# recipe - has never been applied AT that depth. So this is a genuine improvement on what is public
# without touching the 96-slice geometry, which is the thing actually holding the public ceiling
# down while we sit at 0.945.
#
# Runs last: the fold ensemble and its out-of-fold validation set matter more, and this arm is only
# blocked on GPU time, not on any decision.
set -u
cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe

running () { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'finetune_box' } | Measure-Object).Count" 2>/dev/null | tr -d '\r' | head -1; }

# wait for every fold arm, then for the GPU to be idle
for T in LR96_F1 LR96_F2 LR96_F3 LR96_F4; do
  while [ ! -f "raptor_ft_${T}.pt" ]; do sleep 300; done
done
while [ "$(running)" != "0" ]; do sleep 120; done
sleep 60

echo "=== training the public release arm at 44 slices $(date +%H:%M) ==="
# KNEE_DATA_ROOT points at the PUBLISHED corpus, not the 96 one. finetune_box reads the depth from
# the corpus itself and refuses a geometry mismatch, so this cannot silently train at 96.
KNEE_DATA_ROOT='D:\claude2\main_track\knee\phase2\corpus_stage' \
KNEE_VRAM_FRAC=0.80 KNEE_DUTY=0.90 KNEE_TEMP_LIMIT=78 KNEE_WORKERS=0 \
$PY -u _run_sweep.py finetune_box.py \
  --ckpt timm --norm imagenet --workers 0 --grad_ckpt \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --expect_depth 44 \
  --aug geo,erase,blur,noise --k 12 --k_eval 12 --epochs 14 --bs 2 \
  --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --labels 'D:\claude2\main_track\knee\labels_blendimp_cd.parquet' \
  --tag PUBLIC44
echo "=== release arm done $(date +%H:%M) ==="
grep -E '^ep[0-9]' _release44.log | tail -3
