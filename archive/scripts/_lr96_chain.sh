#!/bin/bash
# The bb_lr ladder, re-run at the geometry the kernel actually serves.
#
# The original ladder (3e-5 best, 1e-4 -0.014, 3e-4 diverged) was measured entirely on the 44-slice
# corpus while the kernel serves 96 - windows spanning 2.18x more anatomy than at inference. Its own
# memory marks it "unsettled at correct geometry". The weight-space finding it was meant to refute
# still stands: backbone 0.0220 from ImageNet init, head 0.8953.
#
# Control and treatment differ ONLY in bb_lr. Same corpus, same labels, same fold, same seed.
set -u
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
COMMON="--ckpt timm --norm imagenet --workers 0 --grad_ckpt \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --aug geo,erase,blur,noise --k 12 --k_eval 12 --epochs 10 --bs 2 \
  --head_lr 1e-3 --wd 0.02 --seed 42 \
  --group_csv D:\claude2\main_track\knee\study_fingerprint.csv --group_folds 5 --group_fold 0 \
  --labels D:\claude2\main_track\knee\labels_blendimp_cd.parquet"

echo "=== CONTROL bb_lr 3e-5 @96 ==="
$PY -u _run_sweep.py finetune_box.py $COMMON --bb_lr 3e-5 --tag LR96_CTRL
echo "=== TREATMENT bb_lr 1e-4 @96 ==="
$PY -u _run_sweep.py finetune_box.py $COMMON --bb_lr 1e-4 --tag LR96_HIGH
echo "=== chain done ==="
