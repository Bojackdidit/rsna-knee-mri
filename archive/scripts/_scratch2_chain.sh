#!/bin/bash
# SCRATCH_AUG2 - identical recipe to SCRATCH_AUG (which produced the 0.943 board champion arm),
# seed 4242 instead of 42. Purpose: a parity partner for the 0.943 arm so the SECOND shipped
# model can also be replaced. Self-pairing at parity is measured at +0.004 free; a decorrelated
# but weaker partner is measured at zero. Same labels, same geometry, same augmentation, same EMA.
set -u
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/knee
echo "SCRATCH_AUG2 start $(date -u +%F' '%H:%M:%S)" >> ft_chain.log
python -u finetune_box.py \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --ckpt timm --norm imagenet \
  --labels /workspace/knee/labels_confdist.parquet --bs 2 --grad_ckpt --k 24 --k_eval 24 \
  --epochs 24 --bb_lr 3e-5 --head_lr 1e-3 --workers 4 --seed 4242 \
  --aug geo,erase,blur,noise --ema 0.9998 --tag SCRATCH_AUG2 > run_SCRATCH_AUG2.log 2>&1
echo "SCRATCH_AUG2 done rc=$? $(date -u +%F' '%H:%M:%S)" >> ft_chain.log
