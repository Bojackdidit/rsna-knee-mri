#!/bin/bash
# Arm 3: the winning scratch recipe on the BLENDED label set, sixteen epochs.
#
# Two findings drive this. Both arms so far peaked at epoch 14 and spent the remaining nine epochs
# below their own best, so twenty-four was a third of the compute wasted; sixteen is the right length.
# And label blending is the highest-ranked lever we have never trained on this recipe - our gold 58
# reads 0.8947 -> 0.9132 for it, and a competitor sitting at 0.936 attributes +0.015 of BOARD score to
# it. Everything else is identical to the arm that just took the board to 0.944, so the label set is
# the single variable.
set -u
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/knee
echo "BLEND_ARM start $(date -u +%F' '%H:%M:%S)" >> ft_chain.log
python -u finetune_box.py \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --ckpt timm --norm imagenet \
  --labels /workspace/knee/labels_blendimp_cd.parquet --bs 2 --grad_ckpt --k 24 --k_eval 24 \
  --epochs 16 --bb_lr 3e-5 --head_lr 1e-3 --workers 4 --seed 777 \
  --aug geo,erase,blur,noise --ema 0.9998 --tag BLEND_ARM > run_BLEND_ARM.log 2>&1
echo "BLEND_ARM done rc=$? $(date -u +%F' '%H:%M:%S)" >> ft_chain.log
