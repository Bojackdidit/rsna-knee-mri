#!/bin/bash
# Arm 4, queued behind the blend arm: the winning recipe again on the SHIPPED labels, third seed.
#
# This is the hedge. The blend arm changes the label set, which is the lever with the strongest outside
# evidence but which our own gold 58 has ranked wrongly three times out of three. This one changes
# nothing but the seed, so it extends the thing that has actually paid twice - a parity partner, which
# our own board reads put at about +0.004 - and it is available whatever the blend arm does.
# Sixteen epochs, because both previous arms peaked at fourteen.
set -u
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/knee
while pgrep -f "[f]inetune_box.py" > /dev/null; do sleep 300; done
echo "SEED3_ARM start $(date -u +%F' '%H:%M:%S)" >> ft_chain.log
python -u finetune_box.py \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --ckpt timm --norm imagenet \
  --labels /workspace/knee/labels_confdist.parquet --bs 2 --grad_ckpt --k 24 --k_eval 24 \
  --epochs 16 --bb_lr 3e-5 --head_lr 1e-3 --workers 4 --seed 31337 \
  --aug geo,erase,blur,noise --ema 0.9998 --tag SEED3_ARM > run_SEED3_ARM.log 2>&1
echo "SEED3_ARM done rc=$? $(date -u +%F' '%H:%M:%S)" >> ft_chain.log
