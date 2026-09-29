#!/bin/bash
# When SCANNER_ARM exits, start a PARITY HUNT rather than a second anti-scanner seed.
#
# Why changed: SCANNER_ARM peaked at 0.9247 (ep11) and then declined, so the anti-scanner recipe is
# not clearing arm 2 at 16 epochs. Box 2 already owns the clean test of it (24 epochs, seed 4242, the
# only difference from arm 2), so a second 16-epoch anti-scanner seed here would spend ten hours
# duplicating a probably-dead lever.
#
# What we actually need is a partner at or above arm 2's 0.9315. Three arms on the proven recipe
# landed 0.9260, 0.9243 and ~0.925, so arm 2 is an OUTLIER and another draw is a genuine gamble - but
# it is the only recipe that has ever produced a 0.9315, and Scott reports ensembling is worth +0.005
# once the members are comparable. 24 epochs, matching arm 2 exactly, new seed.
set -u
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/knee
while pgrep -f "[f]inetune_box.py" > /dev/null; do sleep 120; done
echo "chain: SCANNER_ARM done, starting PARITY_S5 $(date -u +%F' '%H:%M:%S)" >> ft_chain.log
setsid nohup python -u finetune_box.py \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --ckpt timm --norm imagenet \
  --labels /workspace/knee/labels_confdist.parquet --bs 2 --grad_ckpt --k 24 --k_eval 24 \
  --epochs 24 --bb_lr 3e-5 --head_lr 1e-3 --workers 4 --seed 1337 \
  --aug geo,erase,blur,noise --ema 0.9998 --tag PARITY_S5 \
  > run_PARITY_S5.log 2>&1 < /dev/null &
echo "chain: launched $(date -u +%F' '%H:%M:%S)" >> ft_chain.log
