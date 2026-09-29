#!/bin/bash
# Raptor - CoAtNet 224. Testing the configuration a top placement reported: a SINGLE model at
# 224 pixels scoring 0.934, against our single 384 model which just scored 0.928 on the board.
#
# One variable changes from the reference run: resolution 384 -> 224, with the matching native
# backbone (coatnet_rmlp_2_rw_224 rather than the 384 checkpoint run off-resolution). Everything
# else is the recipe that produced 0.9263 offline / 0.936-0.938 solo on the board: same corpus,
# same 4,349 trainable studies, same 58-study gate, same schedule and seed.
#
# Why 224 is worth a run even though our 384 arm is decent: 224 is 0.34x the pixels, so it is
# ~3x cheaper per window. Their 224 model beats our 384 model on the board (0.934 vs 0.928) at a
# fraction of the compute, which is the only credible route to the 5-minute scoring the leaders
# report. Speed and score point the same way here, which is rare.
#
# LABELS: confdist, not the hand-extracted set. The board just measured the hand labels at 0.928
# solo against 0.936-0.938 for a confdist solo, so confdist gives the 224 test its best shot and
# keeps resolution as the ONLY change from the reference. Swap --labels to test a different target.
set -u
cd /workspace
export KNEE_DATA_ROOT=/workspace/rsna_knee/raptor_combined

nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu --format=csv,noheader

python finetune_raptor_cnn.py --ckpt timm --norm imagenet --workers 8 \
  --tag raptor_224 --arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 \
  --labels /workspace/labels_cd_rsna.parquet --slots '' \
  --k 12 --k_eval 24 --epochs 16 --bs 16 \
  --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 777 > /workspace/train_224.log 2>&1

B=$(grep -oE 'BEST GOLD macro-AUC [0-9.]+' /workspace/train_224.log | tail -1 | awk '{print $NF}')
echo "raptor_224 gold=${B:-NONE}" >> /workspace/RESULTS.txt
echo "[224] done gold=${B:-NONE}"
