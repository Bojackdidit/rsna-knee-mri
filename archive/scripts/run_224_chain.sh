#!/bin/bash
# Raptor - 224 arm fleet. Three more arms at 224, one per label set, run back to back so the
# GPU never idles between them.
#
# Why: 224 costs 107 s/epoch against 325 s at 384, and the same 3x holds at inference. The
# 29-minute scoring ceiling caps us at TWO 384 arms; at 224 that budget holds three or four.
# Our own combotest says arm count is the lever we have never been able to pull:
#     2 arms 0.9331 -> 3 arms 0.9346 -> 4 arms 0.9356
#
# One arm per label set, because the arms must DISAGREE to be worth stacking. confdist@224 is
# already trained (gold 0.9040); these three complete the set:
#     opus   - labels read by hand from all 4,407 reports
#     blend  - opus averaged with public (0.9230 vs gold, the best label set we hold, never trained)
#     public - the public label set
#
# Everything except the label file and the tag is identical to the reference recipe.
set -u
cd /workspace
export KNEE_DATA_ROOT=/workspace/rsna_knee/raptor_combined

run_one () {
  local TAG="$1" LABELS="$2"
  if [ -f "/workspace/raptor_ft_${TAG}.pt" ]; then echo "[chain] $TAG already done, skipping"; return; fi
  echo "[chain] $(date -u +%H:%M:%S) starting $TAG on $LABELS"
  python finetune_raptor_cnn.py --ckpt timm --norm imagenet --workers 8 \
    --tag "$TAG" --arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 \
    --labels "$LABELS" --slots '' \
    --k 12 --k_eval 24 --epochs 16 --bs 16 \
    --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 777 > "/workspace/train_${TAG}.log" 2>&1
  local RC=$? B
  B=$(grep -oE 'BEST GOLD macro-AUC [0-9.]+' "/workspace/train_${TAG}.log" | tail -1 | awk '{print $NF}')
  echo "${TAG} gold=${B:-NONE} rc=${RC}" >> /workspace/RESULTS224.txt
  echo "[chain] $(date -u +%H:%M:%S) $TAG done gold=${B:-NONE} rc=${RC}"
}

run_one_seed () {
  local TAG="$1" LABELS="$2" SEED="$3"
  if [ -f "/workspace/raptor_ft_${TAG}.pt" ]; then echo "[chain] $TAG already done, skipping"; return; fi
  echo "[chain] $(date -u +%H:%M:%S) starting $TAG on $LABELS seed $SEED"
  python finetune_raptor_cnn.py --ckpt timm --norm imagenet --workers 8     --tag "$TAG" --arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224     --labels "$LABELS" --slots ''     --k 12 --k_eval 24 --epochs 16 --bs 16     --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed "$SEED" > "/workspace/train_${TAG}.log" 2>&1
  local RC=$? B
  B=$(grep -oE 'BEST GOLD macro-AUC [0-9.]+' "/workspace/train_${TAG}.log" | tail -1 | awk '{print $NF}')
  echo "${TAG} gold=${B:-NONE} rc=${RC}" >> /workspace/RESULTS224.txt
  echo "[chain] $(date -u +%H:%M:%S) $TAG done gold=${B:-NONE} rc=${RC}"
}

# ORDER MATTERS. confdist@224 is already trained (gold 0.9040), and confdist+public is the exact
# pairing that holds the board at 0.941 with the 384 arms. So public goes FIRST: it completes the
# best two-arm set at 224 and can ship on its own if the rest never finish. Then the label sets
# that disagree most, then a seed-diverse sibling of the strongest set for a fourth arm.
run_one raptor224_pub   /workspace/labels_publab.parquet        # 1. completes the best PAIR
run_one raptor224_opus  /workspace/labels_opus.parquet          # 2. most decorrelated labels
run_one raptor224_blend /workspace/labels_opus_blend.parquet    # 3. best label set never trained
run_one_seed raptor224_cd999 /workspace/labels_cd_rsna.parquet 999   # 4. seed-diverse confdist

echo "[chain] ALL DONE"
cat /workspace/RESULTS224.txt
