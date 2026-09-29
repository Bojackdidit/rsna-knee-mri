#!/bin/bash
# Second CoAtNet arm, queued behind the first. Same fixes, DIFFERENT label set and seed, so the pair
# disagrees the way the champion pair does. Costs nothing at inference: the kernel puts one arm on
# each T4 and they run concurrently, so wall clock is the slower arm, not the sum.
set -u
cd /workspace/knee
echo "[arm2] waiting for COAT384_FIX $(date -u +%H:%M:%S)"
while pgrep -f "[f]inetune_box.py" > /dev/null 2>&1; do sleep 60; done
echo "[arm2] arm 1 done $(date -u +%H:%M:%S) | $(grep -aoE 'best 0[.][0-9]+' run_COAT384_FIX.log 2>/dev/null | tail -1)"
# GEOMETRY CHECK before committing another 5 hours. The 3-series board read (0.915) could not
# tell us whether 3 series is genuinely costly or those arms were simply weak - and arm 1 is
# trained at 3 series, so if the geometry is wrong BOTH arms are compromised. Score arm 1 on the
# geometry it was trained on and on the full 5, over the same studies. It was TRAINED on 3, so 3
# should win; if 5 wins clearly the geometry is costing us and the plan needs changing.
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
echo "[arm2] geometry check on arm 1"
GE_IDS=valids_COAT384_FIX.csv GE_K=12 GE_SLOTS=0,2,4 python -u gate_eval.py raptor_ft_COAT384_FIX.pt A1_3SER 2>&1 | tail -2
GE_IDS=valids_COAT384_FIX.csv GE_K=12 GE_SLOTS= python -u gate_eval.py raptor_ft_COAT384_FIX.pt A1_5SER 2>&1 | tail -2
echo "[arm2] ^ arm 1 was TRAINED on 3 series, so 3 should win. If 5 wins clearly, rethink."
# Like-for-like against the best small model. Arm 1 validates on 350 studies (it reclaimed the rest
# of its fold) and E1b on 870 - not comparable as they stand. But E1b held out ALL 870, which
# contains arm 1's 350, so scoring E1b on exactly those 350 is honest for both and gives the first
# direct read of CoAtNet+fixes against EfficientNetV2+fixes on identical patients.
GE_IDS=valids_COAT384_FIX.csv GE_K=12 GE_SLOTS=0,2,4 python -u gate_eval.py raptor_ft_E1b_v2s336_3ser.pt E1B_ON350 2>&1 | tail -2
echo "[arm2] ^ E1b on arm 1 s own holdout - same patients, so these two numbers ARE comparable"

BB_LR=3e-5 GRAD_CKPT=1 RECLAIM=0.7 SEED=1337 LABELS=/workspace/knee/labels_cd_ext.parquet \
  bash _run_spec.sh coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k 384 "0,2,4" 12 16 0 0 0 0 8 COAT384_FIX_B \
  > run_COAT384_FIX_B.log 2>&1
echo "[arm2] done $(date -u +%H:%M:%S) | $(grep -aoE 'best 0[.][0-9]+' run_COAT384_FIX_B.log 2>/dev/null | tail -1)"
