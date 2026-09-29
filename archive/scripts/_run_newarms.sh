#!/bin/bash
# NEW ARMS, built for the operating point we actually ship at - not adapted, not swapped.
#
# Everything shipped so far is an arm trained with k=6 windows and then inferred at K_EVAL=48. That
# is a train/test mismatch nobody ever fixed, and it is why cutting windows costs accuracy: the arm
# was never taught to decide from few windows. Every "faster" attempt so far changed the BACKBONE and
# paid 4.2x in accuracy; the one that changed the window geometry by ADAPTATION failed outright
# (0.9182 at ep2 against 0.9340). So build the arm for the budget instead.
#
# Three changes together, all measured, none guessed:
#   res 224   - 0.72x the per-window cost of 384 at the measured res^0.6 scaling
#   k = k_eval = 12  - trained AT the inference window budget, so no mismatch
#   blend labels - the ONLY effect that cleared the paired-bootstrap noise floor on this gate
#                  (+0.0075, sd 0.0033; architecture and seed did not)
#
# Reference to beat: the shipped champion arm evaluated at K=12 scores 0.9375 on this exact gate.
# A purpose-built arm that beats that is FASTER (224, twelve windows) AND BETTER at the same time,
# which is the whole point. Grouped fold 0 held out so the number is honest.
set -u
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--ckpt timm --norm imagenet --workers 6 --grad_ckpt \
 --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
 --arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 \
 --k 12 --k_eval 12 --epochs 14 --bs 4 --head_lr 1e-3 --wd 0.02"

echo "===== ARM A: k12 @224, BLEND labels ====="
python -u finetune_box.py $COMMON --labels /workspace/knee/labels_opus_blend.parquet \
  --bb_lr 3e-5 --seed 42 --tag new224_k12_blend

echo "===== ARM B: k12 @224, cd_ext labels (label control) ====="
python -u finetune_box.py $COMMON --labels /workspace/knee/labels_cd_ext.parquet \
  --bb_lr 3e-5 --seed 42 --tag new224_k12_cd

echo "===== NEW ARMS DONE $(date -u +%H:%M:%S) ====="
for t in new224_k12_blend new224_k12_cd; do
  echo "$t: $(grep -aoE 'best 0\.[0-9]+' /workspace/knee/newarms.log | tail -1)"
done
