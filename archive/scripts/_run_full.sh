#!/bin/bash
# The two arms that decide Dread's bar, retrained on the FULL 4,407-study corpus.
#
# Every arm trained today used `knee-raptor-corpus` - which is named "part 1 of 2" and holds 3,200
# studies. Part 2 (`knee-raptor-corpus-ext`, 1,207 more) was never downloaded. After the grouped fold
# was held out that left 2,524 training studies against the 4,349 the shipped 0.941 arms had: 58% of
# the data. Every gate number produced today, including the EfficientNet verdict, was measured on a
# crippled training set - so they get re-measured before anything is concluded from them.
#
#   eff_224_k6   the 4-minute candidate: Scott's family, k=k_eval=6, blend labels, bb_lr 3e-4.
#                On the partial corpus it scored 0.8728 on common targets.
#   coat224_ctl  the reference, identical settings otherwise. On the partial corpus the shipped
#                champion arm reads 0.9330 at K=6 - but that arm was TRAINED on the full corpus, so
#                comparing it to a part-1 arm was never like-for-like. This control fixes that.
#
# Both hold out grouped fold 0 so the numbers stay honest, and both dump npz so they can be re-scored
# against ONE common target set (eff trains on blend labels, which the gate would otherwise flatter).
set -u
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--ckpt timm --norm imagenet --workers 5 \
 --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
 --res 224 --k 6 --k_eval 6 --epochs 16 --bs 4 --head_lr 1e-3 --wd 0.02 --seed 42"

echo "===== FULL-CORPUS RUNS, KNEE_DATA_ROOT=$KNEE_DATA_ROOT ====="
nohup python -u finetune_box.py $COMMON \
  --arch tf_efficientnetv2_s.in21k_ft_in1k --bb_lr 3e-4 \
  --labels /workspace/knee/labels_opus_blend.parquet --tag full_eff_k6 > arm_full_eff_k6.log 2>&1 &
sleep 20
nohup python -u finetune_box.py $COMMON --grad_ckpt \
  --arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --bb_lr 3e-5 \
  --labels /workspace/knee/labels_opus_blend.parquet --tag full_coat_k6 > arm_full_coat_k6.log 2>&1 &
wait
echo "===== FULL-CORPUS RUNS DONE $(date -u +%H:%M:%S) ====="
for t in full_eff_k6 full_coat_k6; do
  echo "$t: $(grep -aoE 'best 0\.[0-9]+' arm_$t.log | tail -1)"
done
