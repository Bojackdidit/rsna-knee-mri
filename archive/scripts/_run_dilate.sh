#!/bin/bash
# ADAPT the champion arm to DILATED windows, so fewer windows carry the same evidence.
#
# The <=8 min bar is blocked by arithmetic, not by effort: reading is a FIXED 3.7 min, leaving 4.3 min
# of model, i.e. K<=10 windows, i.e. 20 positions covered even with interleaving - which measures
# ~0.935, not 0.941. More WINDOWS cannot get there. More EVIDENCE PER WINDOW can.
#
# A window is currently three ADJACENT slices [c-1, c, c+1]. Dilating it to [c-3, c, c+3] keeps three
# channels - identical cost per forward - but spans seven slices instead of three. Nine dilated
# windows then cover what twenty-seven tight ones do.
#
# ADAPTED from the shipped 0.941 arm rather than retrained: --init_arm loads the full finetuned
# RaptorClassifier, so the arm keeps its quality and only learns the new input geometry. Low backbone
# LR for the same reason.
#
# Grouped fold 0 held out ON PURPOSE: this run exists to MEASURE whether dilation works, and shipping
# it blind is exactly the mistake to avoid. If it wins, retrain on the full corpus for shipping.
set -u
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--ckpt timm --norm imagenet --workers 6 --grad_ckpt \
 --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
 --labels /workspace/knee/labels_cd_ext.parquet \
 --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
 --k 9 --k_eval 9 --epochs 5 --bs 4 --bb_lr 1e-5 --head_lr 2e-4 --wd 0.02 --seed 42 \
 --init_arm /workspace/arms/raptor_ft_confdist_swa.pt"
echo "===== DILATE 3 (adapted) ====="
python -u finetune_box.py $COMMON --slice_dilate 3 --tag dil3_k9
echo "===== DILATE 1 control (adapted, same budget) ====="
python -u finetune_box.py $COMMON --slice_dilate 1 --tag dil1_k9
echo "===== DILATE RUNS DONE $(date -u +%H:%M:%S) ====="
for t in dil3_k9 dil1_k9; do echo "$t: $(grep -aoE 'best 0\.[0-9]+' arm_$t.log | tail -1)"; done
