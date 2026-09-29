#!/bin/bash
# Scanner-grouped A/B: does anti-scanner augmentation help generalisation to UNSEEN equipment?
# Identical settings in both arms; the only difference is --aug_scanner. Reduced batch/window
# count vs the champion recipe because this is a 24 GB 3090, not the 80 GB H100 it was tuned on -
# but both arms share the reduction, so the DELTA is what this measures.
set -u
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--ckpt timm --norm imagenet --workers 8 --grad_ckpt \
 --arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 \
 --labels /workspace/knee/labels_cd_ext.parquet \
 --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
 --k 6 --k_eval 12 --epochs 12 --bs 4 --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42"

echo "===== ARM 1/2: BASELINE (no anti-scanner aug) ====="
python -u finetune_box.py $COMMON --tag gate_base
echo "===== ARM 2/2: ANTI-SCANNER AUG ====="
python -u finetune_box.py $COMMON --aug_scanner --tag gate_aug
echo "===== BOTH ARMS DONE ====="
