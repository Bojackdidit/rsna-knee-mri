#!/bin/bash
# Replicate the crop result at a SECOND SEED. +0.0017 from one run each is inside the run-to-run
# spread on this recipe, so it is not a result until it repeats. Same grouped fold 0, same settings,
# only the seed changes - baseline and crop 0.85 head to head again.
set -u
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--ckpt timm --norm imagenet --workers 8 --grad_ckpt \
 --arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 \
 --labels /workspace/knee/labels_cd_ext.parquet \
 --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
 --k 6 --k_eval 12 --epochs 12 --bs 4 --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 777"
echo "===== SEED777 BASELINE (crop 1.00) ====="
python -u finetune_box.py $COMMON --tag s2_base
echo "===== SEED777 CROP 0.85 ====="
python -u finetune_box.py $COMMON --crop_frac 0.85 --tag s2_crop85
echo "===== SEED2 ARMS DONE ====="
