#!/bin/bash
set -u
cd /workspace
export KNEE_DATA_ROOT=/workspace/rsna_knee/raptor
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$USED" -gt 2000 ]; then echo "[sgate] ABORT: GPU holds ${USED} MiB"; exit 1; fi
echo "[sgate] $(date -u) GPU clear (${USED} MiB), starting"
python finetune_raptor_cnn6.py --ckpt timm --norm imagenet --grad_ckpt --workers 8 \
  --tag coat384_sgate --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --labels labels_confdist.parquet --slots 0,1,2,3,4 --k 12 --k_eval 24 --epochs 16 --bs 8 \
  --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 1234 --spatial_gate > train_coat384_sgate.log 2>&1
B=$(grep -oE 'BEST GOLD macro-AUC [0-9.]+' train_coat384_sgate.log | tail -1 | awk '{print $NF}')
echo "coat384_sgate gold=${B:-NONE}" >> RESULTS.txt
echo "[sgate] $(date -u) DONE gold=${B:-NONE}"
