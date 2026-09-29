#!/bin/bash
set -u
cd /workspace
export KNEE_DATA_ROOT=/workspace/rsna_knee/raptor
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$U" -gt 2000 ]; then echo "[publab] ABORT: GPU holds ${U} MiB"; exit 1; fi
# Single variable vs the 0.9224 control: same arch, res, slots, k, epochs, bs, lr, seed.
# ONLY the label file changes. Public balanced labels score 0.9124 against the 58 gold studies;
# our own LLM severity extraction scores 0.8708 on the same studies.
echo "[publab] $(date -u) starting, labels_publab.parquet"
python finetune_raptor_cnn5.py --ckpt timm --norm imagenet --workers 8 \
  --tag coat384_publab --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --labels /workspace/rsna_knee/labeler/labels_publab.parquet --slots 0,1,2,3,4 \
  --k 12 --k_eval 24 --epochs 16 --bs 8 \
  --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 1337 > train_coat384_publab.log 2>&1
B=$(grep -oE 'BEST GOLD macro-AUC [0-9.]+' train_coat384_publab.log | tail -1 | awk '{print $NF}')
echo "coat384_publab gold=${B:-NONE}" >> RESULTS.txt
echo "[publab] $(date -u) DONE gold=${B:-NONE}"
