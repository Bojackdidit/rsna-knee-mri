#!/bin/bash
set -u
cd /workspace
export KNEE_DATA_ROOT=/workspace/rsna_knee/raptor
for i in $(seq 1 40); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$U" -lt 2000 ] && break
  sleep 15
done
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$U" -gt 2000 ]; then echo "[bblr50] ABORT: GPU holds ${U} MiB"; exit 1; fi
echo "[bblr50] $(date -u) starting bb_lr 5e-4 (second point on the curve; H100 runs 1.5e-4)"
python finetune_raptor_cnn5.py --ckpt timm --norm imagenet --grad_ckpt --workers 8 \
  --tag coat384_bblr50 --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --labels labels_confdist.parquet --slots 0,1,2,3,4 --k 12 --k_eval 24 --epochs 16 --bs 8 \
  --bb_lr 5e-4 --head_lr 1e-3 --wd 0.02 --seed 2026 > train_coat384_bblr50.log 2>&1
B=$(grep -oE 'BEST GOLD macro-AUC [0-9.]+' train_coat384_bblr50.log | tail -1 | awk '{print $NF}')
echo "coat384_bblr50 gold=${B:-NONE}" >> RESULTS.txt
echo "[bblr50] $(date -u) DONE gold=${B:-NONE}"
