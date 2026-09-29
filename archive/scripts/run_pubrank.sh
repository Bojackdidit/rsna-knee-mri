#!/bin/bash
set -u
cd /workspace
export KNEE_DATA_ROOT=/workspace/rsna_knee/raptor
for i in $(seq 1 120); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$U" -lt 2000 ] && break
  sleep 30
done
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$U" -gt 2000 ]; then echo "[pubrank] ABORT: GPU holds ${U} MiB"; exit 1; fi
# Public labels rank BETTER against the 58 gold studies (0.9124 vs our 0.8708) but are much softer
# (far more mass in 0.3-0.7), which weakens the BCE gradient. This keeps the public ORDERING and
# adopts confdist's exact marginal DISTRIBUTION per finding. Same control: 0.9224.
echo "[pubrank] $(date -u) starting, labels_pubrank.parquet"
python finetune_raptor_cnn5.py --ckpt timm --norm imagenet --workers 8 \
  --tag coat384_pubrank --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --labels /workspace/rsna_knee/labeler/labels_pubrank.parquet --slots 0,1,2,3,4 \
  --k 12 --k_eval 24 --epochs 16 --bs 8 \
  --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 1337 > train_coat384_pubrank.log 2>&1
B=$(grep -oE 'BEST GOLD macro-AUC [0-9.]+' train_coat384_pubrank.log | tail -1 | awk '{print $NF}')
echo "coat384_pubrank gold=${B:-NONE}" >> RESULTS.txt
echo "[pubrank] $(date -u) DONE gold=${B:-NONE}"
