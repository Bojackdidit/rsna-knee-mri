#!/bin/bash
set -u
cd /workspace
export KNEE_DATA_ROOT=/workspace/rsna_knee/raptor
# Wait for the spatial-gate run to finish and the GPU to actually release.
for i in $(seq 1 200); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$U" -lt 2000 ]; then echo "[bblr] $(date -u) GPU free (${U} MiB)"; break; fi
  sleep 30
done
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$U" -gt 2000 ]; then echo "[bblr] ABORT: GPU still holds ${U} MiB"; exit 1; fi
# WHY: measured that confdist_swa and w80_full_swa differ by only 0.0220 in the BACKBONE
# (72.9M params) and 0.8953 in the HEAD (0.28M). At bb_lr 3e-5 the backbone barely leaves its
# ImageNet init, so every "arm" is the same frozen feature extractor wearing a different small
# head - which is why eight configurations all land on 0.940/0.941 and every blend is a wash.
# This raises bb_lr 5x. Control is the 16ep confdist run at 0.9224 on the same 45-study gate,
# same labels, same slots, same epochs - bb_lr is the only difference.
echo "[bblr] $(date -u) starting bb_lr 1.5e-4"
python finetune_raptor_cnn5.py --ckpt timm --norm imagenet --grad_ckpt --workers 8 \
  --tag coat384_bblr15 --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --labels labels_confdist.parquet --slots 0,1,2,3,4 --k 12 --k_eval 24 --epochs 16 --bs 8 \
  --bb_lr 1.5e-4 --head_lr 1e-3 --wd 0.02 --seed 2026 > train_coat384_bblr15.log 2>&1
B=$(grep -oE 'BEST GOLD macro-AUC [0-9.]+' train_coat384_bblr15.log | tail -1 | awk '{print $NF}')
echo "coat384_bblr15 gold=${B:-NONE}" >> RESULTS.txt
echo "[bblr] $(date -u) DONE gold=${B:-NONE}"
