#!/bin/bash
# Second scanner-grouped A/B: the ROI CROP lever. finetune_box.py's own comment calls it "the one
# lever on our pipeline still outside the ~0.008 band where seven architectural levers all died",
# and a search of all 50 board submissions finds ZERO mentioning crop/zoom/140mm/centre - it has
# never been fired. Same grouped fold 0, same everything else, so the delta is the crop.
set -u
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# wait for the aug A/B to finish, by MARKER not by pgrep (pgrep -f self-matches the waiting shell)
while ! grep -q "BOTH ARMS DONE" /workspace/knee/gate.log 2>/dev/null; do sleep 30; done
COMMON="--ckpt timm --norm imagenet --workers 8 --grad_ckpt \
 --arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 \
 --labels /workspace/knee/labels_cd_ext.parquet \
 --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
 --k 6 --k_eval 12 --epochs 12 --bs 4 --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42"
for CF in 0.85 0.70; do
  echo "===== CROP_FRAC $CF ====="
  python -u finetune_box.py $COMMON --crop_frac $CF --tag gate_crop$CF
done
echo "===== CROP ARMS DONE ====="
