#!/bin/bash
# A/B: does external data help when integrated PROPERLY (no blank slots)?
# Only difference is whether MRNet's 1,124 studies are in the label file.
set -u
cd /workspace
export KNEE_DATA_ROOT=/workspace/combined
D=/workspace/combined

python - <<'PY'
import pandas as pd
d = pd.read_parquet("/workspace/combined/labels_train.parquet")
d["StudyInstanceUID"] = d["StudyInstanceUID"].astype(str)
r = d[~d["StudyInstanceUID"].str.startswith("mrnet_")]
r.to_parquet("/workspace/combined/labels_rsna_only.parquet", index=False)
print("labels_train %d -> rsna_only %d (mrnet %d)" % (len(d), len(r), len(d)-len(r)))
PY

# array keeps the empty --slots argument intact through expansion
COMMON=(--ckpt timm --norm imagenet --workers 12
        --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384
        --slots "" --k 12 --k_eval 24 --epochs 16 --bs 8
        --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 777)

CUDA_VISIBLE_DEVICES=0 nohup python -u finetune_raptor_cnn8.py "${COMMON[@]}" \
  --tag raptor_ext_mrnet --labels "$D/labels_train.parquet" \
  > /workspace/train_ext_mrnet.log 2>&1 &
echo "GPU0 -> raptor_ext_mrnet (RSNA + MRNet, slots fixed)  pid $!"

CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_raptor_cnn8.py "${COMMON[@]}" \
  --tag raptor_ctrl_rsna --labels "$D/labels_rsna_only.parquet" \
  > /workspace/train_ctrl_rsna.log 2>&1 &
echo "GPU1 -> raptor_ctrl_rsna (RSNA only, control)  pid $!"
