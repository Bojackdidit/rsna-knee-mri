#!/bin/bash
# Push the fast-arm payload to the box and start training. One shot, so nothing waits on a round trip.
set -u
H="$1"; P="$2"; K=/c/Users/johna/.ssh/ddp_ed25519
SSH="ssh -i $K -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$H"
SCP="scp -i $K -P $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
R="C:/Users/johna/AppData/Local/Temp/claude/D--Artificial-Nervous-System/efa12c9d-7852-4b73-9032-6dc08d47204d/scratchpad/rsna_knee"

echo "[1/5] deps"
$SSH 'pip -q install timm pandas pyarrow scikit-learn opencv-python-headless 2>&1 | tail -2; python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"'

echo "[2/5] small files"
$SCP finetune_box.py labels_cd_rsna.parquet "$R/train.csv" root@$H:/workspace/

echo "[3/5] corpus 3.2 GB  (this is the upload, timing it)"
date -u +%H:%M:%S
$SSH 'mkdir -p /workspace/corpus_s02'
$SCP -C corpus_s02/all_ids.npy corpus_s02/all_masks.npy root@$H:/workspace/corpus_s02/
time $SCP corpus_s02/all_vols.npy root@$H:/workspace/corpus_s02/
date -u +%H:%M:%S

echo "[4/5] verify"
$SSH 'python -c "
import numpy as np
v=np.load(\"/workspace/corpus_s02/all_vols.npy\",mmap_mode=\"r\")
print(\"corpus\",v.shape,v.dtype)
import os;print(\"train.csv\",os.path.getsize(\"/workspace/train.csv\"))
"'

echo "[5/5] train"
$SSH 'cd /workspace && KNEE_DATA_ROOT=/workspace/corpus_s02 nohup python -u finetune_box.py \
  --ckpt timm --norm imagenet --workers 8 --tag raptor224_s02 \
  --arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 \
  --labels /workspace/labels_cd_rsna.parquet --slots "0,1" \
  --k 12 --k_eval 24 --epochs 16 --bs 16 --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 999 \
  > /workspace/train_s02.log 2>&1 & sleep 5; echo started; tail -3 /workspace/train_s02.log'
