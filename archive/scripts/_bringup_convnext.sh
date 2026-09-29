#!/bin/bash
# Bring up a third box and test a DIFFERENT ARCHITECTURE on the correct corpus.
#
# Why. The architecture ranking we rely on (CoAtNet 0.941 > ConvNeXt-B 0.923 > effv2s 0.918 > ViT
# 0.914) was measured on arms trained against the 44-slice v2 corpus, and the geometry table states
# plainly that arms on that corpus "cap out at 0.928". So ConvNeXt-Base has very likely never had a
# fair run: its 0.923 may be a geometry ceiling rather than an architecture ceiling.
#
# This matters because the ensemble axis is open and every partner we have ever shipped was a
# PASSENGER (arm 1 is 0.0055 below arm 2, SEED3 0.0072 below, SCANNER_ARM 0.0068 below). The recipe's
# normal output is 0.924-0.930 and arm 2's 0.9315 is an outlier, so another CoAtNet seed is a lottery.
# A different architecture at genuine parity is both a candidate arm AND the decorrelated partner that
# Scott's single 0.949 -> ensemble 0.954 progression says is worth about +0.005.
#
# Everything else matches arm 2 exactly: 24 epochs, seed 4242, same labels, corpus, EMA, LRs, and the
# same augmentation stack. The kernel requires every arm to share resolution 384, and this does.
set -u
C="${1:?contract id}"
K=/c/Users/johna/.ssh/id_ed25519
exec > >(tee -a _bringup_convnext.log) 2>&1
echo "=== bringup $(date -u +%H:%M:%S) contract $C ==="
H=""; P=""
for i in $(seq 1 120); do
  J=$(PYTHONIOENCODING=utf-8 vastai show instance "$C" --raw 2>/dev/null)
  H=$(echo "$J" | python -c "import sys,json;print(json.load(sys.stdin).get('public_ipaddr') or '')" 2>/dev/null)
  P=$(echo "$J" | python -c "import sys,json;d=json.load(sys.stdin);print((d.get('ports') or {}).get('22/tcp',[{}])[0].get('HostPort') or '')" 2>/dev/null)
  S=$(echo "$J" | python -c "import sys,json;print(json.load(sys.stdin).get('actual_status') or '')" 2>/dev/null)
  if [ -n "$P" ] && [ "$S" = "running" ]; then echo "[ssh] $H:$P"; break; fi
  echo "[wait $i] status=$S port=${P:-none}"; sleep 10
done
[ -n "$P" ] || { echo "ABORT: no ssh port"; exit 1; }
SSH="ssh -i $K -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 root@$H"
SCP="scp -i $K -P $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
for i in $(seq 1 60); do $SSH 'echo up' 2>/dev/null | grep -q up && break; echo "[sshwait $i]"; sleep 10; done
$SSH 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -BG --output=avail /workspace | tail -1'

echo "[stage] files"
$SSH 'mkdir -p /workspace/knee /root/.kaggle'
$SCP finetune_box_patched.py root@$H:/workspace/knee/finetune_box.py
$SCP _merge_v11.py train.csv study_fingerprint.csv root@$H:/workspace/knee/
$SCP phase2/box_stage/labeler/labels_confdist.parquet root@$H:/workspace/knee/
$SCP /c/Users/johna/.kaggle/kaggle.json root@$H:/root/.kaggle/ 2>/dev/null
$SCP /c/Users/johna/.kaggle/access_token root@$H:/root/.kaggle/ 2>/dev/null
$SSH 'chmod 600 /root/.kaggle/* 2>/dev/null; pip -q install kaggle timm pandas pyarrow scikit-learn 2>&1 | tail -1'

echo "[corpus] assembling v11 (96 slices, 336x336, 47.8 GB) - about two hours"
$SSH 'cd /workspace/knee && python -u _merge_v11.py --out /workspace/knee/corpus_v11 2>&1 | tail -30'
$SSH 'python -c "
import numpy as np
v=np.load(\"/workspace/knee/corpus_v11/all_vols.npy\",mmap_mode=\"r\")
assert v.shape==(4407,96,336,336), v.shape
print(\"CORPUS VERIFIED\", v.shape)
"' || { echo "ABORT: corpus verification failed"; exit 1; }

echo "[train] CONVNEXT_E24 - a DIFFERENT architecture on the correct 96-slice corpus"
$SSH "cd /workspace/knee && KNEE_DATA_ROOT=/workspace/knee/corpus_v11 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid nohup python -u finetune_box.py \
  --arch convnext_base.fb_in22k_ft_in1k_384 --res 384 --ckpt timm --norm imagenet \
  --labels /workspace/knee/labels_confdist.parquet --bs 2 --grad_ckpt --k 24 --k_eval 24 \
  --epochs 24 --bb_lr 3e-5 --head_lr 1e-3 --workers 4 --seed 4242 \
  --aug geo,erase,blur,noise --ema 0.9998 --tag CONVNEXT_E24 \
  > /workspace/knee/run_CONVNEXT_E24.log 2>&1 < /dev/null &
sleep 20; head -12 /workspace/knee/run_CONVNEXT_E24.log"
echo "=== bringup done $(date -u +%H:%M:%S) | ssh -i $K -p $P root@$H ==="
