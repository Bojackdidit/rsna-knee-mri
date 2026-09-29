#!/bin/bash
# Bring up a second box and run the CLEAN anti-scanner test.
#
# The run in flight on box 1 is NOT a pure single-variable test: I cut epochs 24 -> 16 to fit the old
# credit limit, and OneCycleLR rescales its whole schedule to the epoch count, so its ep14 sits at a
# very different learning rate than arm 2's ep14 did. This run removes that confound completely --
# same seed 4242, same labels, same corpus, same EMA, same LRs, same 24 epochs as arm 2, with
# --aug_scanner as the ONLY difference. Arm 2 (gold 0.9315) is the OFF arm, already measured.
set -u
C="${1:?contract id}"
K=/c/Users/johna/.ssh/id_ed25519
exec > >(tee -a _bringup_scanner24.log) 2>&1
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

echo "[train] SCANNER_E24 - arm 2's exact recipe at 24 epochs, --aug_scanner the only change"
$SSH "cd /workspace/knee && KNEE_DATA_ROOT=/workspace/knee/corpus_v11 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid nohup python -u finetune_box.py \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --ckpt timm --norm imagenet \
  --labels /workspace/knee/labels_confdist.parquet --bs 2 --grad_ckpt --k 24 --k_eval 24 \
  --epochs 24 --bb_lr 3e-5 --head_lr 1e-3 --workers 4 --seed 4242 \
  --aug geo,erase,blur,noise --aug_scanner --ema 0.9998 --tag SCANNER_E24 \
  > /workspace/knee/run_SCANNER_E24.log 2>&1 < /dev/null &
sleep 20; head -12 /workspace/knee/run_SCANNER_E24.log"
echo "=== bringup done $(date -u +%H:%M:%S) | ssh -i $K -p $P root@$H ==="
