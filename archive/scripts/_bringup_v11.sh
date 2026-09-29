#!/bin/bash
# Unattended bring-up: wait for the box, stage everything, assemble the v11 corpus, train two arms.
#
# Arm A isolates GEOMETRY: the champion recipe exactly, only the corpus changes (44 -> 96 slices).
# Arm B isolates RECIPE:   Arm A plus the things no shipped arm has ever had - augmentation,
#                          stochastic depth, macro-normalised loss, the pairwise rank term, and
#                          the per-slot embedding.
# Two arms are wanted anyway, because the kernel ships a pair.
#
# Usage: bash _bringup_v11.sh <contract_id>
set -u
C="${1:-50929431}"
K=/c/Users/johna/.ssh/id_ed25519
LOG=_bringup.log
exec > >(tee -a "$LOG") 2>&1
echo "=== bringup $(date -u +%H:%M:%S) contract $C ==="

# ---- 1. wait for ssh to be routable -------------------------------------------------
H=""; P=""
for i in $(seq 1 120); do
  J=$(vastai show instance "$C" --raw 2>/dev/null)
  H=$(echo "$J" | python -c "import sys,json;print(json.load(sys.stdin).get('public_ipaddr') or '')" 2>/dev/null)
  P=$(echo "$J" | python -c "import sys,json;d=json.load(sys.stdin);print((d.get('ports') or {}).get('22/tcp',[{}])[0].get('HostPort') or '')" 2>/dev/null)
  S=$(echo "$J" | python -c "import sys,json;print(json.load(sys.stdin).get('actual_status') or '')" 2>/dev/null)
  if [ -n "$P" ] && [ "$S" = "running" ]; then echo "[ssh] $H:$P status=$S"; break; fi
  echo "[wait $i] status=$S port=${P:-none}"; sleep 10
done
[ -n "$P" ] || { echo "ABORT: no ssh port after 20 min"; exit 1; }
SSH="ssh -i $K -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 root@$H"
SCP="scp -i $K -P $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
for i in $(seq 1 60); do $SSH 'echo up' 2>/dev/null | grep -q up && break; echo "[sshwait $i]"; sleep 10; done
$SSH 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -BG --output=avail /workspace 2>/dev/null | tail -1'

# ---- 2. stage code, labels, credentials ---------------------------------------------
echo "[stage] files"
$SSH 'mkdir -p /workspace/knee /root/.kaggle'
$SCP finetune_box.py _merge_v11.py _setup_box_v11.sh root@$H:/workspace/knee/
$SCP train.csv study_fingerprint.csv root@$H:/workspace/knee/
$SCP phase2/box_stage/labeler/labels_confdist.parquet root@$H:/workspace/knee/
$SCP /c/Users/johna/.kaggle/kaggle.json /c/Users/johna/.kaggle/access_token root@$H:/root/.kaggle/ 2>/dev/null
$SSH 'chmod 600 /root/.kaggle/* 2>/dev/null; pip -q install kaggle timm pandas pyarrow scikit-learn 2>&1 | tail -1; kaggle --version'

# ---- 3. assemble the 96-slice corpus -------------------------------------------------
echo "[corpus] assembling v11 (96 slices) - this is the whole point of the run"
$SSH 'cd /workspace/knee && python -u _merge_v11.py --out /workspace/knee/corpus_v11 2>&1 | tail -40'
$SSH 'python -c "
import numpy as np
v=np.load(\"/workspace/knee/corpus_v11/all_vols.npy\",mmap_mode=\"r\")
assert v.shape==(4407,96,336,336), v.shape
print(\"CORPUS VERIFIED\", v.shape)
"' || { echo "ABORT: corpus verification failed"; exit 1; }

# ---- 4. train ------------------------------------------------------------------------
# train.csv and the labels live in /workspace/knee; KNEE_DATA_ROOT's parent must be that dir.
COMMON="--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --ckpt timm --norm imagenet \
--labels /workspace/knee/labels_confdist.parquet --bs 8 --grad_ckpt \
--group_csv /workspace/knee/study_fingerprint.csv --group_folds 5 --group_fold 0 --reclaim 0.7 \
--k 24 --k_eval 24 --workers 4"

echo "[train] ARM A - champion recipe, v11 geometry. Isolates the corpus change."
$SSH "cd /workspace/knee && KNEE_DATA_ROOT=/workspace/knee/corpus_v11 nohup python -u finetune_box.py \
$COMMON --epochs 16 --bb_lr 3e-5 --head_lr 1e-3 --tag V11_GEOM_A > /workspace/knee/run_A.log 2>&1 &
echo started; sleep 5; tail -5 /workspace/knee/run_A.log"

echo "=== bringup done $(date -u +%H:%M:%S). Arm A training; poll run_A.log ==="
echo "    ssh: ssh -i $K -p $P root@$H"
