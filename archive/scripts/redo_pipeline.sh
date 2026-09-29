#!/bin/bash
# Full unattended redo: wait for the box, ship the corpus in parallel, train at the reference batch
# size, then pull the arm back and PROVE it with a hash before the box is destroyed.
#
# The first attempt lost its arm because the fetch chunked with integer-megabyte dd offsets that do
# not tile the file, and the only check was a size that had been forced by head -c. Nothing here is
# destroyed until sha256 matches end to end.
set -u
ID=50349021; IP=185.65.93.114
K=/c/Users/johna/.ssh/ddp_ed25519
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

log "waiting for port assignment"
for i in $(seq 1 60); do
  P=$(vastai show instances --raw 2>/dev/null | python -c "
import sys,json
for i in json.loads(sys.stdin.read()):
    if i['id']==$ID: print(i.get('direct_port_start') or -1)
" 2>/dev/null | tr -d '\r')
  [ "${P:--1}" != "-1" ] && [ -n "$P" ] && break
  sleep 20
done
log "direct port $P"
vastai attach ssh $ID "$(cat $K.pub)" >/dev/null 2>&1
SSH="ssh -i $K -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o Compression=no"

log "waiting for ssh"
for i in $(seq 1 90); do $SSH root@$IP 'echo UP' 2>/dev/null | grep -q UP && break; sleep 20; done
$SSH root@$IP 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
log "installing deps"
$SSH root@$IP 'pip -q install timm pandas pyarrow scikit-learn opencv-python-headless 2>&1|tail -1'

log "small files"
scp -i $K -P $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null finetune_box.py labels_cd_rsna.parquet train.csv root@$IP:/workspace/ >/dev/null 2>&1
$SSH root@$IP 'mkdir -p /workspace/corpus_s02'
scp -i $K -P $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null corpus_s02/all_ids.npy corpus_s02/all_masks.npy root@$IP:/workspace/corpus_s02/ >/dev/null 2>&1

log "corpus, 8 parallel streams"
for i in $(seq 0 7); do
  ( dd if=corpus_s02/all_vols.npy bs=1M skip=$((i*383)) count=383 2>/dev/null | $SSH root@$IP "cat > /workspace/c$i.bin" ) &
done
wait
$SSH root@$IP 'cat /workspace/c0.bin /workspace/c1.bin /workspace/c2.bin /workspace/c3.bin /workspace/c4.bin /workspace/c5.bin /workspace/c6.bin /workspace/c7.bin > /workspace/corpus_s02/all_vols.npy && rm -f /workspace/c*.bin'
RS=$($SSH root@$IP "stat -c '%s' /workspace/corpus_s02/all_vols.npy" 2>/dev/null | tr -d '\r')
log "corpus on box: $RS (expect 3211264128)"
[ "$RS" != "3211264128" ] && { log "CORPUS SIZE MISMATCH - stopping, box left alive"; exit 1; }

log "training, bs 16 (reference recipe, 80GB card)"
$SSH root@$IP 'cd /workspace && setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True KNEE_DATA_ROOT=/workspace/corpus_s02 python -u finetune_box.py --ckpt timm --norm imagenet --workers 8 --tag raptor224_s02 --arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 --labels /workspace/labels_cd_rsna.parquet --slots "0,1" --k 12 --k_eval 24 --epochs 16 --bs 16 --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 999 > /workspace/train_s02.log 2>&1 < /dev/null & disown; sleep 5; echo started'
for i in $(seq 1 200); do
  $SSH root@$IP 'ls /workspace/raptor_ft_raptor224_s02.pt >/dev/null 2>&1 && echo SAVED' 2>/dev/null | grep -q SAVED && break
  $SSH root@$IP 'grep -q "OutOfMemory\|Traceback" /workspace/train_s02.log 2>/dev/null && echo FAILED' 2>/dev/null | grep -q FAILED && { log "TRAINING FAILED"; $SSH root@$IP 'tail -5 /workspace/train_s02.log'; exit 1; }
  sleep 45
done
$SSH root@$IP 'grep -E "^ep[0-9]" /workspace/train_s02.log | tail -3'

log "fetching arm with hash verification"
bash fetch_arm.sh $IP $P /workspace/raptor_ft_raptor224_s02.pt arms224/raptor_ft_raptor224_s02.pt
if [ $? -ne 0 ]; then log "FETCH FAILED - box $ID LEFT ALIVE for retry"; exit 1; fi
$SSH root@$IP 'cat /workspace/raptor_gold_raptor224_s02.npz' > goldnew/raptor_gold_raptor224_s02.npz 2>/dev/null
log "arm verified; destroying box $ID"
yes | vastai destroy instance $ID 2>&1 | tail -1
log "DONE"
