#!/bin/bash
# Run the arm queue back to back - no idle GPU between runs.
# combotest showed 3-4 arms beat 2 (0.9331 -> 0.9346 -> 0.9356), so the shipping lever is MORE
# good arms, not one perfect one. Same corpus, different seeds = diverse arms for the blend.
set -u
cd /workspace
export KNEE_DATA_ROOT=/workspace/combined
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

wait_for_gpu(){
  while pgrep -f "finetune_raptor_cnn9[.]py" > /dev/null; do sleep 60; done
  sleep 20
}

run(){  # tag seed labels
  wait_for_gpu
  log "START $1 (seed $2, labels $(basename $3))"
  python -u finetune_raptor_cnn9.py --ckpt timm --norm imagenet --workers 16 \
    --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --slots "" --k 12 --k_eval 24 \
    --epochs 16 --bs 8 --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed "$2" \
    --tag "$1" --labels "$3" > "/workspace/train_$1.log" 2>&1
  B=$(grep -oE "BEST GOLD macro-AUC [0-9.]+" "/workspace/train_$1.log" | tail -1 | awk '{print $NF}')
  log "DONE $1 gold=${B:-NONE}"
  echo "$1 gold=${B:-NONE}" >> /workspace/QUEUE_RESULTS.txt
  # dump the gate file immediately so a later failure can never cost us the measurement
  python -u dump_gold58.py --ckpt "/workspace/raptor_ft_$1.pt" --tag "$1" 2>&1 | tail -2
}

# 1) control: RSNA only, identical everything else - settles whether external data actually pays
run raptor_ctrl_rsna 777 /workspace/combined/labels_rsna_only.parquet
# 2,3) more external-data arms at different seeds - diverse members for a 3-4 arm blend
run raptor_ext_s999  999  /workspace/combined/labels_train.parquet
run raptor_ext_s1234 1234 /workspace/combined/labels_train.parquet
log "QUEUE COMPLETE"
cat /workspace/QUEUE_RESULTS.txt
