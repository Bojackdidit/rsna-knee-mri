#!/bin/bash
# Requeue on confdist - the best label set we have measured (0.9263 arm last night).
# llm_soft differs from confdist on 53.5% of cells, so today's runs were on weaker labels.
set -u
cd /workspace
export KNEE_DATA_ROOT=/workspace/combined
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
wait_for_gpu(){ while pgrep -f "finetune_raptor_cnn9[.]py" > /dev/null; do sleep 60; done; sleep 20; }
run(){
  wait_for_gpu
  log "START $1 (seed $2, labels $(basename $3))"
  python -u finetune_raptor_cnn9.py --ckpt timm --norm imagenet --workers 16 \
    --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --slots "" --k 12 --k_eval 24 \
    --epochs 16 --bs 8 --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed "$2" \
    --tag "$1" --labels "$3" > "/workspace/train_$1.log" 2>&1
  B=$(grep -oE "BEST GOLD macro-AUC [0-9.]+" "/workspace/train_$1.log" | tail -1 | awk '{print $NF}')
  log "DONE $1 gold=${B:-NONE}"
  echo "$1 gold=${B:-NONE}" >> /workspace/QUEUE_RESULTS.txt
  python -u dump_gold58.py --ckpt "/workspace/raptor_ft_$1.pt" --tag "$1" --data /workspace/combined 2>&1 | tail -2
}
run raptor_cd_ext  777 /workspace/combined/labels_cd_ext.parquet
run raptor_cd_rsna 777 /workspace/combined/labels_cd_rsna.parquet
log "QUEUE2 COMPLETE"; cat /workspace/QUEUE_RESULTS.txt
