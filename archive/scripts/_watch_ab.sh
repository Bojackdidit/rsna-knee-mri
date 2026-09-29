#!/bin/bash
# Active watcher: silence is an explicit failure, not "quiet".
H="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p 28561 root@45.135.56.11"
for i in $(seq 1 200); do
  T=$(date -u +%H:%M)
  OUT=$($H 'E=$(grep -c "GOLD macro-AUC" /workspace/train_ext_mrnet.log 2>/dev/null||echo 0);
            C=$(grep -c "GOLD macro-AUC" /workspace/train_ctrl_rsna.log 2>/dev/null||echo 0);
            EL=$(grep "GOLD macro-AUC" /workspace/train_ext_mrnet.log 2>/dev/null|tail -1);
            CL=$(grep "GOLD macro-AUC" /workspace/train_ctrl_rsna.log 2>/dev/null|tail -1);
            ERR=$(grep -cE "Traceback|RuntimeError|CUDA out of memory" /workspace/train_ext_mrnet.log /workspace/train_ctrl_rsna.log 2>/dev/null|awk -F: "{s+=\$2} END{print s+0}");
            G=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits|tr "\n" "/");
            N=$(pgrep -c -f finetune_raptor 2>/dev/null||echo 0);
            SWA=$(grep -c "\[swa\]" /workspace/train_ext_mrnet.log /workspace/train_ctrl_rsna.log 2>/dev/null|awk -F: "{s+=\$2} END{print s+0}");
            echo "ep_ext=$E ep_ctrl=$C err=$ERR gpu=$G proc=$N swa=$SWA|EXT:$EL|CTRL:$CL"' 2>/dev/null)
  if [ -z "$OUT" ]; then echo "$T ‼ BOX DID NOT RESPOND - FAILURE, not quiet"; else echo "$T $OUT"; fi
  case "$OUT" in *"swa=2"*|*"swa=3"*|*"swa=4"*) echo "$T ### BOTH RUNS REACHED SWA"; break;; esac
  case "$OUT" in *"proc=0"*) echo "$T ### NO TRAINER PROCESS - runs ended"; break;; esac
  sleep 100
done
