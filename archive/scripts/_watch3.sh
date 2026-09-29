#!/bin/bash
# Watch the full pipeline. No response = explicit failure. Trips on degraded clocks too.
H="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p 40101 root@84.8.116.228"
for i in $(seq 1 300); do
  T=$(date -u +%H:%M)
  OUT=$($H 'STAGE=$(grep -oE "^\[[0-9:]+\] [a-zA-Z].*" /workspace/pipeline.log 2>/dev/null|tail -1|cut -c12-60);
            EP=$(grep "GOLD macro-AUC" /workspace/train_ext_full.log 2>/dev/null|tail -1|sed "s/ loss.*AUC/ /;s/(best.*| / @/");
            NE=$(grep -c "GOLD macro-AUC" /workspace/train_ext_full.log 2>/dev/null);
            ER=$(grep -cE "Traceback|RuntimeError|out of memory|refusing" /workspace/pipeline.log /workspace/train_ext_full.log 2>/dev/null|awk -F: "{s+=\$2} END{print s+0}");
            CK=$(ls -la /workspace/raptor_ft_raptor_ext_full.pt 2>/dev/null|awk "{print \$5}");
            G=$(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,utilization.gpu --format=csv,noheader,nounits|tr "\n" " ");
            DL=$(tail -1 /workspace/pipeline.log 2>/dev/null|grep -oE "[0-9.]+/265.0 GB"|head -1);
            echo "err=$ER gpu[$G] ckpt=${CK:-none} | $STAGE | $DL | ep($NE)$EP"' 2>/dev/null)
  if [ -z "$OUT" ]; then echo "$T !! BOX UNREACHABLE - FAILURE not quiet"; else echo "$T $OUT"; fi
  case "$OUT" in *"swa"*) : ;; esac
  sleep 120
done
