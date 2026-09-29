#!/bin/bash
H="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p 40101 root@84.8.116.228"
for i in $(seq 1 200); do
  R=$($H 'B=$(grep -oE "BEST GOLD macro-AUC [0-9.]+" /workspace/train_ext_full.log|tail -1);
          S=$(grep -c "\[swa\]" /workspace/train_ext_full.log 2>/dev/null);
          N=$(grep -c "GOLD macro-AUC" /workspace/train_ext_full.log 2>/dev/null);
          Q=$(grep -c "DONE" /workspace/queue.log 2>/dev/null);
          E=$(cat /workspace/train_*.log 2>/dev/null|grep -cE "Traceback|RuntimeError|out of memory");
          C=$(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,utilization.gpu --format=csv,noheader,nounits|tr "\n" " ");
          L=$(grep "GOLD macro-AUC" /workspace/train_ext_full.log|tail -1|sed "s/ loss.*AUC/ /;s/(best.*| / @/");
          echo "$N|$E|$B|$S|$C|$L|$Q"' 2>/dev/null)
  [ -z "$R" ] && { echo "$(date -u +%H:%M) !! BOX UNREACHABLE"; sleep 60; continue; }
  N=$(echo "$R"|cut -d"|" -f1); E=$(echo "$R"|cut -d"|" -f2); B=$(echo "$R"|cut -d"|" -f3); S=$(echo "$R"|cut -d"|" -f4)
  echo "$(date -u +%H:%M) ep=$N err=$E swa=$S gpu=[$(echo "$R"|cut -d"|" -f5)] $(echo "$R"|cut -d"|" -f6) best=$B"
  if [ "${E:-0}" -gt 0 ] 2>/dev/null; then echo "### ERROR"; break; fi
  if [ -n "$B" ]; then echo "### RUN 1 FINISHED"; break; fi
  sleep 100
done
