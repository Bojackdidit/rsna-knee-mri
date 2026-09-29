#!/bin/bash
H="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p 40101 root@84.8.116.228"
for i in $(seq 1 400); do
  R=$($H 'CUR=$(ps -eo cmd|grep -oE "\-\-tag [a-z0-9_]+"|head -1|awk "{print \$2}");
          L=$(ls -t /workspace/train_*.log 2>/dev/null|head -1);
          EP=$(grep "GOLD macro-AUC" $L 2>/dev/null|tail -1|sed "s/ loss.*AUC/ /;s/(best.*| / @/");
          N=$(grep -c "GOLD macro-AUC" $L 2>/dev/null);
          TR=$(grep -oE "train [0-9]+ . gold-val [0-9]+" $L 2>/dev/null|tail -1|tr "|" "/");
          E=$(cat /workspace/train_*.log 2>/dev/null|grep -cE "Traceback|RuntimeError|out of memory");
          C=$(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,utilization.gpu --format=csv,noheader,nounits|tr "\n" " ");
          Q=$(pgrep -c -f "[_]queue2.sh"); A=$(pgrep -c -f "finetune_raptor_cnn9[.]py");
          RES=$(cat /workspace/QUEUE_RESULTS.txt 2>/dev/null|tr "\n" "; ");
          echo "$CUR|$N|$EP|$TR|$E|$C|$Q|$A|$RES"' 2>/dev/null)
  [ -z "$R" ] && { echo "$(date -u +%H:%M) !! BOX UNREACHABLE"; sleep 60; continue; }
  echo "$(date -u +%H:%M) tag=$(echo "$R"|cut -d"|" -f1) ep=$(echo "$R"|cut -d"|" -f2)$(echo "$R"|cut -d"|" -f3) [$(echo "$R"|cut -d"|" -f4)] err=$(echo "$R"|cut -d"|" -f5) gpu=[$(echo "$R"|cut -d"|" -f6)] q2=$(echo "$R"|cut -d"|" -f7) train=$(echo "$R"|cut -d"|" -f8) done=$(echo "$R"|cut -d"|" -f9)"
  E=$(echo "$R"|cut -d"|" -f5); Q=$(echo "$R"|cut -d"|" -f7); A=$(echo "$R"|cut -d"|" -f8)
  if [ "${E:-0}" -gt 0 ] 2>/dev/null; then echo "### ERROR"; break; fi
  if [ "${Q:-0}" -eq 0 ] && [ "${A:-0}" -eq 0 ] 2>/dev/null; then echo "### ALL RUNS COMPLETE"; break; fi
  sleep 180
done
