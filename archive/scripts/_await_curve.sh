#!/bin/bash
# Block until the curve is readable (ep6+) or something breaks. Reports every poll so no silence.
H="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p 40101 root@84.8.116.228"
for i in $(seq 1 200); do
  R=$($H 'N=$(grep -c "GOLD macro-AUC" /workspace/train_ext_full.log 2>/dev/null);
          L=$(grep "GOLD macro-AUC" /workspace/train_ext_full.log 2>/dev/null|tail -1|sed "s/ loss.*AUC/ /;s/(best.*| / @/");
          E=$(grep -cE "Traceback|RuntimeError|out of memory" /workspace/train_ext_full.log 2>/dev/null);
          C=$(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,utilization.gpu --format=csv,noheader,nounits|tr "\n" " ");
          Q=$(pgrep -c -f "[_]queue.sh");
          echo "$N|$E|$L|$C|$Q"' 2>/dev/null)
  [ -z "$R" ] && { echo "$(date -u +%H:%M) !! BOX UNREACHABLE - FAILURE not quiet"; sleep 60; continue; }
  N=$(echo "$R"|cut -d"|" -f1); E=$(echo "$R"|cut -d"|" -f2)
  echo "$(date -u +%H:%M) ep=$N err=$E queue=$(echo "$R"|cut -d"|" -f5) gpu=[$(echo "$R"|cut -d"|" -f4)] $(echo "$R"|cut -d"|" -f3)"
  if [ "${E:-0}" -gt 0 ] 2>/dev/null; then echo "### ERROR"; break; fi
  if [ "${N:-0}" -ge 13 ] 2>/dev/null; then echo "### DECIDING EPOCHS REACHED (ep12)"; break; fi
  sleep 110
done
