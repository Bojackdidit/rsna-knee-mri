#!/bin/bash
# Blocks until the first real training epoch lands, or a failure condition trips.
H="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p 40101 root@84.8.116.228"
for i in $(seq 1 220); do
  R=$($H 'EP=$(grep -c "GOLD macro-AUC" /workspace/train_ext_full.log 2>/dev/null||echo 0);
          ER=$(grep -cE "Traceback|RuntimeError|out of memory|refusing to train" /workspace/pipeline.log /workspace/train_ext_full.log 2>/dev/null|awk -F: "{s+=\$2} END{print s+0}");
          ST=$(grep -E "^\[[0-9:]+\]" /workspace/pipeline.log 2>/dev/null|tail -1|cut -c12-52);
          TR=$(grep -oE "train [0-9]+ \| gold-val [0-9]+" /workspace/train_ext_full.log 2>/dev/null|tail -1);
          CLK=$(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader,nounits|tr "\n" " ");
          echo "$EP|$ER|$ST|$TR|$CLK"' 2>/dev/null)
  [ -z "$R" ] && { echo "$(date -u +%H:%M) BOX UNREACHABLE"; sleep 60; continue; }
  EP=$(echo "$R"|cut -d"|" -f1); ER=$(echo "$R"|cut -d"|" -f2)
  echo "$(date -u +%H:%M) ep=$EP err=$ER stage=$(echo "$R"|cut -d"|" -f3) size=$(echo "$R"|cut -d"|" -f4) gpu=$(echo "$R"|cut -d"|" -f5)"
  if [ "${ER:-0}" -gt 0 ] 2>/dev/null; then echo "### ERROR DETECTED"; break; fi
  if [ "${EP:-0}" -ge 1 ] 2>/dev/null; then echo "### FIRST EPOCH LANDED"; break; fi
  sleep 60
done
