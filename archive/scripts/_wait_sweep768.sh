#!/bin/bash
set -u
S="ssh -i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=12 -p 14354 root@ssh9.vast.ai"
for i in $(seq 1 45); do
  O=$(timeout 30 $S 'pgrep -f "[_]extract_sweep" >/dev/null && echo RUNNING || { grep -E "^(baseline|h0|hyst)" /workspace/solar/sweep_t768.log | sort -k2 -rn | head -6; echo SWEEPDONE; }' 2>/dev/null | grep -vE "Welcome|Have fun|Warning")
  if echo "$O" | grep -q SWEEPDONE; then echo "$O"; break; fi
  sleep 40
done
