#!/bin/bash
set -u
S="ssh -i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=12 -p 14354 root@ssh9.vast.ai"
for i in $(seq 1 40); do
  O=$(timeout 30 $S 'pgrep -f "[b]uild_sub768" >/dev/null && echo RUNNING || { grep -E "fallback|instances |silent|wrote" /workspace/solar/sub768b.log; echo BUILT; }' 2>/dev/null | grep -vE "Welcome|Have fun|Warning")
  if echo "$O" | grep -q BUILT; then echo "$O"; break; fi
  sleep 30
done
