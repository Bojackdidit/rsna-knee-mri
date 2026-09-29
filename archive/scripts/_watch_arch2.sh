#!/bin/bash
# Watch the two newly chained architecture arms through their first epochs. The point is to catch an
# OOM or a dead start early rather than discover it hours later.
set -u
for i in $(seq 1 90); do
  for spec in "1:20566:192.220.55.116:EFFV2M_E16" "2:58677:38.49.42.46:CNXTL_E16"; do
    IFS=: read n port ip tag <<< "$spec"
    OUT=$(timeout 25 ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=12 -p $port root@$ip "grep -E 'GOLD|Error|error|OutOfMemory|Traceback' /workspace/knee/run_${tag}.log 2>/dev/null | tail -3; pgrep -f '[f]inetune_box.py' >/dev/null && echo ALIVE || echo DEAD" 2>/dev/null | grep -vE "Welcome|Have fun|Warning: Perm")
    echo "$(date -u +%H:%M:%S) box$n $tag: $(echo "$OUT" | tr '\n' ' | ')"
  done
  sleep 300
done
