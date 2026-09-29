#!/bin/bash
# Watch for the two things worth waking up for: Swin passing the champion's 0.9315, and the
# pseudo-label arm producing enough epochs to judge. Reports the whole picture each time it fires so
# a single read is enough, and re-checks liveness by hand each round rather than assuming.
set -u
K="-i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=12"
best_swin=0.9296
for i in $(seq 1 200); do
  S=$(timeout 25 ssh $K -p 29453 root@151.237.25.16 "grep -oE 'best [0-9.]+' /workspace/knee/run_SWIN_E16.log | tail -1 | cut -d' ' -f2" 2>/dev/null | tr -d '\r')
  P=$(timeout 25 ssh $K -p 58677 root@38.49.42.46 "grep -c GOLD /workspace/knee/run_PSEUDO50.log 2>/dev/null; grep -oE 'best [0-9.]+' /workspace/knee/run_PSEUDO50.log | tail -1 | cut -d' ' -f2" 2>/dev/null | tr '\n' ' ' | tr -d '\r')
  echo "$(date -u +%H:%M:%S) swin_best=$S  pseudo(epochs best)=$P"
  if [ -n "$S" ] && awk "BEGIN{exit !($S > 0.9315)}" 2>/dev/null; then
    echo "*** SWIN HAS PASSED THE CHAMPION: $S vs 0.9315 ***"
    break
  fi
  sleep 420
done
