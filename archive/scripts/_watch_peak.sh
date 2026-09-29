#!/bin/bash
# Watch arm 2 for a NEW BEST before the 00:00 UTC reset. Both arms peaked at epoch 14 and rolled over,
# so the question is only whether this one is different. Fires nothing; it reports so the decision to
# spend the last submission of the day is made on a number rather than a hope.
set -u
SSH="ssh -i /c/Users/johna/.ssh/id_ed25519 -p 20566 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 root@192.220.55.116"
BEST=0.9315
for i in $(seq 1 40); do
  L=$($SSH 'cd /workspace/knee; grep -oE "^ep[0-9]+ .*GOLD macro-AUC [0-9.]+ \(best [0-9.]+\)" run_SCRATCH_AUG2.log | tail -1' 2>/dev/null)
  B=$(echo "$L" | grep -oE "best [0-9.]+" | grep -oE "[0-9.]+")
  now=$(date -u +%H:%M:%S); mins=$(( ( $(date -u -d "tomorrow 00:00" +%s) - $(date -u +%s) ) / 60 ))
  echo "$now  ${mins}min to reset | $L"
  if [ -n "${B:-}" ] && awk "BEGIN{exit !($B > $BEST)}"; then
    echo "*** NEW BEST $B (was $BEST) — turnaround needs ~35 min, ${mins} min left ***"
    exit 0
  fi
  [ "$mins" -lt 36 ] && { echo "inside the turnaround window with no new best — hold the slot"; exit 1; }
  sleep 420
done
