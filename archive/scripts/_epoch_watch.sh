#!/bin/bash
# Exit the moment a NEW epoch line appears, so each epoch surfaces as it lands.
# Usage: _epoch_watch.sh <count_already_seen>
set -u
SEEN="${1:-0}"
H=192.220.55.116; P=20566
SSH="ssh -i /c/Users/johna/.ssh/id_ed25519 -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 root@$H"
for i in $(seq 1 200); do
  s=$($SSH 'cd /workspace/knee 2>/dev/null || exit 9
    grep -h "^ep" run_F*.log 2>/dev/null | wc -l
    grep -l "OutOfMemoryError\|Traceback" run_F*.log 2>/dev/null | tr "\n" " "' 2>/dev/null)
  n=$(echo "$s" | sed -n 1p); err=$(echo "$s" | sed -n 2p)
  if [ -n "${err// /}" ]; then
    echo "!! CRASH in $err"; $SSH 'tail -12 /workspace/knee/run_F*.log' 2>/dev/null | tail -16; exit 2
  fi
  if [ "${n:-0}" -gt "$SEEN" ]; then
    echo "=== $(date -u +%H:%M:%S) epoch $n of 12 ==="
    $SSH 'cd /workspace/knee; cat ft_chain.log; echo "--- all epochs ---"; grep -H "^ep" run_F*.log' 2>/dev/null | grep -viE "welcome|have fun"
    exit 0
  fi
  sleep 60
done
echo "aged out"; exit 4
