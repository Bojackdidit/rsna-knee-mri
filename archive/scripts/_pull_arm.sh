#!/bin/bash
# Pull a finished arm off the box with hashes verified BOTH ends. Usage: _pull_arm.sh TAG [TAG...]
# Exists because an arm left on a rented box is an arm you cannot ship, and we lost the best upgrade
# of a previous session exactly that way. A size check is not enough - a truncated transfer can match
# on size; only the hash proves it.
set -u
cd /d/claude2/main_track/knee || exit 1
read IP PORT ID < /tmp/specbox
mkdir -p arms_spec
for TAG in "$@"; do
  for F in "raptor_ft_${TAG}.pt" "valids_${TAG}.csv" "raptor_gold_${TAG}.npz" "raptor_ft_${TAG}.json"; do
    R=$(timeout 120 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -p "$PORT" root@"$IP" \
        "cd /workspace/knee && sha256sum '$F' 2>/dev/null | cut -d' ' -f1" 2>/dev/null | tr -d '\r')
    [ -z "$R" ] && { echo "  -- $F: not on the box (skipped)"; continue; }
    scp -o StrictHostKeyChecking=no -P "$PORT" -q "root@$IP:/workspace/knee/$F" arms_spec/ 2>/dev/null
    L=$(sha256sum "arms_spec/$F" 2>/dev/null | cut -d' ' -f1)
    if [ "$R" = "$L" ]; then
      echo "  OK   $F  $(stat -c%s "arms_spec/$F" 2>/dev/null) bytes  ${R:0:16}"
    else
      echo "  FAIL $F  remote ${R:0:16} != local ${L:0:16}  -- RE-PULL BEFORE TEARDOWN"
    fi
  done
done
