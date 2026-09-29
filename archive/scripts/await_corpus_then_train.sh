#!/bin/bash
# Wait for the v11 merge to finish, then run safe_start_train.sh.
#
# Auto-chaining is safe here ONLY because the gate lives inside safe_start_train.sh: it re-verifies
# the corpus CONTENT (NONZERO fraction, not just shape) and refuses if a trainer is already running.
# Never auto-chain onto a step whose preconditions are checked somewhere else.
set -u
cd /workspace/knee
for i in $(seq 1 240); do            # 240 * 60s = 4 h ceiling
  if [ -f corpus_v11/all_masks.npy ] && [ -f corpus_v11/all_ids.npy ]; then
    echo "$(date -u +%H:%M:%S) corpus files present on poll $i -- running the training gate"
    bash safe_start_train.sh
    rc=$?
    echo "$(date -u +%H:%M:%S) safe_start_train.sh exit=$rc"
    exit $rc
  fi
  # if the merge has died and is not going to be retried, stop waiting
  if ! pgrep -f "[_]merge_v11" > /dev/null && ! pgrep -f "[s]tartmerge.sh" > /dev/null; then
    echo "$(date -u +%H:%M:%S) poll $i: no merge running and no corpus -- merge died, giving up"
    tail -5 merge_final.log
    exit 1
  fi
  sleep 60
done
echo "timed out after 4 h"
exit 1
