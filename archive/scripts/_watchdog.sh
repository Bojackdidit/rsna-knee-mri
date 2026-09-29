#!/bin/bash
# Exit (and therefore notify) the moment the v11 chain stops making progress.
#
# Arm A died 30 s into the first launch with a CUDA OOM and the chain walked straight on to arm B,
# which was seconds from the identical crash. Nothing would have told me until I next looked. This
# watches for: no trainer running while the chain is unfinished, a fresh Traceback/OOM in any run
# log, or the whole chain completing. It does NOT restart anything by itself - a silent auto-restart
# into the same wall is how a night gets burned twice.
set -u
cd "$(dirname "$0")"
SSH="ssh -i /c/Users/johna/.ssh/id_ed25519 -p 53473 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 root@154.64.230.50"
for i in $(seq 1 700); do            # 700 * 300 s ~= 58 h
  s=$($SSH 'cd /workspace/knee
    echo "T=$(pgrep -f "[f]inetune_box.py" | wc -l)"
    echo "G=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -d " %")"
    echo "C=$(grep -c "done" chain_v11.log 2>/dev/null)"
    echo "DONE=$(grep -c "CHAIN COMPLETE" chain_v11.log 2>/dev/null)"
    echo "ERR=$(grep -l "OutOfMemoryError\|Traceback" run_V11_AUG_*.log 2>/dev/null | tr "\n" " ")"
    echo "EP=$(grep -h "^ep" run_V11_AUG_*.log 2>/dev/null | tail -1)"' 2>/dev/null)
  T=$(echo "$s" | sed -n 's/^T=//p'); G=$(echo "$s" | sed -n 's/^G=//p')
  DONE=$(echo "$s" | sed -n 's/^DONE=//p'); ERR=$(echo "$s" | sed -n 's/^ERR=//p')
  EP=$(echo "$s" | sed -n 's/^EP=//p')
  echo "$(date -u +%H:%M:%S) trainers=${T:-?} gpu=${G:-?}% | $EP"
  if [ -n "${ERR// /}" ]; then echo "!! CRASH in: $ERR"; $SSH 'tail -6 /workspace/knee/run_V11_AUG_*.log' 2>/dev/null | tail -20; exit 2; fi
  if [ "${DONE:-0}" != "0" ]; then echo "== CHAIN COMPLETE =="; exit 0; fi
  if [ "${T:-0}" = "0" ]; then echo "!! NO TRAINER RUNNING and chain not complete"; exit 3; fi
  sleep 300
done
echo "watchdog aged out"
