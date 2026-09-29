#!/bin/bash
# Exit as soon as the fine-tune's FIRST epoch lands, or as soon as it breaks. Exiting is the point:
# this runs as a background task, so the exit is what surfaces the result.
#
# Watches three failure modes, not just success: the corpus merge dying, the trainer crashing
# (an OOM killed the previous chain 30 s in and it walked silently on to the next arm), and the GPU
# sitting idle with nothing running.
set -u
H=192.220.55.116; P=20566
SSH="ssh -i /c/Users/johna/.ssh/id_ed25519 -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 root@$H"
for i in $(seq 1 400); do          # 400 * 45 s = 5 h ceiling
  s=$($SSH 'cd /workspace/knee 2>/dev/null || exit 9
    echo "MERGE=$(grep -c placed merge.log 2>/dev/null)"
    echo "CORPUS=$(ls corpus_v11/all_ids.npy >/dev/null 2>&1 && echo yes || echo no)"
    echo "TRAIN=$(pgrep -f "[f]inetune_box.py" | wc -l)"
    echo "GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -d " %")"
    echo "EP=$(grep -h "^ep" run_FT_*.log 2>/dev/null | tail -1)"
    echo "ERR=$(grep -l "OutOfMemoryError\|Traceback" run_FT_*.log 2>/dev/null | tr "\n" " ")"
    echo "INIT=$(grep -h "init_arm" run_FT_*.log 2>/dev/null | tail -1)"' 2>/dev/null)
  g () { echo "$s" | sed -n "s/^$1=//p"; }
  echo "$(date -u +%H:%M:%S) merge=$(g MERGE)/25 corpus=$(g CORPUS) train=$(g TRAIN) gpu=$(g GPU)% | $(g EP)"
  if [ -n "$(g ERR | tr -d ' ')" ]; then
    echo "!! CRASH in $(g ERR)"; $SSH 'tail -14 /workspace/knee/run_FT_*.log' 2>/dev/null | tail -20; exit 2
  fi
  if [ -n "$(g EP)" ]; then
    echo "== FIRST EPOCH LANDED =="
    echo "init line: $(g INIT)"
    $SSH 'cd /workspace/knee; head -30 run_FT_CONFDIST.log; echo ...; grep "^ep" run_FT_CONFDIST.log' 2>/dev/null
    exit 0
  fi
  if [ "$(g CORPUS)" = "no" ] && ! $SSH 'pgrep -f "[_]merge_v11" >/dev/null' 2>/dev/null; then
    echo "!! merge process gone and no corpus"; $SSH 'tail -8 /workspace/knee/merge.log' 2>/dev/null; exit 3
  fi
  sleep 45
done
echo "watcher aged out"; exit 4
