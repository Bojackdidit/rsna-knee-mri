#!/bin/bash
# Wake me when SCRATCH_AUG finishes, and confirm the masked run took over the GPU.
# Also exits early on a crash or an idle GPU, so an overnight failure cannot go unnoticed.
set -u
H=192.220.55.116; P=20566
SSH="ssh -i /c/Users/johna/.ssh/id_ed25519 -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 root@$H"
for i in $(seq 1 500); do
  s=$($SSH 'cd /workspace/knee 2>/dev/null || exit 9
    echo "DONE=$(grep -c "SCRATCH_AUG done" ft_chain.log 2>/dev/null)"
    echo "TRAIN=$(pgrep -f "[f]inetune_box.py" | wc -l)"
    echo "GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -d " %")"
    echo "ERR=$(grep -l "OutOfMemoryError\|Traceback" run_SCRATCH_AUG.log run_MASKED.log 2>/dev/null | tr "\n" " ")"
    echo "EP=$(grep -h "^ep" run_SCRATCH_AUG.log 2>/dev/null | tail -1)"' 2>/dev/null)
  g () { echo "$s" | sed -n "s/^$1=//p"; }
  echo "$(date -u +%H:%M:%S) done=$(g DONE) trainers=$(g TRAIN) gpu=$(g GPU)% | $(g EP)"
  if [ -n "$(g ERR | tr -d ' ')" ]; then
    echo "!! CRASH in $(g ERR)"; $SSH 'tail -14 /workspace/knee/run_SCRATCH_AUG.log /workspace/knee/run_MASKED.log 2>/dev/null' 2>/dev/null | tail -24; exit 2
  fi
  if [ "$(g DONE)" != "0" ]; then
    echo "== SCRATCH_AUG COMPLETE =="
    $SSH 'cd /workspace/knee; grep -h "^ep" run_SCRATCH_AUG.log | tail -6; echo "---"; python -c "
import json; d=json.load(open(\"raptor_ft_SCRATCH_AUG.json\")); print(\"best %.4f over %d epochs, n_train %s\"%(d[\"best_gold_auc\"],len(d[\"hist\"]),d[\"n_train\"]))
a=d.get(\"aucs\",{});
print(\"weakest:\", \", \".join(\"%s %.3f\"%(k,v) for k,v in sorted(a.items(), key=lambda kv: kv[1])[:4]))" 2>/dev/null
      echo "--- masked run picked up? ---"; sleep 60; echo "trainers now $(pgrep -f "[f]inetune_box.py" | wc -l)"; tail -2 ft_chain.log' 2>/dev/null | grep -viE "welcome|have fun"
    exit 0
  fi
  if [ "$(g TRAIN)" = "0" ]; then echo "!! NO TRAINER RUNNING and SCRATCH_AUG not done"; exit 3; fi
  sleep 300
done
echo "aged out"; exit 4
