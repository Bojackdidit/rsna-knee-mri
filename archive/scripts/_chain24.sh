#!/bin/bash
# Wait for this box's current run to exit, then start the 24-epoch sibling. Usage: _chain24.sh BBLR TAG
# Never pkill/pgrep -f: this script's own command line contains the pattern and would self-match.
set -u
BBLR="$1"; TAG="$2"
while true; do
  N=$(ps -eo args | grep -ac "[f]inetune_box.py")
  [ "$N" -eq 0 ] && break
  sleep 60
done
echo "16ep run exited $(date -u +%H:%M:%S) - starting 24ep $TAG" >> /workspace/chain24.log
cd /workspace/knee
bash /workspace/knee/_run_lr24.sh "$BBLR" "$TAG" > /workspace/knee/run_$TAG.log 2>&1
