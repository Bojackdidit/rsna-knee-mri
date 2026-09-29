#!/bin/bash
# Kill every merge, PROVE zero, then start exactly one. The gate is INSIDE this script so a
# chained ssh command cannot skip it -- I launched over a running merge three times today by
# putting the check in one command and the launch in the next.
#
# Bracket patterns ([_]merge) so pkill/pgrep never match this script's own command line.
set -u
cd /workspace/knee

echo "=== before ==="
pgrep -af "[_]merge_v11" || echo "  none"

pkill -9 -f "[_]merge_v11" 2>/dev/null
pkill -9 -f "[s]tartmerge.sh" 2>/dev/null
pkill -9 -f "[_]merge_retry" 2>/dev/null
sleep 5

N=$(pgrep -cf "[_]merge_v11" 2>/dev/null || echo 0)
echo "=== after kill: $N merge processes ==="
if [ "$N" -ne 0 ]; then
  echo "ABORT: $N still alive, refusing to start another"
  pgrep -af "[_]merge_v11"
  exit 1
fi

echo "=== clearing corrupt staging + empty output ==="
rm -rf corpus_v11/_stage corpus_v11/all_vols.npy corpus_v11/all_masks.npy corpus_v11/all_ids.npy
mkdir -p corpus_v11
df -BG --output=avail / | tail -1

echo "=== starting ONE merge ==="
setsid nohup bash startmerge.sh > merge_final.log 2>&1 < /dev/null &
disown
sleep 25
M=$(pgrep -cf "[_]merge_v11" 2>/dev/null || echo 0)
echo "=== now running: $M (expect exactly 1) ==="
tail -5 merge_final.log
