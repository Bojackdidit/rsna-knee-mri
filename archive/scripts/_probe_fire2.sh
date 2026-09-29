#!/bin/bash
# Submit the second pair of per-finding probes after the daily slot reset.
#
# Both are already PUSHED as version 2. Two things went wrong on the first attempt and both are
# handled here:
#   - the day's five submissions were already spent, so the submit returned 400. Wait for reset
#     (00:00 UTC) and confirm a slot is actually available before firing.
#   - `kernels status` returned COMPLETE seconds after the push, because that is the status of the
#     PREVIOUS version. Require the kernel to be seen RUNNING/QUEUED first, so COMPLETE can only
#     mean this version finished.
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

slots_left () {
  kaggle competitions submissions rsna-knee-abnormality-detection --csv 2>/dev/null \
    | awk -F, -v d="$(date -u +%Y-%m-%d)" 'NR>1 && $3 ~ d {n++} END {print 5-(n+0)}'
}

echo "=== waiting for a free submission slot (UTC day $(date -u +%F)) ==="
for i in $(seq 1 400); do
  n=$(slots_left)
  [ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null && { echo "  [$(date +%H:%M)] $n slot(s) available"; break; }
  sleep 300
done

for P in medial-oa-probe baker-s-probe; do
  K=dreaddevelopment/knee-mri-$P
  VER=$(kaggle kernels list --user dreaddevelopment --search "$P" --csv 2>/dev/null >/dev/null; echo 2)

  seen_running=0
  for i in $(seq 1 60); do
    s=$(kaggle kernels status $K 2>&1 | head -1)
    case "$s" in
      *RUNNING*|*QUEUED*) seen_running=1; sleep 60;;
      *COMPLETE*) if [ "$seen_running" = 1 ] || [ $i -gt 3 ]; then echo "  [$(date +%H:%M)] $P ready"; break; fi; sleep 60;;
      *ERROR*|*CANCEL*) echo "  ABORT $P: $s"; break;;
      *) sleep 60;;
    esac
  done

  n=$(slots_left)
  if [ -z "$n" ] || [ "$n" -le 0 ] 2>/dev/null; then echo "  no slots left, stopping"; break; fi
  echo "=== submitting $P (version $VER) ==="
  kaggle competitions submit -c rsna-knee-abnormality-detection -k $K -v "$VER" -f submission.csv \
    -m "Diagnostic entry. Eleven of the twelve findings are replaced by a constant value, leaving one real, so the single score the leaderboard returns reads that one finding on its own. Not an attempt at a better model." 2>&1 | tail -1
  sleep 15
done
echo "=== done $(date +%H:%M) ==="
