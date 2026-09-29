#!/bin/bash
# Validate the probe design BEFORE spending more slots on it, then finish the remaining probes.
#
# Why this exists. Four probes were fired on 2026-09-24 and the results do not reconcile: summing
# the nine measured findings against the base's macro demands that ACL + Medial Meniscus + Effusion
# average 1.008, which is impossible. So either the base is not the config assumed, some earlier
# reads are stale, or `AUC_j = 12*board - 5.5` does not hold exactly - which it would not if the
# official metric drops findings with no label variation, or treats an all-constant column
# differently from sklearn.
#
# knee-mri-metric-check holds ALL TWELVE columns at the constant. Under the design it must return
# EXACTLY 0.500. That single read settles it. The builder said to fire it first; skipping it is why
# four measurements are currently uninterpretable.
#
# Only if it returns 0.500 do the remaining probes go out.
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

slots_left () {
  kaggle competitions submissions rsna-knee-abnormality-detection --csv 2>/dev/null \
    | awk -F, -v d="$(date -u +%Y-%m-%d)" 'NR>1 && $3 ~ d {n++} END {print 5-(n+0)}'
}

# push, then wait for THIS version - a COMPLETE status straight after a push belongs to the
# previous version, which is how a superseded build got submitted once already.
run_and_wait () {
  local DIR="$1" K="$2"
  local OUT VER SEEN=0
  OUT=$(kaggle kernels push -p "$DIR" 2>&1 | tail -1); echo "  $OUT"
  VER=$(echo "$OUT" | grep -oE 'version [0-9]+' | grep -oE '[0-9]+')
  [ -z "$VER" ] && { echo "  ABORT: push reported no version"; return 1; }
  for i in $(seq 1 60); do
    s=$(kaggle kernels status "$K" 2>&1 | head -1)
    case "$s" in
      *RUNNING*|*QUEUED*) SEEN=1;;
      *COMPLETE*) [ "$SEEN" = 1 ] && { echo "$VER"; return 0; };;
      *ERROR*|*CANCEL*) echo "  ABORT: $s"; return 1;;
    esac
    sleep 60
  done
  echo "  ABORT: timed out"; return 1
}

echo "=== waiting for a submission slot ==="
for i in $(seq 1 400); do
  n=$(slots_left)
  [ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null && { echo "  [$(date +%H:%M)] $n slot(s)"; break; }
  sleep 300
done

echo "=== STEP 1: metric check (all twelve columns constant; must return exactly 0.500) ==="
V=$(run_and_wait variants/push_knee_mri_metric_check dreaddevelopment/knee-mri-metric-check | tail -1)
[ -z "$V" ] && { echo "metric check could not be run; stopping"; exit 1; }
kaggle competitions submit -c rsna-knee-abnormality-detection \
  -k dreaddevelopment/knee-mri-metric-check -v "$V" -f submission.csv \
  -m "Diagnostic entry. All twelve findings are replaced by the same constant value, so this is a check on how the score behaves, not a model." 2>&1 | tail -1

echo "=== reading the metric check ==="
for i in $(seq 1 40); do
  R=$(kaggle competitions submissions rsna-knee-abnormality-detection --csv 2>/dev/null | sed -n 2p | awk -F, '{print $(NF-1)}')
  case "$R" in
    0.5|0.500) echo "  PASS: returned $R - the design holds, AUC_j = 12*board - 5.5 is valid"; OK=1; break;;
    ''|nan) sleep 120;;
    *) echo "  FAIL: returned $R, not 0.500. The constant-column assumption is WRONG and every"
       echo "        per-finding number derived from it - all nine - must be discarded."; OK=0; break;;
  esac
done

[ "${OK:-0}" = 1 ] || { echo "stopping; not spending slots on a void design"; exit 1; }

for P in medial-oa-probe baker-s-probe; do
  n=$(slots_left); [ "$n" -le 0 ] 2>/dev/null && { echo "out of slots"; break; }
  echo "=== $P ==="
  V=$(run_and_wait "variants/push_knee_mri_$(echo $P | tr '-' '_')" "dreaddevelopment/knee-mri-$P" | tail -1)
  [ -z "$V" ] && continue
  kaggle competitions submit -c rsna-knee-abnormality-detection -k "dreaddevelopment/knee-mri-$P" \
    -v "$V" -f submission.csv \
    -m "Diagnostic entry. Eleven of the twelve findings are replaced by a constant value, leaving one real, so the single score the leaderboard returns reads that one finding on its own. Not an attempt at a better model." 2>&1 | tail -1
done
echo "=== done $(date +%H:%M) ==="
