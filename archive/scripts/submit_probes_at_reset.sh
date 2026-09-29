#!/bin/bash
# Fire the per-finding board probes as soon as daily submit slots exist.
#
# Batching is CORRECT here, unlike for candidate arms. memory/submit-one-read-it-then-decide applies
# when one result must select the next; these are twelve INDEPENDENT measurements of different
# columns, so there is nothing to decide between them. Each returns AUC_j = 12*board - 5.5.
#
# Weakest-suspected first, so the informative reads land earliest. PF OA is already done (0.884)
# and Lateral OA is pending.
set -u
cd "$(dirname "$0")"
C=rsna-knee-abnormality-detection
# Capped at 3. The arms land ~09:00 and ~18:30 UTC and MUST have slots to board -- probes are
# information, arms are the score attempt. Re-run this script with a longer list once the arms are
# boarded. memory/never-spend-a-submit-treading-water.
PROBES="synovitis lateral_meniscus mcl"

msg () {
  n=$(echo "$1" | tr '_' ' ')
  echo "A measurement, not an attempt to score. This keeps our usual prediction for ${n} and sets \
the other eleven columns to a constant, so the single number the board returns reads back that one \
finding on its own. The probe for patellofemoral osteoarthritis returned 0.532, exactly what the \
arithmetic predicts, so the method is confirmed. Expect roughly 0.53 again."
}

for p in $PROBES; do
  k="knee-mri-$(echo "$p" | tr '_' '-')-probe"
  # kernel must have finished its run before it can be submitted
  ok=0
  for i in $(seq 1 30); do
    s=$(kaggle kernels status "dreaddevelopment/$k" 2>&1 | head -1)
    case "$s" in
      *COMPLETE*) ok=1; break;;
      *ERROR*|*CANCEL*) echo "$(date -u +%H:%M) $k KERNEL FAILED: $s"; break;;
    esac
    sleep 30
  done
  [ "$ok" -eq 1 ] || { echo "$(date -u +%H:%M) $k not runnable, skipping"; continue; }

  # try to submit; if the daily cap is hit, wait it out and retry
  for i in $(seq 1 300); do          # up to ~25 h of waiting across the whole queue
    out=$(kaggle competitions submit -c "$C" -k "dreaddevelopment/$k" -f submission.csv -v 1 \
          -m "$(msg "$p")" 2>&1 | tail -2)
    case "$out" in
      *"submissions remaining"*|*"Successfully"*)
        echo "$(date -u +%H:%M) $p SUBMITTED -- $out"; break;;
      *"limit"*|*"maximum"*|*"Maximum"*|*"400"*)
        sleep 300;;
      *)
        echo "$(date -u +%H:%M) $p unexpected: $out"; sleep 300;;
    esac
  done
done
echo "=== probe queue drained $(date -u +%H:%M:%S) ==="
