#!/bin/bash
# Arm 2 alone. Start the runtime poller BEFORE the submit call, or the clock is lost for good.
set -u
K=dreaddevelopment/knee-mri-coatnet-384-single-series-windows
C=rsna-knee-abnormality-detection
MSG="The same single model as our best entry, but reading cleaner pictures. Each knee is stored as five separate scan sequences laid end to end, and we show the model three neighbouring slices at a time. Until now a few of those triples were built across the join between two different sequences, so about one window in ten was a picture of two unrelated things. This entry only ever takes three slices from within the same sequence, which is exactly how the model was trained. Nothing else changed: same model weights, same architecture, same resolution, same number of windows per study."
for i in $(seq 1 120); do
  S=$(kaggle kernels status "$K" 2>&1 | head -2 | tr '\n' ' ')
  echo "$(date -u +%H:%M:%S) $S"
  case "$S" in
    *complete*|*COMPLETE*) break;;
    *error*|*ERROR*|*cancel*|*CANCEL*) echo "KERNEL FAILED"; exit 2;;
  esac
  sleep 45
done
echo "$(date -u +%H:%M:%S) submitting"
kaggle competitions submit -c "$C" -k "$K" -f submission.csv -v 1 -m "$MSG" 2>&1 | tail -2
T0=$(date -u +%s)
for i in $(seq 1 200); do
  S=$(kaggle kernels status "$K" 2>&1 | head -1)
  L=$(kaggle competitions submissions "$C" --csv 2>/dev/null | sed -n 2p)
  el=$(( ($(date -u +%s) - T0) / 60 ))
  echo "$(date -u +%H:%M:%S) +${el}min | $S"
  case "$L" in
    *COMPLETE*) echo "SCORE: $(echo "$L" | awk -F, '{print $(NF-1)}') vs arm2-solo control 0.944 | wall ${el} min"; break;;
    *ERROR*) echo "SUBMISSION ERROR"; break;;
  esac
  sleep 60
done
