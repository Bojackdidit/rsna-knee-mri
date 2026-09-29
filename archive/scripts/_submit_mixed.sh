#!/bin/bash
# Arm 2 alone. Start the runtime poller BEFORE the submit call, or the clock is lost for good.
set -u
K=dreaddevelopment/knee-mri-coatnet-plus-convnext
C=rsna-knee-abnormality-detection
MSG="Two models of genuinely different design reading each knee, rather than two of the same kind. Our entries have always paired two models built the same way, which turns out to add almost nothing because they agree with each other. This pairs our best model with one of a different architecture, trained from nothing on the same scans with the same settings, which reads 0.9257 on the fifty-eight studies carrying real annotator labels against 0.9315 for the model beside it. Same resolution, same slice positions per study, same series."
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
    *COMPLETE*) echo "SCORE: $(echo "$L" | awk -F, '{print $(NF-1)}') vs champion 0.944 | wall ${el} min"; break;;
    *ERROR*) echo "SUBMISSION ERROR"; break;;
  esac
  sleep 60
done
