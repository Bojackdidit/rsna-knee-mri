#!/bin/bash
# Arm 2 alone. Start the runtime poller BEFORE the submit call, or the clock is lost for good.
set -u
K=dreaddevelopment/knee-mri-coatnet-384-second-arm-alone
C=rsna-knee-abnormality-detection
MSG="Our best entry with one of its two models removed, leaving only the stronger of the pair. That model was trained from nothing with image augmentation and a running average of its own weights, and on the fifty-eight studies carrying real annotator labels it reads 0.9315 against 0.9260 for the one taken away. Nothing else changed: same forty-eight slice positions per study, same architecture, same resolution, same series. With one model instead of two it should also finish in roughly half the time."
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
