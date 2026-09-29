#!/bin/bash
set -u
K=dreaddevelopment/knee-mri-coatnet-384-scratch-aug-k48
C=rsna-knee-abnormality-detection
MSG="Our best entry with one of its two models replaced by one trained from nothing rather than continued from an existing model. Same architecture and the same number of slice positions per study as our best entry, trained across every available study with image augmentation and a running average of its own weights, neither of which our shipped models ever had. The second model and the way the two are combined are unchanged."
for i in $(seq 1 120); do
  S=$(kaggle kernels status "$K" 2>&1 | head -2 | tr '\n' ' ')
  echo "$(date -u +%H:%M:%S) $S"
  case "$S" in
    *complete*|*COMPLETE*) break;;
    *error*|*ERROR*|*cancel*|*CANCEL*) echo "KERNEL FAILED"; exit 2;;
  esac
  sleep 60
done
kaggle competitions submit -c "$C" -k "$K" -f submission.csv -v 1 -m "$MSG" 2>&1 | tail -2
for i in $(seq 1 60); do
  L=$(kaggle competitions submissions "$C" --csv 2>/dev/null | sed -n 2p)
  case "$L" in
    *COMPLETE*) echo "SCORE: $(echo "$L" | awk -F, '{print $(NF-1)}') vs champion 0.941"; break;;
    *ERROR*) echo "SUBMISSION ERROR"; break;;
  esac
  sleep 30
done
