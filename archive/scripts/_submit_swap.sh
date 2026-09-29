#!/bin/bash
# Wait for the champ+retrained kernel to finish its run, then submit it and read the score.
set -u
K=dreaddevelopment/knee-mri-coatnet-384-retrained-k48
C=rsna-knee-abnormality-detection
MSG="Our best entry with one of its two models replaced, and this time reading the same number of slice positions per study as our best entry does. The replacement is that model retrained from its own weights across every available study with image augmentation, which it never had. Our previous two attempts at this read only half as many slice positions, which made them impossible to interpret; this one changes the model weights and nothing else.""
for i in $(seq 1 120); do
  S=$(kaggle kernels status "$K" 2>&1 | head -2 | tr '\n' ' ')
  echo "$(date -u +%H:%M:%S) $S"
  case "$S" in
    *complete*|*COMPLETE*) echo "RUN COMPLETE"; break;;
    *error*|*ERROR*|*cancel*|*CANCEL*) echo "KERNEL FAILED"; exit 2;;
  esac
  sleep 60
done
case "$S" in *complete*|*COMPLETE*) ;; *) echo "never completed: $S"; exit 3;; esac
kaggle competitions submit -c "$C" -k "$K" -f submission.csv -v 1 -m "$MSG" 2>&1 | tail -3
for i in $(seq 1 60); do
  L=$(kaggle competitions submissions "$C" --csv 2>/dev/null | sed -n 2p)
  case "$L" in
    *COMPLETE*) echo "SCORE: $(echo "$L" | awk -F, '{print $(NF-1)}')  (champion 0.941)"; break;;
    *ERROR*) echo "SUBMISSION ERROR"; break;;
  esac
  sleep 30
done
