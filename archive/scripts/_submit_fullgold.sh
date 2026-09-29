#!/bin/bash
set -u
K=dreaddevelopment/knee-mri-coatnet-plus-fullgold
C=rsna-knee-abnormality-detection
MSG="Our best model paired with a new one trained on the fifty-eight scans that carry real radiologist labels. Every other scan in the dataset was labelled by an AI reading the report, and those fifty-eight are the only genuine ones; until now we held them back as a test set. Two runs on different held-out portions both came out about 0.010 better than the same recipe without them, which is why this is worth a slot. Same resolution, same forty-eight slice positions per study, same series, same two-model runtime."
for i in $(seq 1 200); do
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
for i in $(seq 1 240); do
  L=$(kaggle competitions submissions "$C" --csv 2>/dev/null | sed -n 2p)
  el=$(( ($(date -u +%s) - T0) / 60 ))
  case "$L" in
    *COMPLETE*) echo "FULLGOLD PAIR: $(echo "$L" | awk -F, '{print $(NF-1)}')  vs champion 0.945 | wall ${el} min"; break;;
    *ERROR*) echo "SUBMISSION ERROR"; break;;
  esac
  echo "$(date -u +%H:%M:%S) +${el}min"
  sleep 60
done
