#!/bin/bash
set -u
K=dreaddevelopment/knee-mri-coatnet-plus-convnext
C=rsna-knee-abnormality-detection
MSG="The same pair of models as our best entry, with the second of them now finished training. That entry used a halfway checkpoint which read 0.9257 on the fifty-eight studies carrying real annotator labels; the completed run reads 0.9270 and was still improving when it ended. Nothing else changed: same two architectures, same first model, same resolution, same forty-eight slice positions per study, same series."
for i in $(seq 1 160); do
  S=$(kaggle kernels status "$K" 2>&1 | head -2 | tr '\n' ' ')
  echo "$(date -u +%H:%M:%S) $S"
  case "$S" in
    *complete*|*COMPLETE*) break;;
    *error*|*ERROR*|*cancel*|*CANCEL*) echo "KERNEL FAILED"; exit 2;;
  esac
  sleep 45
done
echo "$(date -u +%H:%M:%S) submitting"
kaggle competitions submit -c "$C" -k "$K" -f submission.csv -v 2 -m "$MSG" 2>&1 | tail -2
T0=$(date -u +%s)
for i in $(seq 1 200); do
  S=$(kaggle kernels status "$K" 2>&1 | head -1)
  L=$(kaggle competitions submissions "$C" --csv 2>/dev/null | sed -n 2p)
  el=$(( ($(date -u +%s) - T0) / 60 ))
  echo "$(date -u +%H:%M:%S) +${el}min | $S"
  case "$L" in
    *COMPLETE*) echo "SCORE: $(echo "$L" | awk -F, '{print $(NF-1)}') vs champion 0.945 | wall ${el} min"; break;;
    *ERROR*) echo "SUBMISSION ERROR"; break;;
  esac
  sleep 60
done
