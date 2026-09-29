#!/bin/bash
set -u
K=dreaddevelopment/knee-mri-coatnet-convnext-swin
C=rsna-knee-abnormality-detection
MSG="Three models of genuinely different design reading each knee instead of two. We added a transformer to the pair we already ship, and it is the strongest single model we have trained: on the fifty-eight studies carrying real annotator labels it reads 0.9317, just above the 0.9315 of our best. It agrees closely with that best model, so on its own it would be a poor partner, but the older model of the third design disagrees with both of them, and three models that each disagree with the other two combine better than any two of them do. Same resolution, same forty-eight slice positions per study, same series. With three models sharing two cards it will take roughly twice as long to run."
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
    *COMPLETE*) echo "TRIO SCORE: $(echo "$L" | awk -F, '{print $(NF-1)}')  vs champion 0.945 | wall ${el} min"; break;;
    *ERROR*) echo "SUBMISSION ERROR"; break;;
  esac
  echo "$(date -u +%H:%M:%S) +${el}min"
  sleep 60
done
