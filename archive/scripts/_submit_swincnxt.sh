#!/bin/bash
set -u
K=dreaddevelopment/knee-mri-convnext-plus-swin
C=rsna-knee-abnormality-detection
MSG="Our two strongest models paired, with the CoAtNet that has anchored every entry so far taken out. The transformer replacing it is the best single model we have trained: on the fifty-eight studies carrying real annotator labels it reads 0.9317 against 0.9315 for the one it replaces, and it disagrees with the remaining model in different places than the old one did. Offline the two pairs are level, so this is the board deciding between them. Same resolution, same forty-eight slice positions per study, same series, same two-model runtime."
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
    *COMPLETE*) echo "SWIN+CNXT SCORE: $(echo "$L" | awk -F, '{print $(NF-1)}')  vs champion 0.945 | wall ${el} min"; break;;
    *ERROR*) echo "SUBMISSION ERROR"; break;;
  esac
  echo "$(date -u +%H:%M:%S) +${el}min"
  sleep 60
done
