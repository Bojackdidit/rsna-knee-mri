#!/bin/bash
set -u
K=dreaddevelopment/knee-mri-coatnet-plus-efficientnetv2-m
C=rsna-knee-abnormality-detection
MSG="Our best model paired with one from a third family of designs, an EfficientNet, rather than the ConvNeXt we currently ship beside it. On the fifty-eight studies carrying real annotator labels this one reads 0.9175, below the 0.9257 of the model it replaces, but it also disagrees with our best model in different places, and disagreement is what makes a pair worth more than either half. Same resolution, same forty-eight slice positions per study, same series."
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
kaggle competitions submit -c "$C" -k "$K" -f submission.csv -v 1 -m "$MSG" 2>&1 | tail -2
T0=$(date -u +%s)
for i in $(seq 1 200); do
  L=$(kaggle competitions submissions "$C" --csv 2>/dev/null | sed -n 2p)
  el=$(( ($(date -u +%s) - T0) / 60 ))
  echo "$(date -u +%H:%M:%S) +${el}min"
  case "$L" in
    *COMPLETE*) echo "SCORE: $(echo "$L" | awk -F, '{print $(NF-1)}') vs champion 0.945 | wall ${el} min"; break;;
    *ERROR*) echo "SUBMISSION ERROR"; break;;
  esac
  sleep 60
done
