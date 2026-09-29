#!/bin/bash
# Ship the single-arm read: the 0.943 blend with the OLD confdist_swa arm removed entirely.
# Two questions in one slot: does the old arm still contribute anything, and does a single model
# at half the runtime hold the score (the efficiency board is a separate ranking we sit 1729th on).
set -u
K=dreaddevelopment/knee-mri-coatnet-384-scratch-aug-solo
C=rsna-knee-abnormality-detection
MSG="Our best entry with the older of its two models removed, so only the model trained from nothing with image augmentation is left. Nothing else changed: same slice positions per study, same architecture, same resolution. It should also finish in about half the time, since only one model now reads each study."
T0=$(date -u +%s)
for i in $(seq 1 120); do
  S=$(kaggle kernels status "$K" 2>&1 | head -2 | tr '\n' ' ')
  echo "$(date -u +%H:%M:%S) $S"
  case "$S" in
    *complete*|*COMPLETE*) echo "kernel runtime ~$(( ($(date -u +%s)-T0)/60 )) min of watching"; break;;
    *error*|*ERROR*|*cancel*|*CANCEL*) echo "KERNEL FAILED"; kaggle kernels output "$K" -p /tmp/solo_out 2>&1 | tail -3; exit 2;;
  esac
  sleep 60
done
kaggle competitions submit -c "$C" -k "$K" -f submission.csv -v 1 -m "$MSG" 2>&1 | tail -2
for i in $(seq 1 60); do
  L=$(kaggle competitions submissions "$C" --csv 2>/dev/null | sed -n 2p)
  case "$L" in
    *COMPLETE*) echo "SOLO SCORE: $(echo "$L" | awk -F, '{print $(NF-1)}') vs champion 0.943"; break;;
    *ERROR*) echo "SUBMISSION ERROR"; break;;
  esac
  sleep 30
done
