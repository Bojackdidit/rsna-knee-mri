#!/bin/bash
# Ship the pair of scratch-trained arms and MEASURE THE CLOCK THIS TIME.
# The runtime of a scoring run can only be read while it is running, so the poller starts before the
# submit call rather than after it.
set -u
K=dreaddevelopment/knee-mri-coatnet-384-two-scratch
C=rsna-knee-abnormality-detection
MSG="Both halves of our best entry are now models trained from nothing with image augmentation, rather than one of each. The older model, which we measured as contributing only a thousandth, is gone. The replacement is the same recipe as the one that took us to 0.943, differing only in its random seed, and on the fifty-eight studies carrying real annotator labels it reads 0.9315 against 0.9260 for the model it sits beside. It is still training and this is its best checkpoint so far. Same architecture, same forty-eight slice positions per study, same resolution, same series."
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
# poll the SCORING run so the runtime is captured rather than lost
for i in $(seq 1 200); do
  S=$(kaggle kernels status "$K" 2>&1 | head -1)
  L=$(kaggle competitions submissions "$C" --csv 2>/dev/null | sed -n 2p)
  el=$(( ($(date -u +%s) - T0) / 60 ))
  echo "$(date -u +%H:%M:%S) +${el}min | $S"
  case "$L" in
    *COMPLETE*) echo "SCORE: $(echo "$L" | awk -F, '{print $(NF-1)}') vs champion 0.943 | wall ${el} min"; break;;
    *ERROR*) echo "SUBMISSION ERROR"; break;;
  esac
  sleep 60
done
