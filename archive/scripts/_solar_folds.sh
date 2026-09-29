#!/bin/bash
# Train the remaining four folds once fold 0 finishes, so the five can be averaged.
#
# Why an ensemble and not more encoder. Going convnext_small -> convnext_base and 24 -> 40 epochs
# moved PQ 0.4093 -> about 0.42, which is not the missing 0.2. The decomposition says the failure is
# recognition: 276 false instances against 440 truths. Averaging independent folds' probability maps
# is the standard way to kill false positives specifically - a spurious response that one fold invents
# is unlikely to survive the mean of five, while a real filament that all five see is untouched.
set -eux
cd /workspace/solar
while pgrep -f "[s]olar_tiled.py" > /dev/null; do sleep 60; done
for F in 1 2 3 4; do
  echo "folds: starting f$F $(date -u +%F' '%H:%M:%S)" >> folds.log
  python -u solar_tiled.py --enc tu-convnext_base --tile 512 --bs 6 --per_image 10 \
    --epochs 40 --fold "$F" --tag base40 > "run_base40_f$F.log" 2>&1
  echo "folds: f$F rc=$? $(date -u +%F' '%H:%M:%S)" >> folds.log
done
echo "folds: all done $(date -u +%F' '%H:%M:%S)" >> folds.log
