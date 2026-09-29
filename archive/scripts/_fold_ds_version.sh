#!/bin/bash
# As each remaining fold arm lands, add it to the knee-raptor-fold-arms dataset so the ship path
# is never waiting on an upload. The .json is written together with the .pt at the END of
# training, so waiting for both (plus a settle) guarantees the weight file is complete.
# Final authority on upload success is `kaggle datasets status` + the file list, checked before
# any kernel push - never this log alone.
set -u
cd "$(dirname "$0")"
for T in LR96_F2 LR96_F3 LR96_F4; do
  while [ ! -f "raptor_ft_${T}.json" ] || [ ! -f "raptor_ft_${T}.pt" ]; do sleep 180; done
  sleep 30
  cp "raptor_ft_${T}.pt" fold_arms_ds/
  for i in 1 2 3; do
    kaggle datasets version -p fold_arms_ds --dir-mode zip -m "add ${T}" 2>&1 | tail -1
    sleep 60
    if kaggle datasets files dreaddevelopment/knee-raptor-fold-arms --csv 2>/dev/null | grep -q "${T}"; then
      break
    fi
  done
  echo "versioned ${T} $(date +%H:%M)"
done
echo "all fold arms uploaded $(date +%H:%M)"
