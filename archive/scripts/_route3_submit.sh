#!/bin/bash
# Wait for the route3 kernel to finish, submit it, then poll the score to completion.
# Runtime is only observable while a submission is in flight, so the poll starts with the submit.
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
K=dreaddevelopment/knee-mri-coatnet-plus-convnext-route3
for i in $(seq 1 60); do
  s=$(kaggle kernels status $K 2>&1 | head -1)
  echo "[$(date +%H:%M:%S)] $s"
  case "$s" in
    *COMPLETE*) break;;
    *ERROR*|*CANCEL*) echo "ABORT - kernel did not complete"; exit 1;;
  esac
  sleep 90
done
echo "=== submitting ==="
kaggle competitions submit -c rsna-knee-abnormality-detection -k $K -v 1 -f submission.csv \
  -m "The same two models as our best entry. Three of the twelve findings are now read from whichever single model handles them better, instead of averaging both: MCL and Medial OA from the first, Patellofemoral OA from the second. The choice was made on one half of the fifty-eight studies with real annotator labels and checked on the other half, and only these three survived that check; the other nine still use the average." 2>&1 | tail -2
echo "=== polling score ==="
/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe -u _poll_sub.py $(kaggle competitions submissions rsna-knee-abnormality-detection --csv 2>/dev/null | sed -n 2p | cut -d, -f1)
