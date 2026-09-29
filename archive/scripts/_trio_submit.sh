#!/bin/bash
# Wait for the route3 kernel to finish, submit it, then poll the score to completion.
# Runtime is only observable while a submission is in flight, so the poll starts with the submit.
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
K=dreaddevelopment/knee-mri-three-family-trio
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
  -m "Three models of genuinely different design reading each knee instead of two: the CoAtNet and ConvNeXt we already ship, plus an EfficientNetV2. The trio we tried before was three CoAtNets, which are near-copies of each other; these three disagree with one another far more, and disagreement is what has made a pair worth more than either half. Same resolution, same forty-eight slice positions per study, same series. With a third model it takes longer to run." 2>&1 | tail -2
echo "=== polling score ==="
/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe -u _poll_sub.py $(kaggle competitions submissions rsna-knee-abnormality-detection --csv 2>/dev/null | sed -n 2p | cut -d, -f1)
