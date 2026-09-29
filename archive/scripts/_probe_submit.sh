#!/bin/bash
# Wait for a probe kernel to finish, then submit it. My first attempt at this had NO sleep in the
# poll loop, so it burned 40 iterations in seconds, exited RUNNING, and the submit 400'd.
set -u
K="$1"; MSG="$2"
for i in $(seq 1 90); do
  S=$(kaggle kernels status "dreaddevelopment/$K" 2>&1 | head -1)
  case "$S" in
    *COMPLETE*) echo "[$K] complete after $((i*20))s"; break;;
    *ERROR*|*CANCEL*) echo "[$K] FAILED: $S"; exit 1;;
  esac
  sleep 20
done
case "$S" in *COMPLETE*) ;; *) echo "[$K] still not complete: $S"; exit 1;; esac
kaggle competitions submit -c rsna-knee-abnormality-detection -k "dreaddevelopment/$K" -f submission.csv -v 1 -m "$MSG" 2>&1 | tail -3
