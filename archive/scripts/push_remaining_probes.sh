#!/bin/bash
# Push the remaining per-finding probe kernels, respecting Kaggle's limit of 2 concurrent batch
# GPU sessions. Pushing more than 2 at once returns:
#     Kernel push error: Maximum batch GPU session count of 2 reached.
# so they have to be fed in as slots free up.
set -u
cd "$(dirname "$0")"
PROBES="mcl fracture contusion medial_meniscus baker_s medial_oa acl"
for p in $PROBES; do
  d="variants/push_knee_mri_${p}_probe"
  [ -d "$d" ] || { echo "MISSING $d"; continue; }
  for attempt in $(seq 1 40); do
    out=$(kaggle kernels push -p "$d" 2>&1 | tail -1)
    case "$out" in
      *"successfully pushed"*) echo "$(date -u +%H:%M:%S) $p OK"; break;;
      *"Maximum batch GPU session"*) sleep 60;;
      *) echo "$(date -u +%H:%M:%S) $p ERROR: $out"; break;;
    esac
  done
done
echo "=== all probe kernels pushed $(date -u +%H:%M:%S) ==="
kaggle kernels list --user dreaddevelopment --search "knee-mri" --page-size 40 2>/dev/null | grep -c probe
