#!/bin/bash
# Wait for the fold-ensemble kernel (v1) to finish running, then submit it and poll the board score.
set -u
cd "$(dirname "$0")"
K=dreaddevelopment/knee-mri-coatnet-fold-ensemble
VER=1
SEEN=0
for i in $(seq 1 90); do
  s=$(kaggle kernels status $K 2>&1 | head -1)
  echo "[$(date +%H:%M)] $s"
  case "$s" in
    *RUNNING*|*QUEUED*) SEEN=1;;
    *COMPLETE*) [ "$SEEN" = 1 ] && { echo "kernel complete"; break; };;
    *ERROR*|*CANCEL*) echo "ABORT: kernel failed: $s"; exit 1;;
  esac
  sleep 60
done
echo "=== submitting fold ensemble v$VER $(date +%H:%M) ==="
kaggle competitions submit -c rsna-knee-abnormality-detection -k $K -v $VER -f submission.csv \
  -m "Four models of the same design, each trained while holding out a different fifth of the studies, then averaged. This is the fold-ensembling the leading team reports as worth about nine thousandths of a point. Same resolution and settings as our best entry, with one more model on each card, so it runs a little longer." 2>&1 | tail -2
echo "=== polling score $(date +%H:%M) ==="
for i in $(seq 1 45); do
  row=$(kaggle competitions submissions rsna-knee-abnormality-detection --csv 2>/dev/null | sed -n 2p)
  sc=$(echo "$row" | awk -F, '{print $6}')
  st=$(echo "$row" | awk -F, '{print $5}')
  echo "  [$(date +%H:%M)] status=$st score=$sc"
  case "$sc" in
    0.*|1.*) echo "SCORED fold-ensemble board=$sc"; break;;
  esac
  sleep 60
done
echo "=== fold-ensemble submit flow done $(date +%H:%M) ==="
