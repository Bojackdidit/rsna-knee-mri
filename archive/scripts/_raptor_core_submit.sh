#!/bin/bash
# Board-test the Raptor core (solo convnext_small SSL arm, served at norm=none). Waits for the arm
# dataset to be ready, pushes the per-arm-norm kernel, waits for it to run, submits, polls the score.
# This board number - NOT the 0.80 gold - is the verdict on whether the Raptor foundation generalises.
set -u
cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
DS=dreaddevelopment/knee-raptor-core-arm
K=dreaddevelopment/knee-mri-raptor-core

echo "=== wait for dataset ready $(date +%H:%M) ==="
for i in $(seq 1 60); do
  s=$(kaggle datasets status $DS 2>&1 | head -1)
  echo "  [$(date +%H:%M)] ds=$s"; case "$s" in *ready*) break;; esac; sleep 30
done
echo "=== push kernel ==="
OUT=$(kaggle kernels push -p variants/push_raptor_core 2>&1 | tail -1); echo "$OUT"
VER=$(echo "$OUT" | grep -oE 'version [0-9]+' | grep -oE '[0-9]+'); echo "version=$VER"
[ -z "$VER" ] && { echo "ABORT: no version"; exit 1; }
echo "=== champion still resolves? ==="; kaggle kernels status dreaddevelopment/knee-mri-coatnet-plus-convnext 2>&1 | head -1
echo "=== wait for kernel run ==="
SEEN=0
for i in $(seq 1 90); do
  s=$(kaggle kernels status $K 2>&1 | head -1); echo "  [$(date +%H:%M)] $s"
  case "$s" in *RUNNING*|*QUEUED*) SEEN=1;; *COMPLETE*) [ "$SEEN" = 1 ] && break;; *ERROR*|*CANCEL*) echo "ABORT: $s"; exit 1;; esac
  sleep 60
done
echo "=== submit v$VER $(date +%H:%M) ==="
kaggle competitions submit -c rsna-knee-abnormality-detection -k $K -v $VER -f submission.csv \
  -m "A single model built on our own knee self-supervised base instead of a public ImageNet backbone. Same twelve-finding head and settings as our best entry, one model, served at its own input scaling. Testing whether our in-house base generalises better to the hidden test than the public backbones." 2>&1 | tail -2
echo "=== poll score ==="
for i in $(seq 1 45); do
  r=$($PY -c "
import io,subprocess,pandas as pd
d=pd.read_csv(io.StringIO(subprocess.run(['kaggle','competitions','submissions','rsna-knee-abnormality-detection','--csv'],capture_output=True,text=True,timeout=120).stdout)).iloc[0]
print('%s|%s'%(str(d['status']).split('.')[-1],d['publicScore']))" 2>/dev/null)
  echo "  [$(date +%H:%M)] $r"; case "$r" in COMPLETE*0.*|COMPLETE*1.*) echo "RAPTOR CORE BOARD: $r (champion 0.945)"; break;; esac; sleep 60
done
echo "=== done $(date +%H:%M) ==="
