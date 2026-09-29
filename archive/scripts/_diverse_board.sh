#!/bin/bash
# Diverse fold ensemble -> board. Waits for the ConvNeXt-fold dataset to be ready, verifies the
# champion kernel is untouched, pushes the new kernel (separate id), waits for it to run, re-verifies
# the champion, submits, polls the score. best-of protects the 0.945 standing throughout.
set -u
export PATH="/usr/bin:/bin:/usr/local/bin:$PATH"
cd /d/claude2/main_track/knee
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
DS=dreaddevelopment/knee-convnext-fold-arms
K=dreaddevelopment/knee-mri-diverse-fold-ensemble
CHAMP=dreaddevelopment/knee-mri-coatnet-plus-convnext

echo "[board] wait for dataset ready $(date -u +%H:%M)"
for i in $(seq 1 160); do
  s=$(kaggle datasets status $DS 2>&1 | head -1)
  echo "  [$(date -u +%H:%M)] ds=$s"
  case "$s" in *ready*) break;; esac
  sleep 30
done
kaggle datasets status $DS 2>&1 | head -1 | grep -q ready || { echo "[board] ABORT: dataset never ready"; exit 1; }

echo "[board] champion resolves BEFORE push (must stay live):"
kaggle kernels status $CHAMP 2>&1 | head -1

echo "[board] push kernel $(date -u +%H:%M)"
OUT=$(kaggle kernels push -p variants/push_diverse_fold 2>&1 | tail -3); echo "$OUT"
VER=$(echo "$OUT" | grep -oE 'version [0-9]+' | grep -oE '[0-9]+' | tail -1); echo "version=$VER"
[ -z "$VER" ] && { echo "[board] ABORT: no version pushed"; exit 1; }

echo "[board] wait for kernel run $(date -u +%H:%M)"
SEEN=0
for i in $(seq 1 100); do
  s=$(kaggle kernels status $K 2>&1 | head -1); echo "  [$(date -u +%H:%M)] $s"
  case "$s" in
    *RUNNING*|*QUEUED*) SEEN=1;;
    *COMPLETE*) [ "$SEEN" = 1 ] && break;;
    *ERROR*|*CANCEL*) echo "[board] ABORT kernel: $s"; exit 1;;
  esac
  sleep 60
done

echo "[board] champion STILL resolves AFTER push (id-collision guard):"
kaggle kernels status $CHAMP 2>&1 | head -1

echo "[board] submit v$VER $(date -u +%H:%M)"
kaggle competitions submit -c rsna-knee-abnormality-detection -k $K -v $VER -f submission.csv \
  -m "Four models reading each knee together: two convolution-attention hybrids and two convolutional networks, each trained on a different four fifths of the studies. Two kinds of model that also saw different data disagree with one another more, and averaging models that disagree has made our ensembles worth more than their parts. Testing whether adding that data diversity on top of the two-architecture mix lifts our best entry." 2>&1 | tail -3

echo "[board] poll score $(date -u +%H:%M)"
for i in $(seq 1 45); do
  r=$($PY -c "
import io,subprocess,pandas as pd
d=pd.read_csv(io.StringIO(subprocess.run(['kaggle','competitions','submissions','rsna-knee-abnormality-detection','--csv'],capture_output=True,text=True,timeout=120).stdout)).iloc[0]
print('%s|%s'%(str(d['status']).split('.')[-1],d['publicScore']))" 2>/dev/null)
  echo "  [$(date -u +%H:%M)] $r"
  case "$r" in COMPLETE*0.*|COMPLETE*1.*) echo "[board] DIVERSE FOLD ENSEMBLE BOARD: $r  (champion 0.945)"; break;; esac
  sleep 60
done
echo "[board] DONE $(date -u +%H:%M)"
