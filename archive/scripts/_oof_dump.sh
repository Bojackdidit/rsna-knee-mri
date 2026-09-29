#!/bin/bash
# Score every fold arm on ITS OWN held-out fold and assemble the out-of-fold validation set.
#
# The one rule that makes this valid: fold k's arm is graded with GE_FOLD=k. Grading a fold-k arm
# on fold 0 scores it on its own training data, which is what made E0 read 0.9536 against a true
# 0.8961. gate_eval.py takes the fold from GE_FOLD and stamps it into the dump.
set -u
cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
export KNEE_DATA_ROOT='D:\claude2\main_track\knee\phase2\local_box\raptor'
# gate_eval.py reads GE_LABELS / SWEEP_GROUP - NOT SWEEP_LABELS, which is what the sweep scripts
# use. Setting the wrong one left it on its hardcoded Vast default (/workspace/knee/...) and the
# fold-0 dump died on a missing file. Must be the labels the folds were TRAINED on, or the gate is
# scoring against a different target than the model learned.
export GE_LABELS='D:\claude2\main_track\knee\labels_blendimp_cd.parquet'
export SWEEP_LABELS='D:\claude2\main_track\knee\labels_blendimp_cd.parquet'
export SWEEP_GROUP='D:\claude2\main_track\knee\study_fingerprint.csv'
export KNEE_VRAM_FRAC=0.80 KNEE_DUTY=0.90 KNEE_TEMP_LIMIT=78 KNEE_WORKERS=0
export GATE_OUT_DIR=oof
mkdir -p oof

running () { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'finetune_box' } | Measure-Object).Count" 2>/dev/null | tr -d '\r' | head -1; }

declare -A ARM=( [0]=LR96_CTRL [1]=LR96_F1 [2]=LR96_F2 [3]=LR96_F3 [4]=LR96_F4 )

# Wait for EVERY fold weight before dumping ANY of them. Dumping fold-by-fold as each lands looks
# tidier but races the trainer: the folds chain starts the next fold 30 s after the last one ends,
# this loop polls every 120 s, and gate_eval peaks at 6.2 GB against training's 8.5 GB - together
# they exceed the 12.9 GB card. Waiting for all five means the GPU is provably idle when we start.
for F in 0 1 2 3 4; do
  W=raptor_ft_${ARM[$F]}.pt
  while [ ! -f "$W" ]; do echo "[$(date +%H:%M)] waiting for $W"; sleep 300; done
done
echo "=== all five fold arms present $(date +%H:%M) ==="
while [ "$(running)" != "0" ]; do sleep 120; done
sleep 60

for F in 0 1 2 3 4; do
  T=${ARM[$F]}
  W=raptor_ft_${T}.pt
  while [ "$(running)" != "0" ]; do sleep 120; done
  if [ -f "oof/oof_${T}.npz" ]; then echo "$T already dumped"; continue; fi
  echo "=== dumping fold $F from $T $(date +%H:%M) ==="
  GE_FOLD=$F $PY -u _run_sweep.py gate_eval.py "$W" "$T" 2>&1 | grep -viE 'warn|symlink' | tail -3
  [ -f "oof/raptor_gold_${T}.npz" ] && mv "oof/raptor_gold_${T}.npz" "oof/oof_${T}.npz" && echo "  -> oof/oof_${T}.npz"
done

echo "=== assembling ==="
$PY _oof_validate.py --per-finding
