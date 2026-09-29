#!/bin/bash
# Corrected OOF dump. The 2026-09-26 14:01 run crashed every fold on
# '/workspace/knee/study_fingerprint.csv': gate_eval.py reads the group CSV from GE_GROUP, but
# _oof_dump.sh set SWEEP_GROUP (which gate_eval ignores), so it fell back to the Vast default.
# Fix: export GE_GROUP (and GE_K explicitly). Everything else copied from _oof_dump.sh.
set -u
cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
export KNEE_DATA_ROOT='D:\claude2\main_track\knee\phase2\local_box\raptor'
export GE_LABELS='D:\claude2\main_track\knee\labels_blendimp_cd.parquet'
export GE_GROUP='D:\claude2\main_track\knee\study_fingerprint.csv'
export GE_K=12
export KNEE_VRAM_FRAC=0.80 KNEE_DUTY=0.90 KNEE_TEMP_LIMIT=78 KNEE_WORKERS=0
export GATE_OUT_DIR=oof
mkdir -p oof

# absolute powershell path: a Start-Process-launched bash has a bare PATH without powershell, and a
# bare `powershell` returns empty -> the while-loop below waits forever. Full path is launcher-proof.
PS=/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe
running () { "$PS" -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'finetune_box' } | Measure-Object).Count" 2>/dev/null | tr -d '\r' | head -1; }

declare -A ARM=( [0]=LR96_CTRL [1]=LR96_F1 [2]=LR96_F2 [3]=LR96_F3 [4]=LR96_F4 )

for F in 0 1 2 3 4; do
  W=raptor_ft_${ARM[$F]}.pt
  while [ ! -f "$W" ]; do echo "[$(date +%H:%M)] waiting for $W"; sleep 120; done
done
echo "=== all five fold arms present $(date +%H:%M) ==="
while [ "$(running)" != "0" ]; do sleep 120; done
sleep 30

for F in 0 1 2 3 4; do
  T=${ARM[$F]}
  W=raptor_ft_${T}.pt
  if [ -f "oof/oof_${T}.npz" ]; then echo "$T already dumped"; continue; fi
  echo "=== dumping fold $F from $T $(date +%H:%M) ==="
  GE_FOLD=$F $PY -u _run_sweep.py gate_eval.py "$W" "$T" 2>&1 | grep -viE 'warn|symlink' | tail -4
  [ -f "oof/raptor_gold_${T}.npz" ] && mv "oof/raptor_gold_${T}.npz" "oof/oof_${T}.npz" && echo "  -> oof/oof_${T}.npz"
done

echo "=== assembling $(date +%H:%M) ==="
$PY _oof_validate.py --per-finding
echo "=== OOF assembly done $(date +%H:%M) ==="
