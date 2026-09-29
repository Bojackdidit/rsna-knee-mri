#!/bin/bash
# Dump the ConvNeXt fold-0 OOF (once the trainer frees the GPU), then run the diversity comparison.
set -u
cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
PS=/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe
export KNEE_DATA_ROOT='D:\claude2\main_track\knee\phase2\local_box\raptor'
export GE_LABELS='D:\claude2\main_track\knee\labels_blendimp_cd.parquet'
export GE_GROUP='D:\claude2\main_track\knee\study_fingerprint.csv'
export GE_K=12
export KNEE_VRAM_FRAC=0.65 KNEE_DUTY=0.90 KNEE_TEMP_LIMIT=78 KNEE_WORKERS=0
export GATE_OUT_DIR=oof

running () { "$PS" -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'finetune_box' } | Measure-Object).Count" 2>/dev/null | tr -d '\r' | head -1; }

# wait for CNXT_F0 weights + the trainer to fully exit (free the GPU)
while [ ! -f raptor_ft_CNXT_F0.pt ]; do sleep 60; done
while [ "$(running)" != "0" ]; do sleep 60; done
sleep 20

if [ ! -f oof/oof_CNXT_F0.npz ]; then
  echo "=== dumping ConvNeXt fold-0 OOF (GE_FOLD=0) $(date +%H:%M) ==="
  GE_FOLD=0 $PY -u _run_sweep.py gate_eval.py raptor_ft_CNXT_F0.pt CNXT_F0 2>&1 | grep -viE 'warn|symlink' | tail -4
  [ -f oof/raptor_gold_CNXT_F0.npz ] && mv oof/raptor_gold_CNXT_F0.npz oof/oof_CNXT_F0.npz && echo "  -> oof/oof_CNXT_F0.npz"
fi

echo "=== architecture-diversity comparison (fold-0, 870 held-out) ==="
$PY _cnxt_compare.py
echo "=== eval done $(date +%H:%M) ==="
