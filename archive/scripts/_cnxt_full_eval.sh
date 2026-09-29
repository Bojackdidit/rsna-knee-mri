#!/bin/bash
# After all 5 ConvNeXt folds exist and the GPU is free: dump the OOF for F1-F4 (F0 already dumped),
# then run the full-instrument CoAtNet-vs-ConvNeXt-vs-blend comparison over all 4,349 studies.
set -u
cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
PS=/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe
export KNEE_DATA_ROOT='D:\claude2\main_track\knee\phase2\local_box\raptor'
export GE_LABELS='D:\claude2\main_track\knee\labels_blendimp_cd.parquet'
export GE_GROUP='D:\claude2\main_track\knee\study_fingerprint.csv'
export GE_K=12 KNEE_VRAM_FRAC=0.65 KNEE_DUTY=0.90 KNEE_TEMP_LIMIT=78 KNEE_WORKERS=0 GATE_OUT_DIR=oof

running () { "$PS" -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'finetune_box' } | Measure-Object).Count" 2>/dev/null | tr -d '\r' | head -1; }

for F in 1 2 3 4; do while [ ! -f "raptor_ft_CNXT_F${F}.pt" ]; do sleep 120; done; done
while [ "$(running)" != "0" ]; do sleep 60; done
sleep 20

declare -A T=( [0]=CNXT_F0 [1]=CNXT_F1 [2]=CNXT_F2 [3]=CNXT_F3 [4]=CNXT_F4 )
for F in 1 2 3 4; do
  [ -f "oof/oof_${T[$F]}.npz" ] && { echo "${T[$F]} already dumped"; continue; }
  echo "=== dumping ${T[$F]} OOF (GE_FOLD=$F) $(date +%H:%M) ==="
  GE_FOLD=$F $PY -u _run_sweep.py gate_eval.py "raptor_ft_${T[$F]}.pt" "${T[$F]}" 2>&1 | grep -viE 'warn|symlink' | tail -3
  [ -f "oof/raptor_gold_${T[$F]}.npz" ] && mv "oof/raptor_gold_${T[$F]}.npz" "oof/oof_${T[$F]}.npz" && echo "  -> oof/oof_${T[$F]}.npz"
done

echo "=== FULL architecture-diversity comparison (all 4,349 OOF studies) ==="
$PY _cnxt_full_compare.py
echo "=== full eval done $(date +%H:%M) ==="
