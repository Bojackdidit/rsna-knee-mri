#!/bin/bash
# Cheap directional read on the ep6 SCAN_GEN checkpoint (the run was cut at ~3.7h before 10 epochs).
# Dump its held-out-scanner fold-0 OOF and compare to the 10-epoch control. CAVEAT: 7 epochs vs 10,
# and SCAN_GEN was still climbing (gold ep4 .835 -> ep5 .842 -> ep6 .852), so it is UNDERTRAINED here.
# Read: if the undertrained aug_scanner arm already MATCHES/BEATS the control on held-out-scanner
# generalization or shrinks the per-scanner spread -> strong positive, worth the clean 10-epoch run.
# If it's well below -> inconclusive (undertraining, not a verdict), run the full thing.
set -u
cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
export KNEE_DATA_ROOT='D:\claude2\main_track\knee\phase2\local_box\raptor'
export GE_LABELS='D:\claude2\main_track\knee\labels_blendimp_cd.parquet'
export GE_GROUP='D:\claude2\main_track\knee\study_fingerprint.csv'
export GE_K=12 GATE_OUT_DIR=oof KNEE_VRAM_FRAC=0.65 KNEE_DUTY=0.70 KNEE_TEMP_LIMIT=78 KNEE_WORKERS=0

echo "=== dump ep6 SCAN_GEN held-out-scanner OOF $(date +%H:%M) ==="
GE_FOLD=0 $PY -u _run_sweep.py gate_eval.py raptor_ft_SCAN_GEN_F0.pt SCAN_GEN_F0 2>&1 | grep -viE 'warn|symlink' | tail -4
[ -f oof/raptor_gold_SCAN_GEN_F0.npz ] && mv oof/raptor_gold_SCAN_GEN_F0.npz oof/oof_SCAN_GEN_F0.npz
[ -f oof/oof_SCAN_GEN_F0.npz ] || { echo "ABORT: no OOF dumped"; exit 1; }

echo "=== DIRECTIONAL DECISION (ep6, undertrained) $(date +%H:%M) ==="
$PY _scan_gen_compare.py
echo "=== ep6 eval done $(date +%H:%M) ==="
