#!/bin/bash
# Anti-scanner augmentation, tested on the RIGHT instrument for the first time.
#
# aug_scanner was refuted once on gold-58 (SCANNER_ARM 0.9247 vs 0.9315) - but gold is
# scanner-CONTAMINATED and rewards the scanner shortcut aug_scanner removes, and that run also cut
# epochs 24->16 (a second variable, OneCycleLR confound). It has NEVER been measured on the
# scanner-grouped OOF (built 2026-09-26) or the board. The leaders out-generalize us because their
# simpler models overfit the scanner less; aug_scanner attacks exactly that (outputs predict scanner
# at 2.1x chance, per-scanner AUC spread 0.048).
#
# CLEAN single-variable test: LR96_CTRL's EXACT recipe (coatnet_rmlp_2_rw_384, res 384,
# geo/erase/blur/noise aug, k12, 10 epochs, seed 42, scanner-grouped fold-0) with ONE flag added:
# --aug_scanner. Then dump its held-out-scanner OOF and compare to the control on the identical 870.
# Ends in a keep(board-test)/kill(closed) decision. Throttled - Dread may be at the machine.
set -u
export PATH="/usr/bin:/bin:/usr/local/bin:$PATH"   # detached launch has no MSYS PATH; restore coreutils
cd /d/claude2/main_track/knee
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
export KNEE_DATA_ROOT='D:\claude2\main_track\knee\phase2\local_box\raptor'
export KNEE_VRAM_FRAC=0.68 KNEE_DUTY=0.72 KNEE_TEMP_LIMIT=78 KNEE_WORKERS=0

echo "=== SCAN_GEN_F0 train (control recipe + --aug_scanner) starting $(date +%H:%M) ==="
$PY -u _run_sweep.py finetune_box.py \
  --ckpt timm --norm imagenet --workers 0 --grad_ckpt \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
  --aug geo,erase,blur,noise --aug_scanner --k 12 --k_eval 12 --epochs 10 --bs 2 \
  --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --group_csv 'D:\claude2\main_track\knee\study_fingerprint.csv' --group_folds 5 --group_fold 0 \
  --labels 'D:\claude2\main_track\knee\labels_blendimp_cd.parquet' \
  --tag SCAN_GEN_F0
[ -f raptor_ft_SCAN_GEN_F0.pt ] || { echo "ABORT: no weights produced"; exit 1; }

echo "=== dump held-out-scanner fold-0 OOF $(date +%H:%M) ==="
export GE_LABELS='D:\claude2\main_track\knee\labels_blendimp_cd.parquet'
export GE_GROUP='D:\claude2\main_track\knee\study_fingerprint.csv'
export GE_K=12 GATE_OUT_DIR=oof
GE_FOLD=0 $PY -u _run_sweep.py gate_eval.py raptor_ft_SCAN_GEN_F0.pt SCAN_GEN_F0 2>&1 | grep -viE 'warn|symlink' | tail -3
[ -f oof/raptor_gold_SCAN_GEN_F0.npz ] && mv oof/raptor_gold_SCAN_GEN_F0.npz oof/oof_SCAN_GEN_F0.npz
[ -f oof/oof_SCAN_GEN_F0.npz ] || { echo "ABORT: no OOF dumped"; exit 1; }

echo "=== DECISION: aug_scanner vs control on held-out scanners $(date +%H:%M) ==="
$PY _scan_gen_compare.py
echo "=== scanner-gen test done $(date +%H:%M) ==="
