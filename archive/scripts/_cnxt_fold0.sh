#!/bin/bash
# Diverse-arm experiment, gated on the new 4,349-study OOF instrument.
# Train ConvNeXt-Base 384 on the SAME group fold-0 split the CoAtNet folds used, so its held-out
# predictions land on the exact 870 studies as oof_LR96_CTRL. Then we can measure (a) per-study
# decorrelation vs the CoAtNet fold-0 arm and (b) whether rank-mean(CoAtNet_f0, ConvNeXt_f0) lifts
# the fold-0 held-out macro beyond the ~0.006 noise floor on 870 studies. Architecture diversity is
# the ONE board-confirmed lever (0.944->0.945) and ConvNeXt is half the champion, so this builds a
# known-good ensemble component, not a speculative bet. Bounded (~6h), ends in a keep/continue call.
set -u
cd "$(dirname "$0")"
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
export KNEE_DATA_ROOT='D:\claude2\main_track\knee\phase2\local_box\raptor'
export KNEE_VRAM_FRAC=0.65 KNEE_DUTY=0.55 KNEE_TEMP_LIMIT=75 KNEE_WORKERS=0

if [ -f raptor_ft_CNXT_F0.pt ]; then echo "CNXT_F0 already trained"; else
echo "=== ConvNeXt fold-0 starting $(date +%H:%M) ==="
$PY -u _run_sweep.py finetune_box.py \
  --ckpt timm --norm imagenet --workers 0 --grad_ckpt \
  --arch convnext_base.fb_in22k_ft_in1k_384 --res 384 \
  --aug geo,erase,blur,noise --k 12 --k_eval 12 --epochs 10 --bs 2 \
  --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --group_csv 'D:\claude2\main_track\knee\study_fingerprint.csv' --group_folds 5 --group_fold 0 \
  --labels 'D:\claude2\main_track\knee\labels_blendimp_cd.parquet' \
  --tag CNXT_F0
echo "=== ConvNeXt fold-0 done $(date +%H:%M) ==="
fi
