#!/bin/bash
# Scott's families at the settings we now know are correct. Usage: _run_scott.sh ARCH BBLR BS TAG
#
# ResNet has been trained ONCE and badly: resnet50@224 read 0.8440 on the part-1-only corpus, at the
# inherited bb_lr 3e-5, trained at k=6 and inferred at a different budget. Every correction found
# since was applied to EfficientNet and never to ResNet, so half of what Scott names has never had a
# fair test.
#
# Correct settings, each one measured rather than assumed:
#   full 4,407-study corpus   - part 1 alone is 58% of the data and cost ~0.06 on other arms
#   res 384                   - the board priced resolution: 224 tops out at 0.928, 384 gives 0.941
#   k = k_eval = 12           - training at the serve budget removes the window-count penalty
#   bb_lr 3e-4 for convnets   - efficientnet-b0 read 0.8158 at 3e-5 and 0.8598 at 3e-4. This is
#                               family-specific: CoAtNet measured WORSE at every LR above 3e-5 and
#                               diverged at 3e-4, so do not carry this value across families.
#   24 epochs                 - b0 gained +0.0116 going 12 -> 24
#   grouped fold 0, blend labels, seed 42 - same gate as everything else, so it is comparable
set -u
ARCH="$1"; BBLR="$2"; BS="$3"; TAG="$4"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --ckpt timm --norm imagenet --workers 8 --grad_ckpt \
  --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
  --arch "$ARCH" --res 384 \
  --k 12 --k_eval 12 --epochs 24 --bs "$BS" --bb_lr "$BBLR" --head_lr 1e-3 --wd 0.02 --seed 42 \
  --labels /workspace/knee/labels_opus_blend.parquet --tag "$TAG"
