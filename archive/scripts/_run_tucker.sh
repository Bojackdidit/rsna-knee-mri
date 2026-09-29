#!/bin/bash
# a top-10 competitor shape + Dread's reclaim. Usage: _run_tucker.sh ARCH RES POOL K AUG BS TAG
#
# The intel, from 11th place in this competition, seconded by 55th:
#   "Start with 224px / 288px and a simple CNN encoder, like a ResNet34 or EffNetB0 ... You can score
#    in the 0.94+ range with this alone"
#   "Validate against the report extracted labels, not the provided 58 labels and not against the LB"
#   "Then ablate through the standard image augmentation stacks ... Change one thing at a time"
#
# Plus Dread's instruction: 70% of the held-out fold fine-tuned back into the model.
#
# Both arms of a pair MUST share resolution - the kernel builds each study's windows once and asserts
# a single resolution - so a 288 pair is b0@288 + rn34@288, which is also cross-family, the only
# pairing measured to add (0.9269 vs 0.9174 same-family, decorrelation 0.8875 vs 0.9357).
#
# Cost on this box, measured: ~75 s/epoch at 288/k24, so a full 24-epoch run is about $1.90 against
# $16.50 for CoAtNet-384. The whole ablation costs less than one CoAtNet run.
set -u
ARCH="$1"; RES="$2"; POOL="$3"; K="$4"; AUG="$5"; BS="$6"; TAG="$7"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --ckpt timm --norm imagenet --workers 32 \
  --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 --reclaim 0.7 \
  --arch "$ARCH" --res "$RES" --pool "$POOL" --aug "$AUG" \
  --k "$K" --k_eval "$K" --epochs 24 --bs "$BS" --bb_lr 3e-4 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --labels /workspace/knee/labels_opus_blend.parquet --tag "$TAG"
