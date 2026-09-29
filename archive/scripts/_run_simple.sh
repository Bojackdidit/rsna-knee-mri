#!/bin/bash
# The SIMPLE BASELINE, built to an 11th-place competitor's stated recipe. Usage: _run_simple.sh ARCH RES POOL TAG
#
# a top-10 competitor (11th), in the competition discussion:
#   "Start with 224px / 288px and a simple CNN encoder, like a ResNet34 or EffNetB0. No attention
#    mechanisms, just use simple pooling operations. You can score in the 0.94+ range with this alone"
# a top-15 competitor (55th) independently: "also have 0.94+ at 224px with small simple model".
#
# Everything we have ever shipped is the opposite of that: a per-finding softmax ATTENTION head over
# windows, on CoAtNet-384. He lists attention as an experimental thing to try LAST. This is why our
# ResNet-50 read 0.8885 - we put ResNet inside OUR heavy pipeline instead of the light one it belongs
# in, so that was a test of the pipeline, not of ResNet.
#
# It also happens to be the speed answer: a ResNet34 at 224 with mean pooling is roughly an order of
# magnitude cheaper per study than CoAtNet-384 with 48-window attention.
#
# Validation is the 870-study scanner-disjoint fold against REPORT-EXTRACTED labels - which is exactly
# what he prescribes ("not the provided 58 labels and not against the LB - neither are large enough
# sample sizes to resolve 0.002-0.003 differences"). His noise threshold is ~0.003; our measured
# paired MDD on this fold is 0.0053, so demand better than that before believing anything.
#
# bb_lr 3e-4 is the CONVNET value (efficientnet-b0 read 0.8158 at 3e-5 and 0.8598 at 3e-4). Do not
# carry it to CoAtNet, which measured worse at every LR above 3e-5 and diverged at 3e-4.
set -u
ARCH="$1"; RES="$2"; POOL="$3"; TAG="$4"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --ckpt timm --norm imagenet --workers 8 \
  --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
  --arch "$ARCH" --res "$RES" --pool "$POOL" \
  --k 12 --k_eval 12 --epochs 24 --bs 16 --bb_lr 3e-4 --head_lr 1e-3 --wd 0.02 --seed 42 \
  --labels /workspace/knee/labels_opus_blend.parquet --tag "$TAG"
