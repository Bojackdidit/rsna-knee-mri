#!/bin/bash
# Pseudo-labelling, step one: get two DIFFERENT architectures to give an opinion on every training
# study, so their consensus can be blended into the labels the next arm trains on.
#
# Four independent competitors at the top of this board name labels as the lever, and one of them
# reached 0.950 on a single fold after finding exactly what we already measured - that the image
# model's own predictions sit closer to the radiologists than the extracted labels it was trained on.
# We closed the label axis on the gold-58 gate, which we had already proved anti-predictive of arm
# quality three times out of three, and it has never had a board read of its own.
#
# The teacher is the CoAtNet champion plus the ConvNeXt arm, because a single model fed its own
# predictions learns nothing it did not already believe; two architectures that disagree in different
# places carry information neither one has alone.
set -eux
cd /workspace/knee
mkdir -p teach && cd teach
for d in raptor-knee-scratch-aug2 raptor-knee-convnext-arm; do
  [ -f "$d.zip" ] || kaggle datasets download -d "dreaddevelopment/$d" -p . --force
  python - "$d.zip" <<'PY'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(".")
print("extracted", sys.argv[1])
PY
done
ls -la /workspace/knee/teach/*.pt
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --ckpt timm \
  --norm imagenet --labels /workspace/knee/labels_confdist.parquet --bs 2 --grad_ckpt \
  --k 24 --k_eval 24 --epochs 1 --workers 4 --seed 4242 --ema 0.0 \
  --init_arm /workspace/knee/teach/raptor_ft_SCRATCH_AUG2.pt --eval_only --eval_ids all \
  --tag TEACH_COAT > teach_coat.log 2>&1
python -u finetune_box.py --arch convnext_base.fb_in22k_ft_in1k_384 --res 384 --ckpt timm \
  --norm imagenet --labels /workspace/knee/labels_confdist.parquet --bs 2 --grad_ckpt \
  --k 24 --k_eval 24 --epochs 1 --workers 4 --seed 4242 --ema 0.0 \
  --init_arm /workspace/knee/teach/raptor_ft_CONVNEXT_E24.pt --eval_only --eval_ids all \
  --tag TEACH_CNXT > teach_cnxt.log 2>&1
tail -3 teach_coat.log; tail -3 teach_cnxt.log
