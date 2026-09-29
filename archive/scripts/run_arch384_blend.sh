#!/bin/bash
# Raptor - ConvNeXt-Base 384. The diversity lever that has never actually been tested for shipping.
#
# The kernel prepares each study's windows ONCE and asserts "single pass needs one resolution", so
# arms must agree on RESOLUTION - but load_model reads the architecture out of the checkpoint and
# builds it generically, so architecture is free. Every non-CoAtNet arm we ever trained sat at a
# resolution the kernel cannot serve alongside 384 (EfficientNetV2 at 480, ConvNeXt-v2 at 336,
# the 224 arms trained tonight), which is why a second FAMILY has never once ridden in a submission.
#
# So: a different family at the one resolution that ships, on the shippable 44-slot corpus, with
# full slots. Labels are confdist, held constant because that set produced our strongest arm
# (0.9263) - architecture is the single variable being changed.
#
# ConvNeXt-Base rather than Large: 89M against 198M, so it costs about what a CoAtNet arm costs
# and fits the credit left, while still being a genuinely different inductive bias.
set -u
cd /workspace
export KNEE_DATA_ROOT=/workspace/rsna_knee/raptor_combined
TAG=raptor384_cnb_blend

# fail fast: do not burn an hour of GPU to find out the timm tag is wrong
python - <<'PY' || exit 1
import timm, sys
a = "convnext_base.fb_in22k_ft_in1k_384"
try:
    m = timm.create_model(a, pretrained=False, num_classes=0, in_chans=3, global_pool="avg")
    print("[arch] %s OK | features %d" % (a, m.num_features))
except Exception as e:
    print("[arch] FAILED %s: %s" % (a, e)); sys.exit(1)
PY

python finetune_raptor_cnn.py --ckpt timm --norm imagenet --workers 8 \
  --tag "$TAG" --arch convnext_base.fb_in22k_ft_in1k_384 --res 384 \
  --labels /workspace/labels_opus_blend.parquet --slots '' \
  --k 12 --k_eval 24 --epochs 14 --bs 8 --grad_ckpt \
  --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 777 > "/workspace/train_${TAG}.log" 2>&1
RC=$?
B=$(grep -oE 'BEST GOLD macro-AUC [0-9.]+' "/workspace/train_${TAG}.log" | tail -1 | awk '{print $NF}')
echo "${TAG} gold=${B:-NONE} rc=${RC}" >> /workspace/RESULTS224.txt
echo "[arch384] done gold=${B:-NONE} rc=${RC}"
