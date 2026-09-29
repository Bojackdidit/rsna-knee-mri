#!/bin/bash
# Raptor - ViT-Base 384. A fourth arm that costs nothing in wall clock.
#
# Measured directly on this card, per study at 48 windows and 384px:
#   CoAtNet 88.3 ms (1.00x) | Swin-Base 88.7 (1.00x) | EfficientNetV2-S 63.5 (0.72x)
#   ConvNeXt-Base 47.6 (0.54x) | ViT-Base/16 22.6 (0.26x)
# The kernel puts arm i on cuda:i%2, so the submitted trio sits at cuda:0 = confdist + ConvNeXt
# (1.54x) and cuda:1 = w80 (1.00x). Adding ViT to cuda:1 takes that card to 1.26x, still under the
# 1.54x busy card - so the fourth arm is free in wall clock. Nothing else on the list is.
#
# It is also the most different family we can add: a plain transformer against two CoAtNets and a
# ConvNeXt. The two ConvNeXt arms correlated with each other and the 4-arm blend went DOWN
# (0.9324 against 0.9340 for the trio), so the next arm has to disagree, not just be new.
#
# Labels are confdist, the set that produced the best arm in both families tested so far.
set -u
cd /workspace
export KNEE_DATA_ROOT=/workspace/rsna_knee/raptor_combined
TAG=raptor384_vit

python - <<'PY' || exit 1
import timm, sys
a = "vit_base_patch16_384.augreg_in21k_ft_in1k"
try:
    m = timm.create_model(a, pretrained=False, num_classes=0, in_chans=3,
                          global_pool="token", dynamic_img_size=True)
    print("[arch] %s OK | features %d" % (a, m.num_features))
except Exception as e:
    print("[arch] FAILED %s: %s" % (a, e)); sys.exit(1)
PY

python finetune_raptor_cnn.py --ckpt timm --norm imagenet --workers 8 \
  --tag "$TAG" --arch vit_base_patch16_384.augreg_in21k_ft_in1k --res 384 \
  --labels /workspace/labels_cd_rsna.parquet --slots '' \
  --k 12 --k_eval 24 --epochs 14 --bs 8 --grad_ckpt \
  --bb_lr 3e-5 --head_lr 1e-3 --wd 0.02 --seed 777 > "/workspace/train_${TAG}.log" 2>&1
RC=$?
B=$(grep -oE 'BEST GOLD macro-AUC [0-9.]+' "/workspace/train_${TAG}.log" | tail -1 | awk '{print $NF}')
echo "${TAG} gold=${B:-NONE} rc=${RC}" >> /workspace/RESULTS224.txt
echo "[vit384] done gold=${B:-NONE} rc=${RC}"
