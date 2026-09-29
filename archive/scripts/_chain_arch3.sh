#!/bin/bash
# Chain a THIRD architecture on box 3 once CONVNEXT_E24 exits. The corpus is already assembled here,
# so this costs no setup - a fresh box would spend about two hours rebuilding it first.
#
# Architecture is passed in, because which one is worth running depends on the CoAtNet+ConvNeXt board
# read. CoAtNet is a conv/transformer hybrid and ConvNeXt is a modern pure CNN, so a pure transformer
# (swinv2) is the most different third member; effv2 is the safer, faster CNN option.
#
# Usage: chain_arch3.sh <timm-arch> <tag> <epochs>
set -u
ARCH="${1:?arch}"; TAG="${2:?tag}"; EP="${3:-16}"
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/knee
while pgrep -f "[f]inetune_box.py" > /dev/null; do sleep 120; done
echo "chain3: starting $TAG ($ARCH, $EP ep) $(date -u +%F' '%H:%M:%S)" >> ft_chain.log
setsid nohup python -u finetune_box.py \
  --arch "$ARCH" --res 384 --ckpt timm --norm imagenet \
  --labels /workspace/knee/labels_confdist.parquet --bs 2 --grad_ckpt --k 24 --k_eval 24 \
  --epochs "$EP" --bb_lr 3e-5 --head_lr 1e-3 --workers 4 --seed 4242 \
  --aug geo,erase,blur,noise --ema 0.9998 --tag "$TAG" \
  > "run_$TAG.log" 2>&1 < /dev/null &
echo "chain3: launched $(date -u +%F' '%H:%M:%S)" >> ft_chain.log
