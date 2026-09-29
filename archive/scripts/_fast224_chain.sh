#!/bin/bash
# The moment the corpus lands, find out what 224 costs us.
#
# Every top competitor on this board trains at 224 or 288 - BELOW our 384 - and one of them reaches
# 0.950 on a single fold. If 224 costs us little or nothing, everything downstream gets about three
# times cheaper (384^2/224^2 = 2.94): five folds become affordable, label variants can be tested in
# an evening instead of overnight, and the efficiency board, where we sit 20th of 3,771, improves for
# free. If 224 costs us a lot, that is worth knowing too, because it would mean the leaders are ahead
# for a reason we have not found yet.
#
# Single variable: this is the ConvNeXt-Base recipe that scored gold 0.9270 at 384, with ONLY the
# resolution changed. Twenty epochs rather than twenty-four so it lands in a usable time.
set -u
ulimit -n 65536
cd /workspace/knee
while [ ! -f /workspace/knee/corpus_v11/all_vols.npy ] || pgrep -f "[r]sync" > /dev/null; do
  [ -f STOP_QUEUE ] && exit 0
  sleep 120
done
sleep 30
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "fast224: corpus ready, starting $(date -u +%F' '%H:%M:%S)" >> chain.log
python -u finetune_box.py --arch convnext_base.fb_in22k_ft_in1k_384 --res 224 --ckpt timm \
  --norm imagenet --labels /workspace/knee/labels_confdist.parquet --bs 8 --grad_ckpt \
  --k 24 --k_eval 24 --epochs 20 --bb_lr 3e-5 --head_lr 1e-3 --workers 4 --seed 4242 \
  --aug geo,erase,blur,noise --ema 0.9998 --tag CNXT224 > run_CNXT224.log 2>&1
echo "fast224: done rc=$? $(date -u +%F' '%H:%M:%S)" >> chain.log
