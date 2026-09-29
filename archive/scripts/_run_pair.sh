#!/bin/bash
# Build a shippable PAIR at one window budget. Usage: _run_pair.sh ARCH RES K BBLR PREFIX
#
# The kernel runs two arms concurrently, one per GPU, and interleaving gives them different window
# positions - so the unit that ships is a PAIR, not an arm. The two arms are trained on DIFFERENT
# label sets, which is how the shipped champion pair is built and the only axis that ever produced
# decorrelation here.
#
# Full 4,407-study corpus. Grouped fold 0 held out, so the gate is 870 scanner-disjoint studies -
# 38% larger than the 631 used earlier today, which also tightens the paired minimum-detectable
# difference from ~0.0062 to ~0.0053.
#
# k = k_eval throughout: trained AT the window budget it will be inferred at. Every arm shipped so
# far was trained at k=6 and inferred at 48, and that mismatch is why cutting windows cost accuracy.
set -u
ARCH="$1"; RES="$2"; K="$3"; BBLR="$4"; PREFIX="$5"
EXTRA="${6:-}"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--ckpt timm --norm imagenet --workers 8 $EXTRA \
 --group_csv /workspace/knee/study_fingerprint.csv --group_fold 0 \
 --arch $ARCH --res $RES --k $K --k_eval $K \
 --epochs 16 --bs 8 --bb_lr $BBLR --head_lr 1e-3 --wd 0.02"

echo "===== PAIR $PREFIX | $ARCH @ $RES | k=k_eval=$K | bb_lr $BBLR | $(date -u +%H:%M:%S) ====="
nohup python -u finetune_box.py $COMMON --seed 42 \
  --labels /workspace/knee/labels_opus_blend.parquet --tag ${PREFIX}_a > arm_${PREFIX}_a.log 2>&1 &
sleep 25
nohup python -u finetune_box.py $COMMON --seed 777 \
  --labels /workspace/knee/labels_cd_ext.parquet --tag ${PREFIX}_b > arm_${PREFIX}_b.log 2>&1 &
wait
echo "===== PAIR $PREFIX DONE $(date -u +%H:%M:%S) ====="
for s in a b; do echo "${PREFIX}_$s: $(grep -aoE 'best 0\.[0-9]+' arm_${PREFIX}_$s.log | tail -1)"; done
