#!/bin/bash
# The control the fine-tune needs to be interpretable.
#
# The trainer prints "GOLD macro-AUC" but with --group_csv it validates on 350 scanner-disjoint
# studies against SOFT labels; the champion's published 0.9266 is 58 studies against REAL annotator
# labels. Those two numbers cannot be compared, and treating 0.9382 as "+0.0116" would be exactly
# the kind of false gain that has burned this competition before.
#
# So measure the champion on the SAME ruler: same fold, same 350 studies, same soft labels, same
# k_eval. --bb_lr 0 --head_lr 0 means no weight ever changes, and --limit 8 makes the training pass
# trivial, so this is a pure evaluation of the shipped arm. Costs ~3 minutes, not an epoch.
#
# Runs only after the chain is done, so it can never contend for GPU memory with the fine-tune.
set -u
H=192.220.55.116; P=20566
SSH="ssh -i /c/Users/johna/.ssh/id_ed25519 -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 root@$H"
for i in $(seq 1 700); do
  d=$($SSH 'grep -c "FT CHAIN COMPLETE" /workspace/knee/ft_chain.log 2>/dev/null' 2>/dev/null)
  [ "${d:-0}" != "0" ] && break
  e=$($SSH 'grep -l "OutOfMemoryError\|Traceback" /workspace/knee/run_FT_*.log 2>/dev/null | tr "\n" " "' 2>/dev/null)
  if [ -n "${e// /}" ]; then echo "!! CRASH in $e"; $SSH 'tail -12 /workspace/knee/run_FT_*.log' 2>/dev/null | tail -16; exit 2; fi
  sleep 120
done
echo "$(date -u +%H:%M:%S) chain complete -- running the zero-LR control on the shipped arms"
$SSH 'cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --ckpt timm --norm imagenet \
--labels /workspace/knee/labels_confdist.parquet --bs 2 --grad_ckpt --k 24 --k_eval 24 \
--group_csv /workspace/knee/study_fingerprint.csv --group_folds 5 --group_fold 0 --reclaim 0.7 \
--epochs 1 --bb_lr 0 --head_lr 0 --workers 4 --limit 8"
python -u finetune_box.py $C --init_arm arms/raptor_ft_confdist_swa.pt --tag CTRL_CONFDIST > run_CTRL_CONFDIST.log 2>&1
python -u finetune_box.py $C --init_arm arms/raptor_ft_coatnet_w80_full_swa.pt --tag CTRL_W80 > run_CTRL_W80.log 2>&1
echo "=== SAME RULER: 350 scanner-disjoint studies, soft labels, k_eval 24 ==="
for t in CTRL_CONFDIST FT_CONFDIST CTRL_W80 FT_W80; do
  printf "%-14s %s\n" "$t" "$(grep -h "^ep" run_$t.log 2>/dev/null | tail -1)"
done'
