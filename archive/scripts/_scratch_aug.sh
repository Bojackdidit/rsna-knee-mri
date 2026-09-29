#!/bin/bash
# From-scratch training with augmentation, on the champion's own recipe. Fires when stage 2 ends.
#
# This is where augmentation's evidence actually lives: the +0.0150 A/B was a model learning from
# ImageNet init over 24 epochs, not a converged model nudged for six. Fine-tuning the champion was
# measured today at -0.0045 on the gold 58 and 0.936 on the board, so the polish route is closed and
# this is the open one.
#
# Two things carried over from what today taught us:
#   --ema 0.9998  keeps a running average of the weights and SAVES that average. The champion is an
#                 average of three epochs, and destroying that average is the best explanation we
#                 have for why fine-tuning lost. This keeps it by construction.
#   no --group_csv  trains on all 4,349 studies, exactly as the champion did, and validates on the
#                 58 real-doctor-labelled studies - the one ruler nothing ever trains on, and the
#                 same one the champion's 0.9266 is quoted on. Directly comparable, no caveats.
set -u
H=192.220.55.116; P=20566
SSH="ssh -i /c/Users/johna/.ssh/id_ed25519 -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 root@$H"
echo "$(date -u +%H:%M:%S) waiting for stage 2 to complete"
for i in $(seq 1 900); do
  d=$($SSH 'grep -c "STAGE2 COMPLETE" /workspace/knee/ft_chain.log 2>/dev/null' 2>/dev/null)
  [ "${d:-0}" != "0" ] && { echo "$(date -u +%H:%M:%S) stage 2 complete"; break; }
  e=$($SSH 'grep -l "OutOfMemoryError\|Traceback" /workspace/knee/run_FULL_*.log 2>/dev/null | tr "\n" " "' 2>/dev/null)
  if [ -n "${e// /}" ]; then echo "!! stage 2 CRASH in $e"; $SSH 'tail -12 /workspace/knee/run_FULL_*.log' 2>/dev/null | tail -16; exit 2; fi
  sleep 120
done
$SSH 'cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cat > scratch_aug.sh <<EOS
#!/bin/bash
set -u
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/knee
echo "SCRATCH_AUG start \$(date -u +%F\ %H:%M:%S)" >> ft_chain.log
python -u finetune_box.py \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --ckpt timm --norm imagenet \
  --labels /workspace/knee/labels_confdist.parquet --bs 2 --grad_ckpt --k 24 --k_eval 24 \
  --epochs 24 --bb_lr 3e-5 --head_lr 1e-3 --workers 4 \
  --aug geo,erase,blur,noise --ema 0.9998 --tag SCRATCH_AUG > run_SCRATCH_AUG.log 2>&1
echo "SCRATCH_AUG done rc=\$? \$(date -u +%F\ %H:%M:%S)" >> ft_chain.log
EOS
chmod +x scratch_aug.sh
setsid nohup bash scratch_aug.sh > scratch_outer.log 2>&1 < /dev/null &
echo launched'
sleep 120
$SSH 'cd /workspace/knee; echo "trainers $(pgrep -f "[f]inetune_box.py" | wc -l)"; grep -E "init_arm|ema\]|train [0-9]+ \| gold|aug\]|grouped" run_SCRATCH_AUG.log | head -6' 2>/dev/null | grep -viE "welcome|have fun"
