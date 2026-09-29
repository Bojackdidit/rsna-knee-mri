#!/bin/bash
# When SCRATCH_AUG finishes all 24 epochs, run the label-noise variant.
#
# IDENTICAL to SCRATCH_AUG in every respect - same backbone, resolution, labels, LR, epochs,
# augmentation, EMA, all 4,349 studies, scored on the same 58 real-labelled ones - except:
#   --cell_mask   do not compute loss on cells the radiology report never ADDRESSED
#   --loss macro_bce  equalise the per-finding gradient over the cells that survive
# so the comparison against SCRATCH_AUG's 0.9260 is a clean single-variable read.
#
# Why these two together rather than separately: masking alone leaves Synovitis with 678 surviving
# cells and therefore a tiny gradient; macro_bce alone amplifies the 84% of Synovitis cells that
# carry no information. Each fixes what the other breaks. Scott, 1st on both boards, has worse
# labels than ours (0.89 vs 0.8945) and says the whole of his effort went into working AROUND label
# noise - and our three weakest board findings are the three most report-silent ones (0.878 against
# 0.928 for the clean-label findings).
set -u
H=192.220.55.116; P=20566
SSH="ssh -i /c/Users/johna/.ssh/id_ed25519 -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 root@$H"
echo "$(date -u +%H:%M:%S) waiting for SCRATCH_AUG to finish"
for i in $(seq 1 400); do
  d=$($SSH 'grep -c "SCRATCH_AUG done" /workspace/knee/ft_chain.log 2>/dev/null' 2>/dev/null)
  [ "${d:-0}" != "0" ] && { echo "$(date -u +%H:%M:%S) SCRATCH_AUG complete"; break; }
  sleep 120
done
$SSH 'cd /workspace/knee
cat > masked_chain.sh <<EOS
#!/bin/bash
set -u
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/knee
echo "MASKED start \$(date -u +%F\ %H:%M:%S)" >> ft_chain.log
python -u finetune_box.py \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --ckpt timm --norm imagenet \
  --labels /workspace/knee/labels_confdist.parquet --bs 2 --grad_ckpt --k 24 --k_eval 24 \
  --epochs 24 --bb_lr 3e-5 --head_lr 1e-3 --workers 4 \
  --aug geo,erase,blur,noise --ema 0.9998 \
  --cell_mask /workspace/knee/cellmask_unk.parquet --loss macro_bce \
  --tag MASKED > run_MASKED.log 2>&1
echo "MASKED done rc=\$? \$(date -u +%F\ %H:%M:%S)" >> ft_chain.log
EOS
chmod +x masked_chain.sh
setsid nohup bash masked_chain.sh > masked_outer.log 2>&1 < /dev/null &
echo launched'
sleep 150
$SSH 'cd /workspace/knee; echo "trainers $(pgrep -f "[f]inetune_box.py" | wc -l)"; grep -E "cell_mask|loss\]|train [0-9]+ \| gold|ema\]|aug\]" run_MASKED.log 2>/dev/null | head -6; tail -2 run_MASKED.log' 2>/dev/null | grep -viE "welcome|have fun"
