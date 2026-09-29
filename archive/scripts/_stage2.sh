#!/bin/bash
# Unattended: wait for the augmentation fine-tune to finish ALL its epochs, take the control
# reading, then fine-tune each arm FURTHER on the studies that were withheld.
#
# Stage 1 holds out 350 scanner-disjoint studies so the run can be measured. Once measuring is done
# those studies are just unused training data. Dropping --group_csv puts them back: train_ids
# becomes every study except the 58 with real annotator labels, so stage 2 sees 100% of the usable
# corpus - and it validates on those 58, which is the SAME ruler the champion's 0.9266 was measured
# on, so for the first time the numbers are directly comparable.
#
# Runs strictly in sequence: nothing here competes with the training for GPU memory.
set -u
H=192.220.55.116; P=20566
SSH="ssh -i /c/Users/johna/.ssh/id_ed25519 -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 root@$H"

echo "$(date -u +%H:%M:%S) waiting for stage 1 to finish every epoch"
for i in $(seq 1 900); do
  d=$($SSH 'grep -c "FT CHAIN COMPLETE" /workspace/knee/ft_chain.log 2>/dev/null' 2>/dev/null)
  [ "${d:-0}" != "0" ] && { echo "$(date -u +%H:%M:%S) stage 1 complete"; break; }
  e=$($SSH 'grep -l "OutOfMemoryError\|Traceback" /workspace/knee/run_FT_*.log 2>/dev/null | tr "\n" " "' 2>/dev/null)
  if [ -n "${e// /}" ]; then echo "!! stage 1 CRASH in $e"; $SSH 'tail -12 /workspace/knee/run_FT_*.log' 2>/dev/null | tail -16; exit 2; fi
  sleep 120
done

$SSH 'cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ---- control: the SHIPPED arms on stage 1 ruler, learning switched off so they cannot change ----
CTL="--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --ckpt timm --norm imagenet \
--labels /workspace/knee/labels_confdist.parquet --bs 2 --grad_ckpt --k 24 --k_eval 24 \
--group_csv /workspace/knee/study_fingerprint.csv --group_folds 5 --group_fold 0 --reclaim 0.7 \
--epochs 1 --bb_lr 0 --head_lr 0 --workers 4 --limit 8"
python -u finetune_box.py $CTL --init_arm arms/raptor_ft_confdist_swa.pt --tag CTRL_CONFDIST > run_CTRL_CONFDIST.log 2>&1
python -u finetune_box.py $CTL --init_arm arms/raptor_ft_coatnet_w80_full_swa.pt --tag CTRL_W80 > run_CTRL_W80.log 2>&1
echo "CONTROL DONE $(date -u +%F\ %H:%M:%S)" >> ft_chain.log

# ---- stage 2: every study back in, including the 350 that were withheld ----
S2="--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --ckpt timm --norm imagenet \
--labels /workspace/knee/labels_confdist.parquet --bs 2 --grad_ckpt --k 24 --k_eval 24 \
--epochs 6 --bb_lr 1e-5 --head_lr 1e-4 --workers 4 --aug geo,erase,blur,noise"
cat > s2_chain.sh <<EOS
#!/bin/bash
set -u
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/knee
run () {
  echo "\$1 from \$2 start \$(date -u +%F\ %H:%M:%S)" >> ft_chain.log
  python -u finetune_box.py $S2 --init_arm /workspace/knee/\$2 --tag \$1 > run_\$1.log 2>&1
  echo "\$1 done rc=\$? \$(date -u +%F\ %H:%M:%S)" >> ft_chain.log
}
run FULL_CONFDIST raptor_ft_FT_CONFDIST.pt
run FULL_W80      raptor_ft_FT_W80.pt
echo "STAGE2 COMPLETE \$(date -u +%F\ %H:%M:%S)" >> ft_chain.log
EOS
chmod +x s2_chain.sh
setsid nohup bash s2_chain.sh > s2_outer.log 2>&1 < /dev/null &
echo launched-stage2'
sleep 60
echo "=== control readings (same 350 studies, same labels, as stage 1) ==="
$SSH 'cd /workspace/knee; for t in CTRL_CONFDIST CTRL_W80; do printf "%-14s %s\n" "$t" "$(grep -h "^ep" run_$t.log 2>/dev/null | tail -1)"; done
echo "--- stage 1 finals ---"; grep -H "^ep" run_FT_*.log | tail -4
echo "--- stage 2 running: $(pgrep -f "[f]inetune_box.py" | wc -l) ---"' 2>/dev/null | grep -viE "welcome|have fun"
