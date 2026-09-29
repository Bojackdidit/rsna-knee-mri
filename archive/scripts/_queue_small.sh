#!/bin/bash
# When the CoAtNet run finishes, train what the top of the board actually runs.
#
# Scott (1st on both boards) gets LB 0.949 from a SINGLE EfficientNet/ResNet, single fold, ~5 min,
# scoring 0.930 on gold. Our two-model CoAtNet-384 ensemble scores HIGHER on gold (0.9331) and LOWER
# on the board (0.941). Tucker (13th) and tennogh (88th) both put the resolution sweet spot at
# 224-288 and Tucker adds "no attention mechanisms, just simple pooling operations".
#
# So this is the architecture hypothesis, not another tuning pass: small backbone, low resolution,
# MEAN pooling instead of the attention-MIL head, augmentation on, EMA to keep the weight averaging.
# Trains on all 4,349 studies and validates on the 58 real-labelled ones, the same ruler as always.
# ~4 min/epoch instead of 37, so the whole run costs about $0.40.
set -u
H=192.220.55.116; P=20566
SSH="ssh -i /c/Users/johna/.ssh/id_ed25519 -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 root@$H"
echo "$(date -u +%H:%M:%S) waiting for SCRATCH_AUG to finish all 24 epochs"
for i in $(seq 1 400); do
  d=$($SSH 'grep -c "SCRATCH_AUG done" /workspace/knee/ft_chain.log 2>/dev/null' 2>/dev/null)
  [ "${d:-0}" != "0" ] && { echo "$(date -u +%H:%M:%S) SCRATCH_AUG complete"; break; }
  sleep 120
done
$SSH 'cd /workspace/knee
cat > small_chain.sh <<EOS
#!/bin/bash
set -u
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/knee
C="--res 288 --ckpt timm --norm imagenet --labels /workspace/knee/labels_confdist.parquet \
--bs 8 --k 24 --k_eval 24 --epochs 24 --workers 4 --aug geo,erase,blur,noise --ema 0.9998 --pool mean"
run () {
  echo "\$1 start \$(date -u +%F\ %H:%M:%S)" >> ft_chain.log
  python -u finetune_box.py \$C --arch \$2 --bb_lr \$3 --head_lr 1e-3 --tag \$1 > run_\$1.log 2>&1
  echo "\$1 done rc=\$? \$(date -u +%F\ %H:%M:%S)" >> ft_chain.log
}
run SMALL_EFFB0 tf_efficientnet_b0.ns_jft_in1k 3e-4
run SMALL_RN50  resnet50.a1_in1k              3e-4
echo "SMALL CHAIN COMPLETE \$(date -u +%F\ %H:%M:%S)" >> ft_chain.log
EOS
chmod +x small_chain.sh
setsid nohup bash small_chain.sh > small_outer.log 2>&1 < /dev/null &
echo launched'
sleep 120
$SSH 'cd /workspace/knee; echo "trainers $(pgrep -f "[f]inetune_box.py" | wc -l)"; grep -E "pool\]|aug\]|ema\]|train [0-9]+ \| gold|raptor\] timm" run_SMALL_EFFB0.log 2>/dev/null | head -6' 2>/dev/null | grep -viE "welcome|have fun"
