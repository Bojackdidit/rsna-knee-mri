#!/bin/bash
# Corpus, then the two simple baselines an 11th-place competitor described.
set -u
bash /workspace/knee/_setup_box.sh > /workspace/setup.log 2>&1
echo "SETUP DONE $(date -u +%H:%M:%S)" >> /workspace/chain.log
cd /workspace/knee
bash _run_simple.sh resnet34.a1_in1k 224 mean rn34_224_mean > /workspace/knee/run_rn34_224_mean.log 2>&1
echo "RN34 DONE $(date -u +%H:%M:%S)" >> /workspace/chain.log
bash _run_simple.sh tf_efficientnet_b0.ns_jft_in1k 224 mean b0_224_mean > /workspace/knee/run_b0_224_mean.log 2>&1
echo "B0 DONE $(date -u +%H:%M:%S)" >> /workspace/chain.log
