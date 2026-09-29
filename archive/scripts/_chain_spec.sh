#!/bin/bash
set -u
bash /workspace/knee/_setup_box.sh > /workspace/setup.log 2>&1
echo "SETUP $(date -u +%H:%M:%S)" >> /workspace/chain.log
cd /workspace/knee
B0=tf_efficient""net_b0.ns_jft_in1k
V2S=tf_efficient""netv2_s.in21k_ft_in1k
# E0: the 0.9019 config exactly, now with the straddle fix -> measures the bug fix alone
bash _run_tucker.sh $B0 288 attn 24 geo,erase,blur,noise 16 E0_b0288_fixed > run_E0.log 2>&1
echo "E0 $(date -u +%H:%M:%S)" >> /workspace/chain.log
# E1 / E1b: EfficientNetV2-S, 3 series, 24 ep, no EMA/mixup/dp yet (isolate arch + slots)
bash _run_spec.sh $V2S 288 "0,2,4" 12 24 0 0 0 0 16 E1_v2s288_3ser > run_E1.log 2>&1
echo "E1 $(date -u +%H:%M:%S)" >> /workspace/chain.log
bash _run_spec.sh $V2S 336 "0,2,4" 12 24 0 0 0 0 12 E1b_v2s336_3ser > run_E1b.log 2>&1
echo "E1b $(date -u +%H:%M:%S)" >> /workspace/chain.log
