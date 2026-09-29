#!/bin/bash
# Control arm for the backbone-LR experiment: CoAtNet-384, k=k_eval=12, grouped fold 0,
# blend labels, seed 42, 16 epochs -- IDENTICAL to lr1e4 (A100) and lr3e4 (4090) in every
# respect except bb_lr, which is the inherited 3e-5. This is the number that says whether
# the champion family has ever been trained at all, and the gate is valid for it because
# all three runs grade against the SAME label set.
set -eu
bash /workspace/knee/_setup_box.sh > /workspace/setup.log 2>&1
echo "SETUP DONE $(date -u +%H:%M:%S)" >> /workspace/chain.log
cd /workspace/knee
bash _run_lr.sh 3e-5 coat384_ref > /workspace/knee/run_coat384_ref.log 2>&1
echo "TRAIN DONE $(date -u +%H:%M:%S)" >> /workspace/chain.log
