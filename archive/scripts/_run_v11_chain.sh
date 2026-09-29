#!/bin/bash
# Execute the KNEE-START-HERE spec on one box: assemble the 96-slice corpus ONCE, then chain runs.
# Levers in the order the spec ranks them by ARM-level evidence:
#   1. augmentation (+0.015, never once shipped)   2. 5-fold (+0.009, never used for an ensemble)
#   3. correct geometry (+0.005..0.013)            4. labels (gated rung, NOT a headline)
set -u
H=154.64.230.50; P=53473; K=/c/Users/johna/.ssh/id_ed25519
SSH="ssh -i $K -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 root@$H"
SCP="scp -i $K -P $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
echo "=== $(date -u +%H:%M:%S) stage ==="
$SSH 'mkdir -p /workspace/knee /root/.kaggle'
$SCP finetune_box.py _merge_v11.py train.csv study_fingerprint.csv root@$H:/workspace/knee/
$SCP phase2/box_stage/labeler/labels_confdist.parquet labels_expsev_cal.parquet root@$H:/workspace/knee/
$SCP /c/Users/johna/.kaggle/kaggle.json /c/Users/johna/.kaggle/access_token root@$H:/root/.kaggle/
$SSH 'chmod 600 /root/.kaggle/*; pip -q install kaggle timm pandas pyarrow scikit-learn 2>&1|tail -1'

echo "=== $(date -u +%H:%M:%S) corpus (96-slice, assemble ONCE) ==="
$SSH 'cd /workspace/knee && python -u _merge_v11.py --out /workspace/knee/corpus_v11 2>&1 | tail -25'
$SSH 'python -c "
import numpy as np
v=np.load(\"/workspace/knee/corpus_v11/all_vols.npy\",mmap_mode=\"r\")
assert v.shape==(4407,96,336,336), v.shape
print(\"CORPUS OK\",v.shape)"' || { echo "ABORT: corpus bad"; exit 1; }

# 24 GB card -> bs 4 + grad_ckpt. Memory says bs is not the constraint (bs16 0.8996 vs bs8 0.9032).
C="--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --ckpt timm --norm imagenet \
--labels /workspace/knee/labels_confdist.parquet --bs 4 --grad_ckpt --k 24 --k_eval 24 \
--group_csv /workspace/knee/study_fingerprint.csv --group_folds 5 --group_fold 0 --reclaim 0.7 \
--epochs 16 --bb_lr 3e-5 --head_lr 1e-3 --workers 4 --aug geo,erase,blur,noise --drop_path 0.2"

echo "=== $(date -u +%H:%M:%S) launching chained runs (nohup, survives my disconnect) ==="
$SSH "cd /workspace/knee && cat > chain.sh <<'EOS'
#!/bin/bash
set -u
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
cd /workspace/knee
echo \"A start \$(date -u +%H:%M:%S)\" >> chain.log
python -u finetune_box.py $C --seed 42   --tag V11_AUG_A > run_A.log 2>&1
echo \"A done  \$(date -u +%H:%M:%S)\" >> chain.log
python -u finetune_box.py $C --seed 4242 --tag V11_AUG_B > run_B.log 2>&1
echo \"B done  \$(date -u +%H:%M:%S)\" >> chain.log
EOS
chmod +x chain.sh && nohup ./chain.sh > chain_outer.log 2>&1 &
sleep 45; echo '--- run_A head ---'; head -20 run_A.log"
echo "=== $(date -u +%H:%M:%S) launched. poll: ssh ... 'tail -3 /workspace/knee/run_A.log' ==="
