#!/bin/bash
# Launch the v11 training chain -- but ONLY after proving the corpus is real and nothing is running.
# The gate is inside this script deliberately; today I ran a preflight that said "2 running" and
# launched anyway because the check and the launch were separate shell commands.
set -u
cd /workspace/knee
D=/workspace/knee/corpus_v11

echo "=== 1. corpus CONTENT check (not just shape) ==="
python - <<'PYEOF' || exit 1
import numpy as np, os, sys
d = "/workspace/knee/corpus_v11"
for f in ("all_vols.npy", "all_masks.npy", "all_ids.npy"):
    if not os.path.exists(os.path.join(d, f)):
        print("MISSING %s -- merge has not finished" % f); sys.exit(1)
v = np.load(os.path.join(d, "all_vols.npy"), mmap_mode="r")
m = np.load(os.path.join(d, "all_masks.npy"))
ids = np.load(os.path.join(d, "all_ids.npy"), allow_pickle=True)
if v.shape != (4407, 96, 336, 336):
    print("BAD SHAPE %r" % (v.shape,)); sys.exit(1)
if len(ids) != 4407 or len(set(ids.astype(str))) != 4407:
    print("BAD IDS %d" % len(ids)); sys.exit(1)
samp = v[::100]
nz = (samp.reshape(len(samp), -1).sum(1) > 0).mean()
print("shape %r | ids ok | NONZERO fraction %.4f" % (v.shape, nz))
if nz < 0.99:
    print("ABORT: corpus is mostly zeros -- open_memmap allocates full size, so this is the real check")
    sys.exit(1)
val = m.mean(0); acc = 0
for s, b in enumerate([27, 21, 18, 12, 18]):
    print("  slot %d validity %.3f" % (s, val[acc:acc + b].mean())); acc += b
s3 = val[66:78].mean()
if not (0.55 <= s3 <= 0.92):
    print("WARNING slot 3 (Coronal non-fluid) validity %.3f outside 0.55-0.92" % s3)
print("CORPUS VERIFIED")
PYEOF

echo "=== 2. no training already running ==="
pgrep -af "[f]inetune_box.py" || echo "  none"
N=$(pgrep -cf "[f]inetune_box.py" 2>/dev/null | head -1); N=${N:-0}
if [ "$N" -ne 0 ]; then echo "ABORT: $N trainer(s) already running"; exit 1; fi

echo "=== 3. launching chain ==="
# 24 GB card -> bs 4 + grad_ckpt. Augmentation ON (the +0.015 lever that has NEVER shipped) and
# drop_path 0.2, at CORRECT 96-slice geometry. labels_confdist, NOT the E[s] set: label gold-AUC
# has been anti-predictive of arm quality 3/3, so E[s] is a later gated rung, not a headline.
cat > chain_v11.sh <<'EOS'
#!/bin/bash
set -u
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
cd /workspace/knee
C="--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --ckpt timm --norm imagenet \
--labels /workspace/knee/labels_confdist.parquet --bs 4 --grad_ckpt --k 24 --k_eval 24 \
--group_csv /workspace/knee/study_fingerprint.csv --group_folds 5 --group_fold 0 --reclaim 0.7 \
--epochs 16 --bb_lr 3e-5 --head_lr 1e-3 --workers 4 --aug geo,erase,blur,noise --drop_path 0.2"
echo "A start $(date -u +%H:%M:%S)" >> chain_v11.log
python -u finetune_box.py $C --seed 42   --tag V11_AUG_A > run_A.log 2>&1
echo "A done  $(date -u +%H:%M:%S)" >> chain_v11.log
python -u finetune_box.py $C --seed 4242 --tag V11_AUG_B > run_B.log 2>&1
echo "B done  $(date -u +%H:%M:%S)" >> chain_v11.log
EOS
chmod +x chain_v11.sh
setsid nohup bash chain_v11.sh > chain_outer.log 2>&1 < /dev/null &
disown
sleep 40
echo "=== trainers running: $(pgrep -cf '[f]inetune_box.py') (expect 1) ==="
head -25 run_A.log 2>/dev/null
