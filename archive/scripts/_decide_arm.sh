#!/bin/bash
# One command: score a finished arm on the 58 gold studies, pull its gold file, and rank it as a
# partner at the weight the kernel actually ships. Turns a twenty-minute manual sequence into one
# step, so a finished run gets judged immediately instead of sitting while I assemble the pieces.
#
# Usage: _decide_arm.sh <TAG> <ARCH> <RES> <PORT> <IP>
set -u
TAG="${1:?tag}"; ARCH="${2:?arch}"; RES="${3:?res}"; PORT="${4:?port}"; IP="${5:?ip}"
K="-i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
cat > /tmp/_ev_$TAG.sh <<INNER
set -u
ulimit -n 65536
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u finetune_box.py --arch $ARCH --res $RES --ckpt timm --norm imagenet \
  --labels /workspace/knee/labels_confdist.parquet --bs 2 --grad_ckpt --k 24 --k_eval 24 \
  --epochs 1 --workers 2 --seed 4242 --ema 0.0 \
  --init_arm /workspace/knee/raptor_ft_$TAG.pt --eval_only --tag GE_$TAG 2>&1 | \
  grep -E "init_arm|eval_only|saved|rror" | head -4
INNER
scp -q $K -P "$PORT" /tmp/_ev_$TAG.sh "root@$IP:/workspace/knee/"
timeout 400 ssh $K -p "$PORT" "root@$IP" "cd /workspace/knee && bash _ev_$TAG.sh" 2>/dev/null | grep -vE "Welcome|Have fun"
scp -q $K -P "$PORT" "root@$IP:/workspace/knee/raptor_gold_GE_$TAG.npz" "gold_oof/raptor_gold_$TAG.npz"
python - "$TAG" <<'PY'
import glob, os, sys
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
NEW = sys.argv[1]; CH = "SCRATCH_AUG2"; W = 0.55
A = {os.path.basename(f)[12:-4]: np.load(f, allow_pickle=True) for f in glob.glob("gold_oof/raptor_gold_*.npz")}
ids = A[CH]["ids"]; A = {k: v for k, v in A.items() if np.array_equal(v["ids"], ids)}
truth = A[CH]["truth"]; n, nf = truth.shape; Y = (truth >= 0.5).astype(int)
R = {k: np.stack([rankdata(v["pred"][:, j]) / n for j in range(nf)], 1) for k, v in A.items()}
cols = [j for j in range(nf) if 0 < Y[:, j].sum() < n]
macro = lambda p: float(np.mean([roc_auc_score(Y[:, j], p[:, j]) for j in cols]))
bar = macro(W * R[CH] + (1 - W) * R["CONVNEXT_E24"])
solo = macro(A[NEW]["pred"])
rho = float(np.mean([spearmanr(R[NEW][:, j], R[CH][:, j]).statistic for j in range(nf)]))
pair = macro(W * R[CH] + (1 - W) * R[NEW])
print(f"\n{NEW}: solo {solo:.4f}  rho {rho:.4f}  pair@0.55 {pair:.4f}")
print(f"bar (champion + ConvNeXt) {bar:.4f}   difference {pair-bar:+.4f}")
d = pair - bar
print("VERDICT:", "SHIP IT" if d > 0.003 else
      ("worth a slot - inside what 58 studies can resolve" if d > -0.003 else "do not ship"))
PY
