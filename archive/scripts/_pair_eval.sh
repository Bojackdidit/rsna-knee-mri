#!/bin/bash
# Measure the INTERLEAVED pair exactly as the kernel serves it. Usage: _pair_eval.sh TAG_A TAG_B
#
# Two traps this avoids:
#  1. The arms are trained on DIFFERENT label sets, and a gate measures agreement with whatever
#     labels it is handed - so arm B's own training score is NOT comparable to arm A's. Both are
#     scored here against ONE label set (opus_blend) so the numbers mean the same thing.
#  2. The kernel does not give both arms the same windows: it plans 2K positions and hands arm 0
#     the even ones and arm 1 the odd. Scoring both on the same K positions measures a different
#     ensemble than the one that ships. GE_PARITY reproduces the real split.
set -u
A="$1"; B="$2"; K="${K:-12}"; IDS="${IDS:-valids_COAT384_FIX.csv}"
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
for spec in "$A 0" "$B 1"; do
  set -- $spec
  GE_IDS="$IDS" GE_K="$K" GE_SLOTS=0,2,4 GE_PARITY="$2" \
  GE_LABELS=/workspace/knee/labels_opus_blend.parquet \
    python -u gate_eval.py "raptor_ft_$1.pt" "PAIR_$1_p$2" 2>&1 | grep -E "INTERLEAVE|macro-AUC|config"
done
python - "$A" "$B" <<'PYEOF'
import sys, json, numpy as np
from sklearn.metrics import roc_auc_score
A,B=sys.argv[1],sys.argv[2]
def L(t,p):
    z=np.load("raptor_gold_PAIR_%s_p%d.npz"%(t,p),allow_pickle=True)
    return z["ids"].astype(str),z["pred"],z["truth"],json.loads(str(z["cfg"]))
ia,Pa,Y,ca=L(A,0); ib,Pb,_,cb=L(B,1)
assert list(ia)==list(ib), "different studies - not comparable"
assert ca["parity"]!=cb["parity"], "SAME parity - the interleave did not happen"
assert ca["labels"]==cb["labels"], "different label sets - scores not comparable"
rank=lambda x: x.argsort(0).argsort(0)/max(1,x.shape[0]-1)
def sc(P):
    return float(np.mean([roc_auc_score((Y[:,j]>=0.5).astype(int),P[:,j]) for j in range(Y.shape[1])
                          if len(set(((Y[:,j]>=0.5).astype(int)).tolist()))>1]))
sa,sb=sc(Pa),sc(Pb)
pair=sc((rank(Pa)+rank(Pb))/2)
c=float(np.mean([np.corrcoef(rank(Pa)[:,j],rank(Pb)[:,j])[0,1] for j in range(Y.shape[1])]))
print()
print("  graded against : %s   on %d studies" % (ca["labels"], len(ia)))
print("  %-22s %.4f  (even positions)" % (A,sa))
print("  %-22s %.4f  (odd positions)"  % (B,sb))
print("  INTERLEAVED PAIR       %.4f" % pair)
print("  gain over better solo  %+.4f" % (pair-max(sa,sb)))
print("  output correlation     %.4f   (lower = more diverse; cross-family runs ~0.89, siblings ~0.94)" % c)
PYEOF
