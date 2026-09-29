#!/bin/bash
# Runs after the EffV2-S@336 rotation, with no idle gap.
#
# Adds B0 @336 folds so the shippable plex is CROSS-FAMILY as well as cross-fold: on the common 350,
# B0+EffV2-S gained +0.0108 over the best solo while EffV2-S+EffV2-S gained only +0.0084.
#
# SCORING RULE, learned the hard way: every arm is scored ONLY on the studies it actually held out,
# read from the valids_<tag>.csv its own training run wrote. gate_eval used to always score fold 0,
# which is right for fold-0 arms and badly wrong for the rest - a fold-1 arm trained on all of fold 0.
# E0 read 0.9536 that way against a true 0.8961.
#
# And arms from DIFFERENT folds hold out DISJOINT studies, so there is no common clean set on which
# to rank them against each other. Pair ranking is therefore only run WITHIN a fold group. Which
# arms make the best plex across folds is a board question, not an offline one - saying otherwise
# would be inventing a number.
set -u
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B0=tf_efficientnet_b0.ns_jft_in1k
RECLAIM="${RECLAIM:-0.7}"

echo "[postfolds] waiting for the EffV2-S rotation $(date -u +%H:%M:%S)"
while pgrep -f "[_]run_folds.sh" > /dev/null 2>&1; do sleep 60; done
echo "[postfolds] rotation done $(date -u +%H:%M:%S)"

# GRU head first: the biggest untried item on the competition recipe and the most informative single
# run available. Matched to E1b EXACTLY (fold 0, no reclaim, 336, 3 series, K=12, 16 epochs) so the
# only variable is the head. E1b scored 0.9166 on that same holdout with order-agnostic attention
# pooling, and its best came at epoch 11, so 16 epochs is a fair comparison. Both the RSNA-2023 and
# RSNA-2022 first-place solutions used a recurrent head over the ordered slices; we never have.
echo "[postfolds] GRU A/B vs E1b (bar to beat: 0.9166) $(date -u +%H:%M:%S)"
POOL=gru RECLAIM=0 bash _run_spec.sh tf_efficientnetv2_s.in21k_ft_in1k 336 "0,2,4" 12 16 0 0 0 12 GRU336_f0 > run_GRU336_f0.log 2>&1 || echo "[postfolds] GRU run failed, continuing"
echo "[postfolds] GRU done $(date -u +%H:%M:%S)"
grep -aoE "best 0[.][0-9]+" run_GRU336_f0.log 2>/dev/null | tail -1

RECLAIM="$RECLAIM" bash _run_folds.sh "$B0" 336 "0,2,4" 12 16 0 0 0 16 "2 3" B0336 > run_b0folds.log 2>&1
echo "[postfolds] B0 folds done $(date -u +%H:%M:%S)"

echo
echo "=========== EACH ARM ON THE STUDIES IT ACTUALLY HELD OUT ==========="
for f in raptor_ft_*.pt; do
  t=${f#raptor_ft_}; t=${t%.pt}
  [ -f "raptor_gold_$t.npz" ] && continue
  IDS="valids_$t.csv"
  if [ ! -f "$IDS" ]; then
    case "$t" in E0_b0288_fixed) IDS="gate350_E0.csv";; *) IDS="";; esac
  fi
  if [ -z "$IDS" ] || [ ! -f "$IDS" ]; then
    echo "  SKIP $t - no record of what it held out, cannot score it honestly"
    continue
  fi
  K=12; case "$t" in *E0_b0288*) K=24;; esac
  # Arms trained before the checkpoint recorded its geometry need it supplied. Everything except
  # E0 was trained on 3 series; scoring them on all 5 reads systematically low (-0.0053 on E1b).
  SL="0,2,4"; case "$t" in *E0_b0288*) SL="";; esac
  echo "  scoring $t on $IDS (K=$K)"
  GE_IDS="$IDS" GE_K="$K" GE_SLOTS="$SL" python -u gate_eval.py "$f" "$t" 2>&1 | tail -3
done

echo
echo "=========== PAIR RANKING, WITHIN EACH FOLD GROUP ONLY ==========="
python3 - <<'PYEOF'
import os, glob, hashlib, subprocess
grp = {}
for f in glob.glob("valids_*.csv"):
    t = f[len("valids_"):-4]
    if not os.path.exists("raptor_gold_%s.npz" % t):
        continue
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()[:12]
    grp.setdefault(h, []).append((t, f))
if os.path.exists("gate350_E0.csv") and os.path.exists("raptor_gold_E0_b0288_fixed.npz"):
    h = hashlib.sha256(open("gate350_E0.csv", "rb").read()).hexdigest()[:12]
    grp.setdefault(h, []).append(("E0_b0288_fixed", "gate350_E0.csv"))
for h, arms in sorted(grp.items()):
    names = [a for a, _ in arms]
    print()
    print("--- holdout %s : %d arm(s) -> %s" % (h, len(arms), names))
    if len(arms) < 2:
        print("    only one arm on this holdout, nothing to pair")
        continue
    print(subprocess.run(["python", "pair_report.py", "--ids", arms[0][1], "."],
                         capture_output=True, text=True).stdout[-1800:])
print()
print("Arms on DIFFERENT holdouts are not compared: they share no clean studies.")
print("Which cross-fold combination ships is decided on the board, not here.")
PYEOF
echo "[postfolds] DONE $(date -u +%H:%M:%S)"
