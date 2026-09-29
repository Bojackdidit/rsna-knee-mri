#!/bin/bash
# Fires the moment the training chain exits, so the box never sits idle burning $0.478/hr.
# Scores every arm on the SAME fold and dumps predictions, then ranks solos and pairs on the
# common 350 studies - the only id set all three arms are honest on (E0 used --reclaim 0.7 and
# trained on the other 520 of fold 0; E1/E1b trained on none of fold 0).
set -u
cd /workspace/knee
export KNEE_DATA_ROOT=/workspace/knee/corpus_full
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[post] waiting for the training chain to finish $(date -u +%H:%M:%S)"
while pgrep -f "finetune_box.py" > /dev/null 2>&1; do sleep 30; done
echo "[post] chain done $(date -u +%H:%M:%S)"

# K must match what each arm was TRAINED at - an arm asked for a window count it never saw in
# training loses ~0.012 (measured), which would corrupt the comparison.
score () {  # tag  ckpt  k
  if [ -f "raptor_gold_$1.npz" ]; then echo "[post] $1 already scored"; return; fi
  if [ ! -f "$2" ]; then echo "[post] MISSING $2 - skipping $1"; return; fi
  echo "[post] scoring $1 (K=$3) $(date -u +%H:%M:%S)"
  GE_K="$3" python -u gate_eval.py "$2" "$1" 2>&1 | tail -4
}
score E0  raptor_ft_E0_b0288_fixed.pt   24
score E1  raptor_ft_E1_v2s288_3ser.pt   12
score E1b raptor_ft_E1b_v2s336_3ser.pt  12

echo
echo "================ RANKING ON THE COMMON 350 ================"
python -u pair_report.py --ids gate350_E0.csv . 2>&1 | tail -30
echo
echo "================ FOR REFERENCE: full 870 fold ============="
echo "(valid for E1 vs E1b only - E0 trained on 520 of these)"
python -u pair_report.py . 2>&1 | tail -20
echo "[post] DONE $(date -u +%H:%M:%S)"
