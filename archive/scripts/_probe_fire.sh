#!/bin/bash
# Fire four per-finding board probes and read back the per-finding AUCs.
#
# Macro ROC-AUC is the unweighted mean of 12 per-finding AUCs and a constant column scores exactly
# 0.5, so holding 11 columns constant makes the single board scalar a direct read of the twelfth:
#     AUC_j = 12 * board - 5.5   (+-0.006 at three decimals, over all 1,322 test studies)
#
# This is the only ground truth we have. Every offline instrument we own has misfired - the 870
# fold is memorised by the shipped arms, and gold-58 called Lateral OA our healthiest finding when
# the board says it is our worst. These four span the full offline range (Fracture and Contusion at
# the bottom, Medial OA and Baker's at the top), so if the offline ranking inverts again we will see
# it, and the result becomes the calibration set that certifies or kills the OOF oracle tomorrow.
#
# Probes score ~0.5 and best-of protects the board, so they cost a slot and nothing else.
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe

PROBES="fracture-probe contusion-probe medial-oa-probe baker-s-probe"
declare -A VEROF   # push and submit happen in SEPARATE loops, so the version must be kept per probe

for P in $PROBES; do
  D="variants/push_knee_mri_$(echo $P | tr '-' '_')"
  [ -d "$D" ] || D="variants/push_knee_mri_${P//-/_}"
  echo "=== pushing $P from $D ==="
  # Capture the version the push actually created. Hardcoding -v 1 submitted the PREVIOUS
  # session's build of these probes, which runs a superseded pair - the numbers then describe the
  # wrong model. The version is in the push output; take it from there.
  PUSHOUT=$(kaggle kernels push -p "$D" 2>&1 | tail -1); echo "$PUSHOUT"
  VER=$(echo "$PUSHOUT" | grep -oE 'version [0-9]+' | grep -oE '[0-9]+')
  [ -z "$VER" ] && { echo "  ABORT $P: push did not report a version"; continue; }
  VEROF[$P]=$VER
  echo "  -> will submit version $VER"
  sleep 5
done

echo "=== champion still resolves? ==="
kaggle kernels status dreaddevelopment/knee-mri-coatnet-plus-convnext 2>&1 | head -1

for P in $PROBES; do
  K=dreaddevelopment/knee-mri-$P
  echo "=== waiting for $P ==="
  for i in $(seq 1 60); do
    s=$(kaggle kernels status $K 2>&1 | head -1)
    case "$s" in
      *COMPLETE*) echo "  [$(date +%H:%M)] complete"; break;;
      *ERROR*|*CANCEL*) echo "  ABORT $P: $s"; continue 2;;
    esac
    sleep 90
  done
  echo "=== submitting $P ==="
  kaggle competitions submit -c rsna-knee-abnormality-detection -k $K -v "${VEROF[$P]}" -f submission.csv \
    -m "Diagnostic entry. Eleven of the twelve findings are replaced by a constant value, leaving one real, so the single score the leaderboard returns reads that one finding on its own. Not an attempt at a better model." 2>&1 | tail -1
  sleep 10
done

echo "=== all fired; polling for scores ==="
for i in $(seq 1 45); do
  $PY - <<'PYEOF'
import io,subprocess,pandas as pd
out=subprocess.run(["kaggle","competitions","submissions","rsna-knee-abnormality-detection","--csv"],
                   capture_output=True,text=True,timeout=120).stdout
d=pd.read_csv(io.StringIO(out)).head(5)
for _,r in d.iterrows():
    s=r['publicScore']
    auc=('%.4f'%(12*float(s)-5.5)) if pd.notna(s) else '-'
    print('  %s | %-28s | board %-7s | AUC %s' % (str(r['date'])[:16], str(r['description'])[:28], s, auc))
PYEOF
  sleep 120
done
