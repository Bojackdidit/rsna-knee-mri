#!/bin/bash
# Take FULL96 from "training finished" to "submitted", with no idle gap.
#
# FULL96 is the same recipe as LR96_CTRL but at full strength (no holdout, 14 epochs). It matters
# because SCRATCH_AUG2 - the CoAtNet half of the shipped champion - was trained on a rented box and
# its log never came back, so we ship a champion we cannot rebuild. This one is written down.
#
# The board decides, not the offline number: gold-58 predicted the wrong sign twice on 2026-09-24.
# The gold-58 figure here is only a sanity check that the arm trained at all.
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY=/d/claude2/conjecturer/.venv_cuda/Scripts/python.exe
DS=dreaddevelopment/raptor-knee-full96-arm
SLUG=knee-mri-coatnet-full96-plus-convnext
TITLE="Knee MRI CoAtNet Full96 Plus ConvNeXt"
CHAMP="C:/Users/johna/AppData/Local/Temp/claude/D--claude2/7cd517ed-06bc-44cd-94a6-0b69785d4cd0/scratchpad/champ"

running () { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'finetune_box' } | Measure-Object).Count" 2>/dev/null | tr -d '\r' | head -1; }
while [ "$(running)" != "0" ]; do sleep 120; done
sleep 30

if [ ! -f raptor_ft_FULL96.pt ]; then echo "ABORT - no raptor_ft_FULL96.pt"; exit 1; fi
echo "=== training done at $(date +%H:%M) ==="
grep -E '^ep[0-9]' _full96.log | tail -3

cp raptor_ft_FULL96.pt ship_full96/
echo "=== creating dataset ==="
kaggle datasets create -p ship_full96 -r zip 2>&1 | tail -2 || kaggle datasets version -p ship_full96 -m "full96" -r zip 2>&1 | tail -2

echo "=== waiting for dataset ready ==="
for i in $(seq 1 60); do
  s=$(kaggle datasets status $DS 2>&1 | head -1)
  echo "  [$(date +%H:%M:%S)] $s"
  case "$s" in *ready*) break;; esac
  sleep 30
done

echo "=== building variant (id_no dropped, title slugifies to id) ==="
$PY - <<PYEOF
import json,os,re,difflib
CH=r"$CHAMP"; OUT="variants/push_full96"; SLUG="$SLUG"; TITLE="$TITLE"
assert re.sub(r'[^a-z0-9]+','-',TITLE.lower()).strip('-')==SLUG, "title/id slug mismatch"
OLD='{"file": "raptor_ft_SCRATCH_AUG2.pt",  "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5}'
NEW='{"file": "raptor_ft_FULL96.pt",        "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5}'
nb=json.load(open(os.path.join(CH,'knee-mri-coatnet-plus-convnext.ipynb'),encoding='utf-8'))
code=lambda: '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
before=code(); hits=0
for c in nb['cells']:
    if c['cell_type']!='code': continue
    s=''.join(c['source'])
    if OLD in s:
        c['source']=s.replace(OLD,NEW).splitlines(keepends=True); hits+=1
assert hits==1, "anchor hits=%d"%hits
os.makedirs(OUT,exist_ok=True)
json.dump(nb, open(os.path.join(OUT,SLUG+'.ipynb'),'w',encoding='utf-8'))
m=json.load(open(os.path.join(CH,'kernel-metadata.json'),encoding='utf-8'))
m.pop('id_no',None)
m['id']='dreaddevelopment/'+SLUG; m['title']=TITLE; m['code_file']=SLUG+'.ipynb'
m['dataset_sources']=['$DS' if 'scratch-aug2' in d else d for d in m.get('dataset_sources',[])]
json.dump(m, open(os.path.join(OUT,'kernel-metadata.json'),'w',encoding='utf-8'), indent=1)
print('datasets:',m['dataset_sources'])
d=[l for l in difflib.unified_diff(before.split('\n'),code().split('\n'),lineterm='',n=0) if l.startswith(('+','-')) and not l.startswith(('+++','---'))]
print('diff lines:',len(d))
for l in d: print('   ',l[:110])
PYEOF

echo "=== pushing ==="
kaggle kernels push -p variants/push_full96 2>&1 | tail -2
sleep 10
echo "champion still resolves? $(kaggle kernels status dreaddevelopment/knee-mri-coatnet-plus-convnext 2>&1 | head -1)"

echo "=== waiting for kernel run ==="
for i in $(seq 1 80); do
  s=$(kaggle kernels status dreaddevelopment/$SLUG 2>&1 | head -1)
  echo "  [$(date +%H:%M:%S)] $s"
  case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "ABORT"; exit 1;; esac
  sleep 90
done

echo "=== submitting ==="
kaggle competitions submit -c rsna-knee-abnormality-detection -k dreaddevelopment/$SLUG -v 1 -f submission.csv \
  -m "The same pair as our best entry with the CoAtNet half replaced by a freshly trained one. The model it replaces was trained on a rented machine whose log we no longer have, so it could not be rebuilt; this one was trained here on the full set with a recipe we have written down, at the same resolution and the same slice geometry the scoring notebook uses. Nothing else about the entry changes." 2>&1 | tail -2

echo "=== polling score ==="
$PY -u _poll_sub.py $(kaggle competitions submissions rsna-knee-abnormality-detection --csv 2>/dev/null | sed -n 2p | cut -d, -f1)
