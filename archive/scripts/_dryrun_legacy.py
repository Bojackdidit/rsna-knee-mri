"""Execute BOTH legacy loaders locally against the real downloaded bundles.

Catches logic errors before a run is spent: member count, unique ids, per-target weights, the
pixel-group key, and that the family's total share of each finding is identical between the
one-run and four-run variants.
"""
import json, os, re, sys
from pathlib import Path
import torch

TARGETS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
           'Effusion','Synovitis',"Baker's",'Contusion','Fracture']
SLOTS = [('SAG_FLUID_FS','Sagittal',True,True),('COR_FLUID_FS','Coronal',True,True),
         ('AX_FLUID_FS','Axial',True,True),('SAG_FLUID_NOFS','Sagittal',True,False),
         ('COR_T1','Coronal',False,False),('SAG_T1','Sagittal',False,False)]
RULES_LEGACY = {'order':'dominant_axis','lat':'corner_x','slot_fallback':True,'decode_fill':'zero'}
LEGACY_MEMBER_WEIGHT_BY_TARGET = {'Lateral Meniscus':15.0,'Medial OA':2.5,'Lateral OA':15.0,'Contusion':5.0}
LEGACY_WEIGHT = 0.5
LEGACY_BUNDLE_FILE = 'rsna_20260807_v1.pt'
BASE = Path('legacy_probe')
def log(m): print('   ', m)

def extract(nb_path, fn_names):
    src = ''.join(''.join(c['source']) for c in json.load(open(nb_path, encoding='utf-8'))['cells']
                  if c['cell_type'] == 'code')
    out = []
    for name in fn_names:
        i = src.index('def %s(' % name)
        j = src.index('\ndef ', i + 1)
        out.append(src[i:j])
    return '\n\n'.join(out)

def run(nb, fns, label):
    g = dict(globals()); g['Path'] = Path
    code = extract(nb, fns).replace("Path('/kaggle/input')", "BASE")
    exec(code, g)
    ms_by_key = g['legacy_group_members']()
    assert len(ms_by_key) == 1, f'{label}: expected one pixel group, got {len(ms_by_key)}'
    key, ms = next(iter(ms_by_key.items()))
    ids = [m['id'] for m in ms]
    assert len(set(ids)) == len(ids), f'{label}: duplicate member ids'
    tw = {t: sum(m['target_weight'][i] for m in ms) for i, t in enumerate(TARGETS)}
    print(f'\n=== {label}')
    print(f'    members: {len(ms)}   ids: {ids[:4]}{" ..." if len(ids) > 4 else ""}')
    print(f'    variant: {ms[0]["config"]["variant"]}  unfreeze_last={ms[0]["config"]["unfreeze_last"]}  pool={ms[0]["config"]["pool"]}')
    print(f'    geometry key: {json.loads(key)["img"]}px, {json.loads(key)["slices"]} slices, '
          f'crop {json.loads(key)["crop_mm"]}mm, band {json.loads(key)["band"]}')
    print(f'    FAMILY TOTAL weight per finding: ' + ', '.join(f'{t}={v:g}' for t, v in tw.items() if v))
    print(f'    every state_dict present: {all("state" in m and m["state"] for m in ms)}')
    return len(ms), tw

n1, w1 = run('variants/push_legacy/knee-mri-restored-second-model-family.ipynb',
             ['find_legacy_bundle', 'legacy_group_members'], 'ONE RUN (v1 only)')
n4, w4 = run('variants/push_legacy4/knee-mri-second-family-four-runs.ipynb',
             ['find_legacy_bundles', 'legacy_group_members'], 'FOUR RUNS (v1-v4)')

print('\n=== the property that matters')
print(f'    members: {n1} -> {n4}  ({n4//n1}x the models)')
same = all(abs(w1[t] - w4[t]) < 1e-9 for t in TARGETS)
print(f'    family TOTAL share per finding identical between the two: {same}')
assert same, 'four-run variant changed the family share -- that would confound the experiment'
print('    -> the four-run variant is a better ESTIMATE of the same opinion, not a louder vote')
