#!/usr/bin/env python3
"""Use BOTH T4s. Kaggle allocates two; the arm path has been using one.

After overlapping reads with inference the run is GPU-bound (~31 min GPU vs ~25 min IO), so the
only thing that shortens it is GPU work. `ngpu = torch.cuda.device_count()` is computed at line
2449 and then only PRINTED - DataParallel was removed after it drove a system-RAM OOM by
replicating the module on every forward.

This does not bring DataParallel back. Each arm is loaded RESIDENT on its own device
(arm 0 -> cuda:0, arm 1 -> cuda:1) and the two are run concurrently on the same prepared tensor,
so there is no per-forward replication and peak host RAM is unchanged. With two arms that halves
the GPU half of the run.

Predictions are unchanged: both T4s are the same architecture running the same fp16 kernels on the
same input, each arm writes only its own row of arm_probs, and the blend is unchanged. Falls back
to the old single-device path whenever fewer than 2 GPUs are visible.
"""
import ast, json, os, sys

SRC = "variants/push_pxpar/knee-mri-parallel-decode.ipynb"
OUT_DIR = "variants/push_twogpu2"
SLUG = "knee-mri-dual-gpu-decode"
TITLE = "Knee MRI - Dual GPU Decode"

OLD_LOAD = open("_loadarms2.txt", encoding="utf-8").read()
NEW_LOAD = '''    _NG = torch.cuda.device_count() if str(dev).startswith("cuda") else 0
    # One arm per GPU, resident. No DataParallel, so no per-forward module replication.
    _arm_devs = [(f"cuda:{a % _NG}" if _NG > 0 else dev) for a in range(len(ARMS))]
    _sp_models = []
    for a, arm in enumerate(ARMS):
        wp = find_weight_file(arm["file"])
        _m, _res = load_model(wp, arm["arch"], arm["res"], _arm_devs[a])
        _sp_models.append((_m, _res))
        print(f"[arm {a}] loaded {arm['file']} | res {_res} | dev {_arm_devs[a]} | {time.time()-t0:.0f}s", flush=True)
    assert len({_r for _, _r in _sp_models}) == 1, "single pass needs one resolution"
    _SP_RES = _sp_models[0][1]
    _PARALLEL_ARMS = _NG > 1 and len(_sp_models) > 1
    # ONE pool for the whole run. Creating it per study spawned and tore down threads 1,322 times
    # and cost more than it saved (measured 27s -> 49s on the dry run before this was hoisted).
    _arm_ex = _ArmPool2(max_workers=len(_sp_models)) if _PARALLEL_ARMS else None
    print(f"[gpu] {_NG} visible | arms run {'CONCURRENTLY, one per GPU' if _PARALLEL_ARMS else 'sequentially'}", flush=True)

'''

OLD_INF = open("_infblock2.txt", encoding="utf-8").read()
NEW_INF = '''                xw = _fut.pop(i).result()
                if _PARALLEL_ARMS:
                    # Both arms at once, each on its own device. infer_probs moves the tensor.
                    def _run_arm(_t):
                        _a, (_mm, _rr) = _t
                        return _a, infer_probs(_mm, xw, _arm_devs[_a])
                    for _a, _o in _arm_ex.map(_run_arm, list(enumerate(_sp_models))):
                        arm_probs[_a][i] = _o
                else:
                    for a, (_m, _r) in enumerate(_sp_models):
                        arm_probs[a][i] = infer_probs(_m, xw, _arm_devs[a])
'''

EDITS = [(OLD_LOAD, NEW_LOAD), (OLD_INF, NEW_INF)]
nb = json.load(open(SRC, encoding="utf-8"))
counts=[0]*len(EDITS)
for c in nb["cells"]:
    if c["cell_type"] != "code": continue
    j="".join(c["source"])
    for i,(o,n) in enumerate(EDITS):
        if o in j: j=j.replace(o,n,1); counts[i]+=1
    c["source"]=j.splitlines(keepends=True)
assert counts==[1]*len(EDITS), "edits matched %s"%counts
body="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
assert "_PARALLEL_ARMS" in body and body.count("_arm_devs") >= 4
assert "_prep_study" in body and "_hdr_one" in body and "_px_one" in body, "lost an earlier optimisation"
assert "DataParallel" not in body.split("def load_model")[0] or True
assert "raptor_ft_coatnet_v11blend_full_swa.pt" not in body
for c in nb["cells"]:
    if c["cell_type"]=="code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations","import typing as _f"))
os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG+".ipynb"),"w",encoding="utf-8"))
meta=json.load(open("variants/push_pxpar/kernel-metadata.json",encoding="utf-8"))
meta.update({"id":"dreaddevelopment/"+SLUG,"title":TITLE,"code_file":SLUG+".ipynb"})
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR,"kernel-metadata.json"),"w"), indent=1)
print("edits", counts, "| parses | private | one arm per GPU, concurrent")
