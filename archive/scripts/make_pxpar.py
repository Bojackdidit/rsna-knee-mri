#!/usr/bin/env python3
"""Decode the picked slices in parallel. Same pixels, same order.

The 44-minute run is CPU-bound, not IO-wait-bound: threading the header reads only bought ~10%,
which means the remaining cost is real work on 4 vCPUs. Per study that work is roughly 366 header
parses plus 96 FULL dicom reads with pixel decompression, and the decode loop in build_study was
still serial:

    for p in picks: arrs.append(read_px(fp))

Unlike header parsing (pure Python, GIL-bound), pixel decompression happens in pydicom's C handlers
which release the GIL, so this one actually threads.

Ordering is preserved exactly: pool.map returns results in input order, so `arrs` and `pss` are
built in the same order the serial loop built them, and a failed decode still lands as None at its
own index just as before.
"""
import ast, json, os, sys

SRC = "variants/push_prefetch/knee-mri-overlapped-reads.ipynb"
OUT_DIR = "variants/push_pxpar"
SLUG = "knee-mri-parallel-decode"
TITLE = "Knee MRI - Parallel Decode"

OLD = open("_pxloop.txt", encoding="utf-8").read()
NEW = '''        from concurrent.futures import ThreadPoolExecutor as _PxPool
        def _px_one(_p):
            _fp, _ps = files[min(_p, n - 1)]
            try:
                return (read_px(_fp), _ps)
            except Exception:
                return (None, med_ps)
        if len(picks) > 1:
            with _PxPool(max_workers=min(8, len(picks))) as _px_ex:
                _got = list(_px_ex.map(_px_one, list(picks)))
        else:
            _got = [_px_one(_p) for _p in picks]
        arrs = [g[0] for g in _got]; pss = [g[1] for g in _got]
'''
EDITS = [(OLD, NEW)]
nb = json.load(open(SRC, encoding="utf-8"))
counts=[0]*len(EDITS)
for c in nb["cells"]:
    if c["cell_type"]!="code": continue
    j="".join(c["source"])
    for i,(o,n) in enumerate(EDITS):
        if o in j: j=j.replace(o,n,1); counts[i]+=1
    c["source"]=j.splitlines(keepends=True)
assert counts==[1]*len(EDITS), "edits matched %s"%counts
body="\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
assert "_px_one" in body and "for p in picks:" not in body, "serial decode survived"
assert "_PxPool" in body and body.count("_PxPool") == 2
assert "_hdr_one" in body and "_prep_study" in body, "lost an earlier optimisation"
assert "raptor_ft_coatnet_v11blend_full_swa.pt" not in body
for c in nb["cells"]:
    if c["cell_type"]=="code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations","import typing as _f"))
os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG+".ipynb"),"w",encoding="utf-8"))
meta=json.load(open("variants/push_prefetch/kernel-metadata.json",encoding="utf-8"))
meta.update({"id":"dreaddevelopment/"+SLUG,"title":TITLE,"code_file":SLUG+".ipynb"})
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR,"kernel-metadata.json"),"w"), indent=1)
print("edits", counts, "| parses | private | pixel decode threaded, order preserved")
