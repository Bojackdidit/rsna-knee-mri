#!/usr/bin/env python3
"""Overlap DICOM reading with GPU inference. Same predictions, turns a sum into a max.

Measured decomposition of the one-pass kernel (80 min): ~49 min DICOM, ~31 min GPU. The loop runs
them STRICTLY in series - read a study, infer it, read the next - so the GPU is idle during every
read and the disk is idle during every inference. They are different resources.

This prefetches the next studies' window tensors on a small thread pool while the GPU consumes the
current one, so the run costs about max(IO, GPU) instead of IO + GPU. With the threaded header
reads also in place (~25 min IO) that is ~max(25, 31) rather than ~56.

Predictions are unchanged and the reasoning is the same as the one-pass rewrite: every study is
prepared by the identical code path, each result is assigned to its own index, and the GPU still
consumes studies strictly in order 0..N-1. Only WHEN a study is prepared changes.

Memory: one window tensor is 48*3*384*384 float32 = 85 MB; PREFETCH=2 keeps at most 3 in flight.
"""
import ast, json, os, sys

SRC = "variants/push_fastread/knee-mri-parallel-header-reads.ipynb"
OUT_DIR = "variants/push_prefetch"
SLUG = "knee-mri-overlapped-reads"
TITLE = "Knee MRI - Overlapped Reads And Inference"

OLD = open("_looponepass.txt", encoding="utf-8").read()
NEW = '''    _PF = int(os.environ.get("ARM_PREFETCH", "2"))

    def _prep_study(sid):
        vol, mask = build_study(sid, SER, tsdir, reader)
        xw = eval_windows(vol, mask, k=K_EVAL, res=_SP_RES, norm=NORM)
        del vol, mask
        return xw

    # Producer/consumer: the pool prepares study i+1.._PF while the GPU consumes study i.
    # Consumption order is still strictly 0..N-1, so arm_probs is filled exactly as before.
    with _ArmPool2(max_workers=max(1, _PF)) as _pf_ex:
        _fut = {}
        for _j in range(min(_PF + 1, N)):
            _fut[_j] = _pf_ex.submit(_prep_study, test_ids[_j])
        for i, sid in enumerate(test_ids):
            try:
                xw = _fut.pop(i).result()
                for a, (_m, _r) in enumerate(_sp_models):
                    arm_probs[a][i] = infer_probs(_m, xw, dev)
                del xw
            except Exception as e:
                _fut.pop(i, None)
                print(f"  study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
            _nx = i + _PF + 1
            if _nx < N:
                _fut[_nx] = _pf_ex.submit(_prep_study, test_ids[_nx])
            if (i + 1) % 100 == 0 or i + 1 == N:
                print(f"  [1pass] {i+1}/{N} | {time.time()-t0:.0f}s", flush=True)
'''

IMP_OLD = "    reader = _make_reader()\n"
IMP_NEW = ("    from concurrent.futures import ThreadPoolExecutor as _ArmPool2\n"
           "    reader = _make_reader()\n")

EDITS = [(IMP_OLD, IMP_NEW), (OLD, NEW)]
nb = json.load(open(SRC, encoding="utf-8"))
counts = [0]*len(EDITS)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    for i,(old,new) in enumerate(EDITS):
        if old in j:
            j = j.replace(old,new,1); counts[i]+=1
    c["source"] = j.splitlines(keepends=True)
assert counts == [1]*len(EDITS), "edits matched %s" % counts
body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code")
assert "_prep_study" in body and body.count("_ArmPool2") == 2
assert "vol, mask = build_study(sid, SER, tsdir, reader)" in body
assert body.count("infer_probs(_m, xw, dev)") == 1
assert "_hdr_one" in body, "lost the threaded header reader"
assert "raptor_ft_coatnet_v11blend_full_swa.pt" not in body
for c in nb["cells"]:
    if c["cell_type"]=="code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations","import typing as _f"))
os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG+".ipynb"),"w",encoding="utf-8"))
meta=json.load(open("variants/push_fastread/kernel-metadata.json",encoding="utf-8"))
meta.update({"id":"dreaddevelopment/"+SLUG,"title":TITLE,"code_file":SLUG+".ipynb"})
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR,"kernel-metadata.json"),"w"), indent=1)
print("edits", counts, "| parses | private | reads overlapped with inference")
