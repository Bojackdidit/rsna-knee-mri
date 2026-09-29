#!/usr/bin/env python3
"""Decode ONLY the slices the windows actually read. Usage: build_slicelimit.py K_PER_ARM [DILATE]

Reading is 3.7 of the champion's 23.8 minutes and I had been treating it as a fixed floor. It is not:
`build_study` decodes every slice of every slot (SLOTS sums to 96) and assembles the whole volume
BEFORE `eval_windows` chooses which slices to look at. At K=6 per arm interleaved, the two arms
between them read 12 centres x 3 slices = ~36 of ~88 decoded slices. The other ~52 decodes are thrown
away.

Plan first, then decode:
  1. work out which slots will be filled - that needs HEADERS only, which are already read;
  2. build the planned mask and run the SAME `_eval_centers` over it, so the centres are identical to
     what the unmodified kernel would have chosen;
  3. decode only the slices those centres touch (c-d, c, c+d).

⚠ THIS IS NOT A FREE CHANGE, and that is the point of measuring it. The 2/98 intensity normalisation
is computed PER SLOT from that slot's decoded slices, so decoding fewer shifts the percentiles and
therefore the pixel values. The commit run compares the output against the unmodified kernel on the
same studies so the cost is measured rather than assumed.
"""
import io, json, os, sys, ast, glob

K = int(sys.argv[1])
DIL = int(sys.argv[2]) if len(sys.argv) > 2 else 1
SRC = "variants/push_fastdcm"
SLUG = "knee-mri-coatnet-384-sl%dd%d" % (K, DIL)
TITLE = "Knee MRI - CoAtNet 384 SL%d D%d" % (K, DIL)
OUT = "variants/push_" + SLUG.replace("-", "_")

OLD_HEAD = """def build_study(sid, ser_records, tsdir, reader):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()
    for plane, fluid, k in SLOTS:"""

NEW_HEAD = """_OM_CACHE = {}


def _om_cached(order_and_meta, path):
    \"\"\"One header read per series. The planning pass and the decode pass both need the file list,
    and without this the plan DOUBLES header reads - measured 373 -> 746 on the commit run, which
    cancelled most of the pixel saving.\"\"\"
    v = _OM_CACHE.get(path)
    if v is None:
        v = order_and_meta(path)
        if len(_OM_CACHE) > 4096:
            _OM_CACHE.clear()
        _OM_CACHE[path] = v
    return v


def _plan_needed(sid, ser_records, tsdir, reader):
    \"\"\"Which GLOBAL slice indices will the windows actually read? Headers only, no pixel decode.\"\"\"
    order_and_meta, _rp, _mm = reader
    rows = ser_records.get(sid, [])
    filled, idx, used = [], 0, set()
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r['SeriesInstanceUID'])
        files, _ps = _om_cached(order_and_meta, f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k; continue
        for _ in range(k):
            if idx < MAXS: filled.append(idx)
            idx += 1
    m = np.zeros(MAXS, np.uint8)
    m[np.array(filled, int)] = 1 if filled else 0
    cs = _eval_centers(m, MAXS, K_EVAL)
    need = set()
    for c in cs:
        for o in (-_SL_DIL, 0, _SL_DIL):
            j = int(min(max(c + o, 0), MAXS - 1))
            need.add(j)
    return need, m


def build_study(sid, ser_records, tsdir, reader):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    _need, _planmask = _plan_needed(sid, ser_records, tsdir, reader)
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()
    for plane, fluid, k in SLOTS:"""

OLD_PX = """        def _px_one(_p):
            _fp, _ps = files[min(_p, n - 1)]
            try:
                return (read_px(_fp), _ps)
            except Exception:
                return (None, med_ps)"""
NEW_PX = """        _base = idx
        def _px_one(_t):
            _j, _p = _t
            if _j not in _need:                      # this slice is never read by any window
                return (None, med_ps)
            _fp, _ps = files[min(_p, n - 1)]
            try:
                return (read_px(_fp), _ps)
            except Exception:
                return (None, med_ps)"""

OLD_MAP = """        if len(picks) > 1:
            with _PxPool(max_workers=min(8, len(picks))) as _px_ex:
                _got = list(_px_ex.map(_px_one, list(picks)))
        else:
            _got = [_px_one(_p) for _p in picks]"""
NEW_MAP = """        _tasks = [(_base + _o, int(_p)) for _o, _p in enumerate(picks)]
        if len(picks) > 1:
            with _PxPool(max_workers=min(8, len(picks))) as _px_ex:
                _got = list(_px_ex.map(_px_one, _tasks))
        else:
            _got = [_px_one(_t) for _t in _tasks]"""

OLD_OM = "        files, med_ps = order_and_meta(f\"{tsdir}/{sid}/{r['SeriesInstanceUID']}\")"
NEW_OM = "        files, med_ps = _om_cached(order_and_meta, f\"{tsdir}/{sid}/{r['SeriesInstanceUID']}\")"
OLD_MASK = "    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)"
NEW_MASK = ("    # the PLANNED mask, not the decoded one: _eval_centers must reproduce the same centres\n"
            "    # the unmodified kernel would have chosen, or the windows would move.\n"
            "    mask = _planmask")

nb = json.load(io.open(glob.glob(os.path.join(SRC, "*.ipynb"))[0], encoding="utf-8"))
subs = [(OLD_HEAD, NEW_HEAD), (OLD_PX, NEW_PX), (OLD_MAP, NEW_MAP), (OLD_MASK, NEW_MASK),
        (OLD_OM, NEW_OM),
        ("K_EVAL = 48", "K_EVAL = %d\n_SL_DIL = %d" % (2 * K, DIL)),
        ("return _a, infer_probs(_mm, xw, _arm_devs[_a])",
         "return _a, infer_probs(_mm, xw[_a % 2::2], _arm_devs[_a])"),
        ("arm_probs[a][i] = infer_probs(_m, xw, _arm_devs[a])",
         "arm_probs[a][i] = infer_probs(_m, xw[a % 2::2], _arm_devs[a])")]
hits = [0] * len(subs)
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = o = "".join(c["source"])
    for i, (a, b) in enumerate(subs):
        if a in s:
            s = s.replace(a, b, 1); hits[i] += 1
    if s != o:
        c["source"] = s.splitlines(keepends=True)
print("hits:", hits)
if any(h != 1 for h in hits):
    sys.exit("anchors did not all match exactly once")
body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
for n, cond in [("planner added", "_plan_needed" in body),
                ("header cache", body.count("_om_cached(order_and_meta") == 3),
                ("need-gate in decode", "_j not in _need" in body),
                ("planned mask used", "mask = _planmask" in body),
                ("interleaved", "xw[_a % 2::2]" in body),
                ("K_EVAL", "K_EVAL = %d" % (2 * K) in body),
                ("champion arms", body.count("raptor_ft_confdist_swa.pt") == 1),
                ("fastdcm intact", "_fd_read_one" in body)]:
    print("  %-20s %s" % (n, "OK" if cond else "FAIL"))
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))
os.makedirs(OUT, exist_ok=True)
json.dump(nb, io.open(os.path.join(OUT, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(io.open(os.path.join(SRC, "kernel-metadata.json"), encoding="utf-8"))
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
meta.pop("id_no", None)
json.dump(meta, io.open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("built", OUT)
