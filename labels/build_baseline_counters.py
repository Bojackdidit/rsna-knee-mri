#!/usr/bin/env python3
"""Champion kernel, unchanged reader, with the SAME counters bolted on.

Gives a like-for-like Kaggle-side price for the pydicom reader on the same public studies the
fastdcm build measured, so the speedup can be read off Kaggle's own disk instead of my warm cache.
"""
import io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "variants", "push_twogpu2", "knee-mri-dual-gpu-decode.ipynb")
SLUG = "knee-mri-raptor-reader-baseline"
OUT = os.path.join(HERE, "variants", "push_baseline")

COUNTERS = '''
import atexit as _fd_atexit, threading as _fd_threading
from time import perf_counter as _fd_clock
_FD = {"meta_fast": 0, "meta_fb": 0, "px_fast": 0, "px_fb": 0, "t_meta": 0.0, "t_px": 0.0}
_FD_LOCK = _fd_threading.Lock()


def _fd_note(kind, fast, dt):
    with _FD_LOCK:
        _FD[kind + ("_fast" if fast else "_fb")] += 1
        _FD["t_" + kind] += dt


def _fd_report():
    print("[baseline] meta n=%d px n=%d | reader wall: meta %.1fs px %.1fs"
          % (_FD["meta_fb"], _FD["px_fb"], _FD["t_meta"], _FD["t_px"]), flush=True)
'''

nb = json.load(io.open(SRC, encoding="utf-8"))
t = next(i for i, c in enumerate(nb["cells"])
         if c["cell_type"] == "code" and "def _make_reader():" in "".join(c["source"]))
src = "".join(nb["cells"][t]["source"])

old_hdr_open = "    def _hdr_one(f):\n        \"\"\"One header -> (pos, file, pixel_spacing, ok). Identical arithmetic to the serial loop.\"\"\"\n        try:\n"
new_hdr_open = "    def _hdr_one(f):\n        \"\"\"One header -> (pos, file, pixel_spacing, ok). Identical arithmetic to the serial loop.\"\"\"\n        _t0 = _fd_clock()\n        try:\n"
assert old_hdr_open in src
src = src.replace(old_hdr_open, new_hdr_open, 1)
old_hdr_ret = "            ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5\n            return (pos, f, ps, True)"
new_hdr_ret = "            ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5\n            _fd_note('meta', False, _fd_clock() - _t0)\n            return (pos, f, ps, True)"
assert old_hdr_ret in src
src = src.replace(old_hdr_ret, new_hdr_ret, 1)

old_px = """    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a
"""
new_px = """    def read_px(f):
        _t0 = _fd_clock()
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        _fd_note('px', False, _fd_clock() - _t0)
        return a
"""
assert old_px in src
src = src.replace(old_px, new_px, 1)
nb["cells"][t]["source"] = src.splitlines(keepends=True)
nb["cells"].insert(t, {"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": COUNTERS.splitlines(keepends=True)})
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {},
                    "outputs": [], "source": ["_fd_report()\n"]})

os.makedirs(OUT, exist_ok=True)
json.dump(nb, io.open(os.path.join(OUT, SLUG + ".ipynb"), "w", encoding="utf-8"), ensure_ascii=False)
meta = json.load(io.open(os.path.join(HERE, "variants", "push_twogpu2", "kernel-metadata.json"), encoding="utf-8"))
meta["id"] = "dreaddevelopment/" + SLUG
meta["title"] = "Knee MRI - Raptor Reader Baseline"
meta["code_file"] = SLUG + ".ipynb"
meta.pop("id_no", None)
json.dump(meta, io.open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8"), indent=1)
print("built baseline counter kernel:", SLUG)
