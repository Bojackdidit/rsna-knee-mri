"""Run the PATCHED notebook's own reader against the ORIGINAL notebook's own reader.

Both _make_reader cells are exec'd out of their notebooks and driven over a real series. This is
the shipped code, not a reimplementation, so agreement here is the thing that licenses the submit.
"""
import glob, io, json, os, sys, time
import numpy as np
import pydicom, cv2
from pydicom.pixel_data_handlers.util import apply_modality_lut
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "dcmprobe")
ORIG = os.path.join(HERE, "variants", "push_twogpu2", "knee-mri-dual-gpu-decode.ipynb")
PATCHED = os.path.join(HERE, "variants", "push_fastdcm", "knee-mri-raptor-direct-dicom-read.ipynb")
IMG, CROP_MM = 336, 140.0


def load_reader(path):
    nb = json.load(io.open(path, encoding="utf-8"))
    cells = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {"os": os, "glob": glob, "np": np, "pydicom": pydicom, "cv2": cv2,
          "apply_modality_lut": apply_modality_lut, "ThreadPoolExecutor": ThreadPoolExecutor,
          "IMG": IMG, "CROP_MM": CROP_MM, "time": time, "sys": sys}
    for src in cells:
        if "def parse_header(buf)" in src or "def _make_reader():" in src:
            exec(compile(src, "<nb>", "exec"), ns)
    return ns["_make_reader"]()


o_order, o_px, o_crop = load_reader(ORIG)
p_order, p_px, p_crop = load_reader(PATCHED)
print("both readers constructed")

a_files, a_ps = o_order(PROBE)
b_files, b_ps = p_order(PROBE)
print("order_and_meta files identical : %s  (%d slices)" % ([f for f, _ in a_files] == [f for f, _ in b_files], len(a_files)))
print("order_and_meta spacings identical: %s" % ([p for _, p in a_files] == [p for _, p in b_files]))
print("median pixel spacing           : %r vs %r  -> %s" % (a_ps, b_ps, a_ps == b_ps))
ok = ([f for f, _ in a_files] == [f for f, _ in b_files]) and ([p for _, p in a_files] == [p for _, p in b_files]) and a_ps == b_ps

bad = 0
for f, ps in a_files:
    xa, xb = o_px(f), p_px(f)
    if xa.shape != xb.shape or xa.dtype != xb.dtype or not np.array_equal(xa, xb):
        print("   read_px differs: %s" % os.path.basename(f)[-18:]); bad += 1
        continue
    ca, cb = o_crop(xa, ps), p_crop(xb, ps)
    if not np.array_equal(ca, cb):
        print("   crop differs: %s" % os.path.basename(f)[-18:]); bad += 1
print("read_px + crop identical       : %s  (%d slices checked)" % (bad == 0, len(a_files)))

if not ok or bad:
    print("\nRESULT: NOT IDENTICAL - do not submit"); sys.exit(1)


def bench(fn, reps=8):
    fn(); t = time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter() - t) / reps


def full(order, px, crop):
    fs, mps = order(PROBE)
    for f, ps in fs:
        crop(px(f), ps)


t_o = bench(lambda: full(o_order, o_px, o_crop))
t_p = bench(lambda: full(p_order, p_px, p_crop))
print("\nwhole-series read+crop  original %7.1f ms   patched %7.1f ms   %4.2fx" % (t_o*1e3, t_p*1e3, t_o/t_p))
print("RESULT: IDENTICAL OUTPUT, %.2fx faster on the read path" % (t_o/t_p))
