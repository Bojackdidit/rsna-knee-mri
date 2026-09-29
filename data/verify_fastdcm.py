"""Prove fastdcm returns exactly what pydicom returns, then time both paths."""
import glob, os, sys, time
import numpy as np
import pydicom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastdcm

FILES = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcmprobe", "*.dcm")))
ORDER_TAGS = ['ImagePositionPatient', 'InstanceNumber']
print("files: %d" % len(FILES))

# ---------- correctness ----------
bad = 0
for f in FILES:
    ds = pydicom.dcmread(f)
    m = fastdcm.read_meta(f)
    assert m is not None, "fastdcm declined %s" % f
    checks = [
        ("Rows", int(ds.Rows), m.rows),
        ("Cols", int(ds.Columns), m.cols),
        ("Bits", int(ds.BitsAllocated), m.bits),
        ("PixRepr", int(ds.PixelRepresentation), m.pixrepr),
        ("Inst", int(ds.InstanceNumber), m.inst),
        ("Slope", float(ds.RescaleSlope), float(m.slope)),
        ("Inter", float(ds.RescaleIntercept), float(m.inter)),
    ]
    for name, a, b in checks:
        if a != b:
            print("  MISMATCH %s: pydicom=%r fast=%r (%s)" % (name, a, b, os.path.basename(f)))
            bad += 1
    ipp_p = [float(x) for x in ds.ImagePositionPatient]
    if not np.allclose(ipp_p, m.ipp, rtol=0, atol=0):
        print("  MISMATCH IPP: %r vs %r" % (ipp_p, m.ipp)); bad += 1
    psp_p = [float(x) for x in ds.PixelSpacing]
    if not np.allclose(psp_p, m.spacing, rtol=0, atol=0):
        print("  MISMATCH PixelSpacing: %r vs %r" % (psp_p, m.spacing)); bad += 1
    a_py = ds.pixel_array
    a_fast = fastdcm.read_pixels(m)
    if a_fast is None:
        print("  fastdcm declined pixels: %s" % os.path.basename(f)); bad += 1
    elif a_py.shape != a_fast.shape or not np.array_equal(a_py, a_fast):
        print("  MISMATCH PIXELS %s" % os.path.basename(f)); bad += 1
print("correctness: %s" % ("ALL IDENTICAL" if bad == 0 else "%d MISMATCHES" % bad))
if bad:
    sys.exit(1)

# ---------- timing ----------
def bench(fn, reps):
    fn()  # warm
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / (reps * len(FILES))

R = 60
def py_hdr():
    for f in FILES:
        ds = pydicom.dcmread(f, force=True, stop_before_pixels=True, specific_tags=ORDER_TAGS)
        _ = getattr(ds, 'ImagePositionPatient', None), getattr(ds, 'InstanceNumber', None)

def fast_hdr():
    for f in FILES:
        m = fastdcm.read_meta(f)
        _ = m.ipp, m.inst

def py_px():
    for f in FILES:
        ds = pydicom.dcmread(f, force=True)
        _ = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

def fast_px():
    for f in FILES:
        m = fastdcm.read_meta(f)
        _ = fastdcm.read_pixels(m).astype(np.float32) * m.slope + m.inter

a = bench(py_hdr, R); b = bench(fast_hdr, R)
print("\nheader read   pydicom(specific_tags) %8.3f ms/slice   fastdcm %8.3f ms/slice   speedup %5.1fx" % (a*1e3, b*1e3, a/b))
c = bench(py_px, 20); d = bench(fast_px, 20)
print("pixel  read   pydicom.pixel_array   %8.3f ms/slice   fastdcm %8.3f ms/slice   speedup %5.1fx" % (c*1e3, d*1e3, c/d))
print("\n(warm page cache: this isolates parse/alloc overhead, not disk bandwidth)")
