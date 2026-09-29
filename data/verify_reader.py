"""Exact-contract check for the two champion-kernel functions being replaced by fastdcm."""
import glob, os, sys, time
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastdcm

FILES = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcmprobe", "*.dcm")))
print("files: %d" % len(FILES))

# ---------------- verbatim from the champion kernel ----------------
def hdr_one_ref(f):
    try:
        h = pydicom.dcmread(f, stop_before_pixels=True, force=True,
                            specific_tags=[(0x0020, 0x0032), (0x0020, 0x0037),
                                           (0x0028, 0x0030), (0x0020, 0x0013)])
        iop = getattr(h, 'ImageOrientationPatient', None)
        ipp = getattr(h, 'ImagePositionPatient', None)
        if iop is not None and ipp is not None and len(iop) == 6:
            r = np.array(iop[:3], float); c = np.array(iop[3:], float)
            n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
        else:
            pos = float(getattr(h, 'InstanceNumber', 0) or 0)
        ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5
        return (pos, f, ps, True)
    except Exception:
        return (0.0, f, 0.5, False)

def read_px_ref(f):
    d = pydicom.dcmread(f)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
        a = a.max() - a
    return a

# ---------------- fastdcm replacements ----------------
def hdr_one_fast(f):
    try:
        m = fastdcm.read_meta(f)
        if m is None:
            return hdr_one_ref(f)
        iop, ipp = m.iop, m.ipp
        if iop and ipp and len(iop) == 6:
            r = np.array(iop[:3], float); c = np.array(iop[3:], float)
            n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
        else:
            pos = float(m.inst or 0)
        ps = float(m.spacing[0]) if m.spacing else 0.5
        return (pos, f, ps, True)
    except Exception:
        return (0.0, f, 0.5, False)

def read_px_fast(f):
    m = fastdcm.read_meta(f)
    if m is None or m.has_modlut:
        return read_px_ref(f)
    a = fastdcm.read_pixels(m)
    if a is None:
        return read_px_ref(f)
    if m.has_rescale and not (m.slope == 1.0 and m.inter == 0.0):
        a = (a.astype(np.float64) * m.slope + m.inter).astype(np.float32)
    else:
        # slope 1 / intercept 0: uint16 -> float32 is exact (65535 < 2**24), same array
        a = a.astype(np.float32)
    if m.photo == 'MONOCHROME1':
        a = a.max() - a
    return a

bad = 0
for f in FILES:
    a, b = hdr_one_ref(f), hdr_one_fast(f)
    if a[0] != b[0] or a[2] != b[2] or a[3] != b[3]:
        print("  HDR MISMATCH %s: ref=%r fast=%r" % (os.path.basename(f)[-18:], a[:1]+a[2:], b[:1]+b[2:])); bad += 1
    pa, pb = read_px_ref(f), read_px_fast(f)
    if pa.shape != pb.shape or pa.dtype != pb.dtype or not np.array_equal(pa, pb):
        print("  PX MISMATCH %s" % os.path.basename(f)[-18:]); bad += 1
print("contract: %s" % ("EXACT MATCH on ordering key, pixel spacing and pixels" if bad == 0 else "%d MISMATCHES" % bad))
if bad: sys.exit(1)

# sort order over the whole series must be identical
oa = [f for _, f, _, _ in sorted([hdr_one_ref(f) for f in FILES], key=lambda x: x[0])]
ob = [f for _, f, _, _ in sorted([hdr_one_fast(f) for f in FILES], key=lambda x: x[0])]
print("series sort order identical: %s" % (oa == ob))

def bench(fn, reps=25):
    fn(); t = time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter() - t) / (reps * len(FILES))

h1 = bench(lambda: [hdr_one_ref(f) for f in FILES])
h2 = bench(lambda: [hdr_one_fast(f) for f in FILES])
p1 = bench(lambda: [read_px_ref(f) for f in FILES], 10)
p2 = bench(lambda: [read_px_fast(f) for f in FILES], 10)
print("\n_hdr_one  ref %7.3f ms  fast %7.3f ms  %4.1fx" % (h1*1e3, h2*1e3, h1/h2))
print("read_px   ref %7.3f ms  fast %7.3f ms  %4.1fx" % (p1*1e3, p2*1e3, p1/p2))
