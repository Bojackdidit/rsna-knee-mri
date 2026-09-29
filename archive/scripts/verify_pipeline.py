"""End-to-end identity check: the kernel's ordering + slice-reading, pydicom vs fastdcm.

Replicates _order_dominant_axis and the plane loop of read_slot exactly as they appear in
champ/_src.py, once per reader, and asserts the ordered file list and the stacked volume are
identical. This is the check that matters - a per-slice match does not prove the SELECTION
of slices is unchanged, and selection is what would move the score.
"""
import glob, os, sys, time
import numpy as np
import pydicom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastdcm

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcmprobe")
FILES = sorted(os.path.basename(p) for p in glob.glob(os.path.join(D, "*.dcm")))
GROUP, SLICE_BAND = 3, (0.2, 0.8)
print("series files: %d" % len(FILES))


def order_pydicom(files, d):
    rows = []
    for pos, f in enumerate(files):
        ipp = inst = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True,
                                 specific_tags=['ImagePositionPatient', 'InstanceNumber'])
            raw = getattr(ds, 'ImagePositionPatient', None)
            if raw is not None and len(raw) >= 3:
                c = np.asarray(raw[:3], dtype=np.float64)
                if np.isfinite(c).all():
                    ipp = c
            n = getattr(ds, 'InstanceNumber', None)
            if n is not None:
                inst = float(n)
        except Exception:
            pass
        rows.append((f, ipp, inst, pos))
    return _finish(rows)


def order_fast(files, d):
    rows = []
    for pos, f in enumerate(files):
        ipp = inst = None
        try:
            m = fastdcm.read_meta(os.path.join(d, f))
            if m is None:
                raise ValueError
            if m.ipp and len(m.ipp) >= 3:
                c = np.asarray(m.ipp[:3], dtype=np.float64)
                if np.isfinite(c).all():
                    ipp = c
            if m.inst is not None:
                inst = float(m.inst)
        except Exception:
            pass
        rows.append((f, ipp, inst, pos))
    return _finish(rows)


def _finish(rows):
    placed = [r for r in rows if r[1] is not None]
    need = max(2, int(0.8 * len(rows)))
    if len(placed) >= need:
        xyz = np.stack([r[1] for r in placed])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        rows.sort(key=lambda r: (float(r[1][axis]) if r[1] is not None else spare,
                                 r[2] if r[2] is not None else float('inf'), r[3]))
    elif sum((r[2] is not None for r in rows)) >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float('inf'), r[3]))
    return [r[0] for r in rows]


def planes_pydicom(ordered, d, idx):
    out = []
    for i in idx:
        ds = pydicom.dcmread(os.path.join(d, ordered[int(i)]), force=True)
        a = ds.pixel_array.astype(np.float32)
        a = a * float(getattr(ds, 'RescaleSlope', 1) or 1) + float(getattr(ds, 'RescaleIntercept', 0) or 0)
        out.append(a)
    return np.stack(out)


def planes_fast(ordered, d, idx):
    out = []
    for i in idx:
        m = fastdcm.read_meta(os.path.join(d, ordered[int(i)]))
        a = fastdcm.read_pixels(m)
        if a is None:
            ds = pydicom.dcmread(os.path.join(d, ordered[int(i)]), force=True)
            a = ds.pixel_array
            m_slope, m_inter = float(getattr(ds, 'RescaleSlope', 1) or 1), float(getattr(ds, 'RescaleIntercept', 0) or 0)
        else:
            m_slope, m_inter = m.slope, m.inter
        out.append(a.astype(np.float32) * m_slope + m_inter)
    return np.stack(out)


oa = order_pydicom(FILES, D)
ob = order_fast(FILES, D)
print("ordering identical : %s" % (oa == ob))
if oa != ob:
    for x, y in zip(oa, ob):
        if x != y:
            print("   first diff: %s vs %s" % (x[-24:], y[-24:])); break
    sys.exit(1)

n = len(oa)
lo, hi = int(SLICE_BAND[0] * (n - 1)), int(SLICE_BAND[1] * (n - 1))
idx = np.unique(np.linspace(lo, hi, GROUP).astype(int)) if hi > lo else np.array([n // 2])
print("selected indices   : %s  of %d" % (list(idx), n))

va = planes_pydicom(oa, D, idx)
vb = planes_fast(ob, D, idx)
print("volume shape       : %s" % (va.shape,))
print("volume identical   : %s   (max abs diff %g)" % (np.array_equal(va, vb), np.abs(va - vb).max()))
if not np.array_equal(va, vb):
    sys.exit(1)

def bench(fn, reps=15):
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps

t_ord_py = bench(lambda: order_pydicom(FILES, D))
t_ord_fa = bench(lambda: order_fast(FILES, D))
t_px_py = bench(lambda: planes_pydicom(oa, D, idx), 10)
t_px_fa = bench(lambda: planes_fast(ob, D, idx), 10)
print("\nper series (%d slices ordered, %d decoded)" % (n, len(idx)))
print("  ordering pass  pydicom %7.1f ms   fastdcm %7.1f ms   %4.1fx" % (t_ord_py*1e3, t_ord_fa*1e3, t_ord_py/t_ord_fa))
print("  decode pass    pydicom %7.1f ms   fastdcm %7.1f ms   %4.1fx" % (t_px_py*1e3, t_px_fa*1e3, t_px_py/t_px_fa))
tot_py, tot_fa = t_ord_py + t_px_py, t_ord_fa + t_px_fa
print("  series total   pydicom %7.1f ms   fastdcm %7.1f ms   %4.1fx" % (tot_py*1e3, tot_fa*1e3, tot_py/tot_fa))
print("\nextrapolated to 1322 studies x 6 series (same slice counts):")
for label, t in (("pydicom", tot_py), ("fastdcm", tot_fa)):
    s = t * 6 * 1322
    print("  %-8s %8.1f s  (%5.1f min)" % (label, s, s/60))
