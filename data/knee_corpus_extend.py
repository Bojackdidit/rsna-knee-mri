#!/usr/bin/env python3
"""Build the fixed-size slice stacks for the RSNA knee studies from the raw DICOM.

Outputs: extra_vols.npy (N,44,336,336 uint8), extra_masks.npy, extra_ids.npy
"""
import os, glob, time, json
import numpy as np, pandas as pd

IMG = 336
CROP_MM = 140.0
SLOTS = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
         ("Coronal", 0, 6), ("Axial", -1, 8)]
MAXS = sum(s[2] for s in SLOTS)                     # 44
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def find_root():
    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
             "/kaggle/input/rsna-knee-abnormality-detection"]
    for b in cands:
        if os.path.exists(b + "/train.csv"):
            return b
    for d, _, f in os.walk("/kaggle/input"):
        if "train.csv" in f and os.path.isdir(d + "/train_series"):
            return d
    raise RuntimeError("no train root under /kaggle/input")


def find_corpus_ids():
    """all_ids.npy from the published corpus dataset -> the studies we already have."""
    for d, _, fs in os.walk("/kaggle/input"):
        if "all_ids.npy" in fs:
            return np.load(os.path.join(d, "all_ids.npy"), allow_pickle=True).astype(str)
    raise RuntimeError("all_ids.npy not found - attach dreaddevelopment/knee-raptor-corpus")


def _make_reader():
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        fs = glob.glob(sdir + "/*.dcm"); recs = []; ps_list = []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, "ImageOrientationPatient", None)
                ipp = getattr(h, "ImagePositionPatient", None)
                if iop is not None and ipp is not None and len(iop) == 6:
                    r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                    n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
                else:
                    pos = float(getattr(h, "InstanceNumber", 0) or 0)
                ps = getattr(h, "PixelSpacing", None); ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps); recs.append((pos, f, ps))
            except Exception:
                recs.append((0.0, f, 0.5))
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, f, ps in recs], med_ps

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_px, mm_crop_resize


def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(sid, ser_records, tsdir, reader):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r["SeriesInstanceUID"])
        files, med_ps = order_and_meta("%s/%s/%s" % (tsdir, sid, r["SeriesInstanceUID"]))
        if not files:
            idx += k; continue
        n = len(files); lo, hi = int(n * 0.15), int(n * 0.85) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs = []; pss = []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp)); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= MAXS:
                break
            if a is None:
                idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        if idx >= MAXS:
            break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


def main():
    t0 = time.time()
    ROOT = find_root(); print("root", ROOT, flush=True)
    have = set(find_corpus_ids().tolist()); print("already in corpus:", len(have), flush=True)
    tr = pd.read_csv(ROOT + "/train.csv", encoding="latin-1")
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    tser = pd.read_csv(ROOT + "/train_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    ser = {}
    for r in tser.to_dict("records"):
        ser.setdefault(r["StudyInstanceUID"], []).append(r)
    tsdir = ROOT + "/train_series"

    gold = set(tr[tr[LAB].notna().all(axis=1)]["StudyInstanceUID"])
    todo = [u for u in tr["StudyInstanceUID"] if u not in have and u in ser]
    # gold studies first, so they land in the output even if the kernel times out part-way.
    todo.sort(key=lambda u: (u not in gold))
    print("to build: %d  (of which gold: %d)" % (len(todo), len(set(todo) & gold)), flush=True)

    reader = _make_reader()
    vols = np.zeros((len(todo), MAXS, IMG, IMG), np.uint8)
    masks = np.zeros((len(todo), MAXS), np.uint8)
    ids = []
    ok = fail = 0
    for i, sid in enumerate(todo):
        try:
            v, m = build_study(sid, ser, tsdir, reader)
            vols[len(ids)] = v; masks[len(ids)] = m; ids.append(sid); ok += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print("  FAIL %s %s: %s" % (sid[:20], type(e).__name__, e), flush=True)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print("  %d/%d ok=%d fail=%d | %.0fs | eta %.0fm"
                  % (i + 1, len(todo), ok, fail, el, el / (i + 1) * (len(todo) - i - 1) / 60),
                  flush=True)
    n = len(ids)
    np.save("/kaggle/working/extra_vols.npy", vols[:n])
    np.save("/kaggle/working/extra_masks.npy", masks[:n])
    np.save("/kaggle/working/extra_ids.npy", np.array(ids))
    json.dump({"built": n, "failed": fail, "gold_included": len(set(ids) & gold),
               "seconds": time.time() - t0},
              open("/kaggle/working/extra_meta.json", "w"), indent=1)
    print("DONE built=%d failed=%d gold=%d size=%.2fGB in %.0fm"
          % (n, fail, len(set(ids) & gold), vols[:n].nbytes / 1e9, (time.time() - t0) / 60),
          flush=True)


if __name__ == "__main__":
    main()
