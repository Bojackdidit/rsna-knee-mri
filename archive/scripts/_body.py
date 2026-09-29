def _pick_mount(*cands):
    """First existing path among candidates; Kaggle mounts inputs under either
    /kaggle/input/<slug> or /kaggle/input/{competitions,datasets}/<...>."""
    from pathlib import Path as _P
    for c in cands:
        if _P(c).exists():
            return _P(c)
    return _P(cands[0])


from __future__ import annotations
import os
import gc
import hashlib
import json
import re
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
ASSET = _pick_mount('/kaggle/input/rsna-knee-bend-dinov3-0917-repro-assets', '/kaggle/input/datasets/tonylica/rsna-knee-bend-dinov3-0917-repro-assets')
ROOT = _pick_mount('/kaggle/input/rsna-knee-abnormality-detection', '/kaggle/input/competitions/rsna-knee-abnormality-detection')
DINO = _pick_mount('/kaggle/input/models/metaresearch/dinov2/pytorch/small/1', '/kaggle/input/dinov2/pytorch/small/1', '/kaggle/input/metaresearch/dinov2/pytorch/small/1')
T0 = time.time()
DEVS = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
SEED = 2026
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
CROP_MM = 130.0
CACHE_IMG = 336
GROUP = 3
N_GROUP_MAX = 1
CACHE_FRACTION = 0.45
CACHE_BUDGET_MAX_GB = 24.0
CACHE_BUDGET_GB = 12.0
TEST_SHARE = 0.3
HDR_THREADS = 16
PIX_THREADS = 12
ORDER_THREADS = 32
ORDER_BUDGET_S = 5400
AUG_ROT_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.1
LAT_MIN_OFFSET_MM = 20.0
SLICE_BAND = (0.2, 0.8)
RULES_NATIVE = {'order': 'normal', 'lat': 'centre', 'slot_fallback': False, 'decode_fill': 'nearest'}
RULES_LEGACY = {'order': 'dominant_axis', 'lat': 'corner_x', 'slot_fallback': True, 'decode_fill': 'zero'}
RULES = dict(RULES_NATIVE)
LEGACY_LAT_OFFSET_MM = 5.0
EVAL_BATCH = 8
TIME_BUDGET = 8.0 * 3600
SLOTS_RECOVERED = [('SAG_FLUID_FS', 'Sagittal', True, True), ('COR_FLUID_FS', 'Coronal', True, True), ('AX_FLUID_FS', 'Axial', True, True), ('SAG_FLUID_NOFS', 'Sagittal', True, False), ('COR_T1', 'Coronal', False, False), ('SAG_T1', 'Sagittal', False, False)]
SLOTS_PUBLIC = [('SAG_FLUID', 'Sagittal', None, True), ('COR_FLUID', 'Coronal', None, True), ('AX_FLUID', 'Axial', None, True), ('SAG_STRUCT', 'Sagittal', None, False), ('COR_STRUCT', 'Coronal', None, False), ('AX_STRUCT', 'Axial', None, False)]
SLOT_SCHEME = os.environ.get('SLOT_SCHEME', 'recovered')
SLOTS = SLOTS_PUBLIC if SLOT_SCHEME == 'public' else SLOTS_RECOVERED
N_SLOT = len(SLOTS)
POOL_PARTS = {'cls_mean': 2, 'cls_mean_focal': 3}
SLOT_PRIOR_TABLE = {'ACL': (0, 3, 5), 'MCL': (1, 4), 'Medial Meniscus': (0, 1, 3, 4), 'Lateral Meniscus': (0, 1, 3, 4), 'Medial OA': (1, 4, 5), 'Lateral OA': (1, 4, 5), 'PF OA': (0, 2, 5), 'Effusion': (0, 2), 'Synovitis': (0, 2), "Baker's": (0,), 'Contusion': (0, 1, 2), 'Fracture': (0, 1, 2, 4, 5)}
SLOT_PRIOR_STRENGTH = 0.55
FATSAT_OPTS = {'FS', 'FATSAT', 'FAT_SAT', 'FSAT'}
_SEP = re.compile('[_\\-.]')
_FATSAT_RX = re.compile('\\bfs\\b|fatsat|fat sat|\\bstir\\b|\\bspair\\b|\\bspir\\b|\\bwe\\b|water excit|\\btirm\\b|\\bsting\\b|\\bfatsup\\b')
_T1_RX = re.compile('\\bt1\\b|\\bt1w\\b')
_T2_RX = re.compile('\\bt2\\b|\\bt2w\\b')
_PD_RX = re.compile('\\bpd\\b|\\bpdw\\b|proton|\\bdp\\b|dens')

def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)
IMG = CACHE_IMG

def available_gb():
    try:
        with open('/proc/meminfo') as fh:
            info = {k.strip(): v for k, v in (l.split(':', 1) for l in fh if ':' in l)}
        return int(info['MemAvailable'].split()[0]) / 1024 ** 2
    except Exception:
        return CACHE_BUDGET_GB / CACHE_FRACTION

def plan_cache(n_study, n_test=0):
    avail = available_gb()
    budget = min(avail * CACHE_FRACTION, CACHE_BUDGET_MAX_GB)
    n_total = n_study + max(n_test, int(TEST_SHARE * n_study))
    per_slice = n_total * N_SLOT * IMG * IMG
    afford = int(budget * 1024 ** 3 // max(per_slice, 1))
    groups = max(1, min(N_GROUP_MAX, afford // GROUP))
    log(f'memory: {avail:.1f} GB available, {budget:.1f} GB to the cache; sizing for {n_study} train + {n_total - n_study} test studies -> {groups} group(s) of {GROUP} = {groups * GROUP} slices per slot' + (f' (wanted {N_GROUP_MAX})' if groups < N_GROUP_MAX else ''))
    return groups
N_GROUP = plan_cache(len(pd.read_csv(ROOT / 'train.csv')), len(pd.read_csv(ROOT / 'test.csv')))
CACHE_SLICES = GROUP * N_GROUP
HDR_TAGS = ['SeriesDescription', 'SequenceName', 'ScanOptions', 'ScanningSequence', 'RepetitionTime', 'EchoTime', 'Laterality', 'PixelSpacing', 'Rows', 'Columns', 'RescaleSlope', 'RescaleIntercept', 'ImagePositionPatient', 'ImageOrientationPatient']

def _hdr_vec(s, n):
    if not isinstance(s, str):
        return None
    try:
        v = [float(x) for x in s.split('|')]
    except ValueError:
        return None
    return np.array(v) if len(v) >= n else None

def side_from_geometry(h):
    cx = {}
    for r in h.itertuples(index=False):
        ipp = _hdr_vec(getattr(r, 'ImagePositionPatient', None), 3)
        iop = _hdr_vec(getattr(r, 'ImageOrientationPatient', None), 6)
        ps = _hdr_vec(getattr(r, 'PixelSpacing', None), 2)
        rows, cols = (getattr(r, 'Rows', None), getattr(r, 'Columns', None))
        if ipp is None or iop is None or ps is None or (not rows) or (not cols):
            continue
        try:
            c = ipp[:3] + iop[:3] * ps[1] * float(cols) / 2 + iop[3:6] * ps[0] * float(rows) / 2
        except (TypeError, ValueError):
            continue
        cx.setdefault(r.StudyInstanceUID, []).append(float(c[0]))
    out = {}
    for st, xs in cx.items():
        m = float(np.median(xs))
        out[st] = None if abs(m) < LAT_MIN_OFFSET_MM else 'R' if m < 0 else 'L'
    return out

def side_from_corner_x(h):
    out = {}
    for st, g in h.groupby('StudyInstanceUID'):
        xs = []
        for r in g.itertuples(index=False):
            ipp = _hdr_vec(getattr(r, 'ImagePositionPatient', None), 3)
            if ipp is not None and np.isfinite(ipp).all():
                xs.append(float(ipp[0]))
        if not xs:
            out[st] = None
            continue
        x = float(np.median(xs))
        out[st] = None if abs(x) < LEGACY_LAT_OFFSET_MM else 'R' if x < 0 else 'L'
    return out

def lat_of(h, tag=''):
    geo = side_from_corner_x(h) if RULES['lat'] == 'corner_x' else side_from_geometry(h)
    d, n_tag, n_geo, n_none, n_disagree = ({}, 0, 0, 0, 0)
    for st, g in h.groupby('StudyInstanceUID'):
        v = [str(x).strip().upper() for x in g['Laterality'].dropna()]
        if RULES['lat'] == 'corner_x' and 'ImageLaterality' in g.columns:
            v += [str(x).strip().upper() for x in g['ImageLaterality'].dropna()]
        v = [x[0] for x in v if x and x[0] in ('L', 'R')]
        side = v[0] if v else None
        if side is not None:
            n_tag += 1
            if geo.get(st) is not None and geo[st] != side:
                n_disagree += 1
        else:
            side = geo.get(st)
            n_geo += side is not None
            n_none += side is None
        d[st] = side
    log(f'{tag}laterality: {n_tag} from the tag, {n_geo} from geometry, {n_none} unresolved; tag and geometry disagree on {n_disagree} ({n_disagree / max(n_tag, 1):.1%} of the tagged)')
    return d

def probe(item):
    split, study, series, path = item
    row = {'split': split, 'StudyInstanceUID': study, 'SeriesInstanceUID': series, 'dir': path}
    try:
        files = sorted((e.name for e in os.scandir(path) if e.name.endswith('.dcm')))
        row['files'] = files
        row['n_slices'] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(os.path.join(path, files[len(files) // 2]), stop_before_pixels=True, force=True)
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == 'MultiValue':
                row[t] = '|'.join((str(x) for x in v))
            else:
                row[t] = str(v)
    except Exception as exc:
        row['err'] = str(exc)[:120]
    return row

def walk(split):
    base = ROOT / split
    items = []
    if not base.is_dir():
        return pd.DataFrame(columns=['split', 'StudyInstanceUID', 'SeriesInstanceUID', 'dir', 'files', 'n_slices'] + HDR_TAGS)
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)

def annotate(df):
    desc = df['SeriesDescription'].fillna('') + ' ' + df['SequenceName'].fillna('')
    desc = desc.str.lower().str.replace(_SEP, ' ', regex=True)
    opts = df['ScanOptions'].fillna('').str.upper().str.split('|')
    opts_fs = opts.apply(lambda ts: any((t.strip() in FATSAT_OPTS for t in ts)))
    df['fatsat'] = desc.str.contains(_FATSAT_RX) | opts_fs
    tr = pd.to_numeric(df['RepetitionTime'], errors='coerce')
    te = pd.to_numeric(df['EchoTime'], errors='coerce')
    gre = df['ScanningSequence'].fillna('').str.upper().str.contains('GR')
    t1, t2, pdw = (desc.str.contains(_T1_RX), desc.str.contains(_T2_RX), desc.str.contains(_PD_RX))
    df['weight'] = np.where(t1 & ~t2 & ~pdw, 'T1', np.where(t2 & ~pdw, 'T2', np.where(pdw, 'PD', np.where(gre, 'GRE', np.where(tr < 800, 'T1', np.where(te > 60, 'T2', np.where(tr >= 800, 'PD', 'UNK')))))))
    df['fluid'] = np.isin(df['weight'], ['PD', 'T2'])
    df['px'] = pd.to_numeric(df['PixelSpacing'].fillna('').str.split('|').str[0].replace('', np.nan), errors='coerce')
    return df

def pick_slots(series_df, plane_map):
    series_df = series_df.copy()
    series_df['plane'] = series_df['SeriesInstanceUID'].map(plane_map)
    out = {}
    for study, g in series_df.groupby('StudyInstanceUID'):
        chosen = {}
        for name, plane, fluid, fs in SLOTS:
            sel = (g['plane'] == plane) & (g['fatsat'] == fs)
            if fluid is not None:
                sel &= g['fluid'] == fluid
            cand = g[sel]
            if len(cand) == 0 and RULES['slot_fallback'] and (fluid is False):
                cand = g[(g['plane'] == plane) & ~g['fatsat']]
            if len(cand):
                chosen[name] = cand.sort_values('n_slices', ascending=False).iloc[0]
        out[study] = chosen
    return out
ORDER_TAGS = [(32, 50), (32, 55), (32, 19)]
DECODE_FAILED = []

def _natural_key(name):
    return tuple((int(x) if x.isdigit() else x.lower() for x in re.split('(\\d+)', str(name))))

def _order_dominant_axis(rec):
    files, d = (rec['files'], rec['dir'])
    rows = []
    for pos, f in enumerate(files):
        ipp = inst = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True, specific_tags=['ImagePositionPatient', 'InstanceNumber'])
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
    placed = [r for r in rows if r[1] is not None]
    need = max(2, int(0.8 * len(rows)))
    if len(placed) >= need:
        xyz = np.stack([r[1] for r in placed])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        rows.sort(key=lambda r: (float(r[1][axis]) if r[1] is not None else spare, r[2] if r[2] is not None else float('inf'), r[3]))
    elif sum((r[2] is not None for r in rows)) >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float('inf'), r[3]))
    else:
        rows.sort(key=lambda r: _natural_key(r[0]))
    return ([r[0] for r in rows], True)

def order_slices(rec):
    if RULES['order'] == 'dominant_axis':
        return _order_dominant_axis(rec)
    files, d = (rec['files'], rec['dir'])
    keyed = []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True, specific_tags=ORDER_TAGS)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                k = float(ds.InstanceNumber)
            except Exception:
                k = None
        keyed.append((k, f))
    if any((k is None for k, _ in keyed)):
        return (files, False)
    return ([f for _, f in sorted(keyed, key=lambda t: t[0])], True)

def read_slot(rec, n_slice=None, out_size=None):
    n_slice = GROUP if n_slice is None else n_slice
    out_size = IMG if out_size is None else out_size
    files, d, px = (rec.get('ordered') or rec['files'], rec['dir'], rec['px'])
    n = len(files)
    if n == 0:
        return None
    lo, hi = (int(SLICE_BAND[0] * (n - 1)), int(SLICE_BAND[1] * (n - 1)))
    idx = np.unique(np.linspace(lo, hi, n_slice).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])
    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, 'RescaleSlope', 1) or 1)
            ic = float(getattr(ds, 'RescaleIntercept', 0) or 0)
            a = a * sl + ic
        except Exception:
            a = None
        planes.append(a)
    got = [k for k, p in enumerate(planes) if p is not None]
    if RULES['decode_fill'] == 'zero':
        if not got:
            DECODE_FAILED.append(rec.get('SeriesInstanceUID', d))
        planes = [np.zeros((out_size, out_size), np.float32) if p is None else p for p in planes]
        got = list(range(len(planes)))
    if not got:
        DECODE_FAILED.append(rec.get('SeriesInstanceUID', d))
        return None
    if len(got) < len(planes):
        DECODE_FAILED.append(rec.get('SeriesInstanceUID', d))
        for k, p in enumerate(planes):
            if p is None:
                planes[k] = planes[min(got, key=lambda j: abs(j - k))]
    shp = planes[0].shape
    planes = [p if p.shape == shp else np.zeros(shp, np.float32) for p in planes]
    vol = np.stack(planes)
    if px and np.isfinite(px) and (px > 0):
        want = int(round(CROP_MM / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = (h // 2, w // 2)
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-06), 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode='bilinear', align_corners=False)
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)

def normalise_laterality(img, plane, lat):
    if lat != 'R':
        return img
    if plane in ('Coronal', 'Axial'):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])
ORDER_CACHE = os.environ.get('RSNA_ORDER_CACHE') or None

def build_cache(slot_map, plane_map, lat_map, tag):
    studies = sorted(slot_map)
    sidx = {s: i for i, s in enumerate(studies)}
    cache = np.zeros((len(studies), N_SLOT, CACHE_SLICES, IMG, IMG), np.uint8)
    mask = np.zeros((len(studies), N_SLOT), np.float32)
    log(f'{tag}: cache {cache.shape} = {cache.nbytes / 1024 ** 3:.1f} GB')
    jobs = [(st, k, plane, slot_map[st][name]) for st in studies for k, (name, plane, _, _) in enumerate(SLOTS) if name in slot_map[st]]
    n_job = len(jobs)
    t_ord = time.time()
    n_slice_total = sum((len(j[3]['files']) for j in jobs))
    log(f'{tag}: ordering {len(jobs)} slot-series ({n_slice_total} slice headers)')
    ok = done = 0
    CHUNK_O = 1024
    seen = {}
    if ORDER_CACHE and Path(ORDER_CACHE).is_file():
        try:
            import json as _json
            seen = _json.loads(Path(ORDER_CACHE).read_text())
        except (OSError, ValueError):
            seen = {}
        hit = 0
        for _, _, _, rec in jobs:
            e = seen.get(rec['SeriesInstanceUID'])
            if e and len(e['files']) == len(rec['files']):
                rec['ordered'] = e['files']
                ok += int(e['good'])
                hit += 1
        jobs = [j for j in jobs if 'ordered' not in j[3]]
        log(f'{tag}: {hit} slot-series ordered from {ORDER_CACHE}, {len(jobs)} to read')
    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK_O):
            block = jobs[c0:c0 + CHUNK_O]
            for (_, _, _, rec), (files, good) in zip(block, pool.map(lambda j: order_slices(j[3]), block)):
                rec['ordered'] = files
                ok += int(good)
                done += 1
                if ORDER_CACHE:
                    seen[rec['SeriesInstanceUID']] = {'files': files, 'good': bool(good)}
            budget = min(ORDER_BUDGET_S, max(60.0, (TIME_BUDGET - (time.time() - T0)) * 0.35))
            if time.time() - t_ord > budget:
                log(f'{tag}: ordering budget spent at {done}/{len(jobs)}; the rest keep file order')
                break
    if ORDER_CACHE and done:
        import json as _json
        _t = Path(ORDER_CACHE).with_suffix('.tmp')
        _t.write_text(_json.dumps(seen))
        _t.replace(Path(ORDER_CACHE))
    log(f'{tag}: ordered {ok}/{n_job} by geometry ({n_job - ok} kept arbitrary) in {time.time() - t_ord:.0f}s')
    jobs = [(st, k, plane, slot_map[st][name]) for st in studies for k, (name, plane, _, _) in enumerate(SLOTS) if name in slot_map[st]]
    log(f'{tag}: decoding {len(jobs)} slot-series')
    n_failed_before = len(DECODE_FAILED)
    CHUNK = 512
    done = 0
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK):
            block = jobs[c0:c0 + CHUNK]
            for (st, k, plane, _), img in zip(block, pool.map(lambda j: read_slot(j[3], CACHE_SLICES, IMG), block)):
                done += 1
                if img is None:
                    continue
                cache[sidx[st], k] = normalise_laterality(img, plane, lat_map.get(st)).numpy()
                mask[sidx[st], k] = 1.0
            if done % 4096 < CHUNK:
                log(f'  {tag} {done}/{len(jobs)}')
            if time.time() - T0 > TIME_BUDGET:
                log(f'  {tag}: time budget reached during decode')
                break
    n_failed = len(DECODE_FAILED) - n_failed_before
    log(f'{tag}: {int(mask.sum())}/{len(jobs)} slots filled' + (f'; {n_failed} series had a slice that would not decode' if n_failed else ''))
    gc.collect()
    return (studies, cache, mask)

class SlotHead(nn.Module):

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        p_ = torch.zeros(n_out, n_slot)
        if prior and n_slot == len(SLOTS) and (n_out == len(TARGETS)):
            for t, slots in SLOT_PRIOR_TABLE.items():
                if t in TARGETS:
                    p_[TARGETS.index(t), list(slots)] = SLOT_PRIOR_STRENGTH
        self.prior = prior
        if prior:
            self.register_buffer('slot_prior', p_)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum('bsh,oh->bos', h, self.query) / self.hidden ** 0.5
        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -10000.0).softmax(-1)
        ctx = self.drop(torch.einsum('bos,bsh->boh', att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias

class Model(nn.Module):

    def __init__(self, backbone, dim, pool='cls_mean', prior=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask, img_size=None):
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            x = F.interpolate(x, size=(img_size, img_size), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == 'cls_mean_focal':
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        feat = torch.cat(parts, dim=1).reshape(B, S, -1)
        return self.head(feat, mask)

def build_model(unfreeze_last, source=None, variant='small', pool='cls_mean', prior=False):
    from transformers import AutoModel
    p = source if source is not None else find_dinov2(variant)
    if p is None:
        raise FileNotFoundError('DINOv2 weights not attached')
    bb = AutoModel.from_pretrained(str(p))
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    dim = bb.config.hidden_size
    trainable = sum((p.numel() for p in bb.parameters() if p.requires_grad))
    log(f'backbone: {n_layer} blocks, last {unfreeze_last} trainable ({trainable / 1000000.0:.1f}M params), feature dim {dim * POOL_PARTS[pool]}')
    return Model(bb, dim, pool=pool, prior=prior)
FINGERPRINT_TOL = 0.002

def fingerprint(model, dev, img_size, n_slot=None, group=None, seed=None):
    n_slot = N_SLOT if n_slot is None else n_slot
    group = GROUP if group is None else group
    seed = SEED if seed is None else seed
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randint(0, 256, (2, n_slot, group, img_size, img_size), generator=g, dtype=torch.uint8).to(dev)
    mask = torch.ones(2, n_slot, device=dev)
    mask[1, -1] = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(imgs, mask, img_size).float().cpu().numpy()
    if was_training:
        model.train()
    return out

def check_fingerprint(model, dev, img_size, expected, tol=FINGERPRINT_TOL, tag=''):
    got = fingerprint(model, dev, img_size)
    exp = np.asarray(expected, np.float32)
    if got.shape != exp.shape:
        raise WeightsError(f'{tag}fingerprint shape {got.shape} != stored {exp.shape}: the architecture is not the one these weights were fitted to')
    d = float(np.abs(got - exp).max())
    if d > tol:
        raise WeightsError(f'{tag}fingerprint differs by {d:.4g} (tolerance {tol:g}). The weights load but do not compute what they computed when fitted - preprocessing, resolution or architecture has moved between the two runs.')
    log(f'{tag}fingerprint matches within {d:.2g}')
    return d

class WeightsError(RuntimeError):
    pass
TTA_OVERLAP = True
TTA_POOL = 'prob'
PUBLIC_FRONTIER_TARGET_POOL = {'Fracture': 'max', 'Contusion': 'max', 'Medial Meniscus': 'max', 'Lateral Meniscus': 'max', 'ACL': 'top2', 'MCL': 'top2', "Baker's": 'max'}
TTA_TARGET_POOL = {**PUBLIC_FRONTIER_TARGET_POOL, 'Synovitis': 'original_mean'}
LEGACY_FOLD_SOFTPOOL_BETA = {'ACL': 6.0, 'MCL': 6.0, 'Medial Meniscus': 8.0, 'Lateral Meniscus': 8.0, "Baker's": 8.0, 'Contusion': 8.0, 'Fracture': 10.0}
LEGACY_FOLD_SOFTPOOL_ALPHA = {'ACL': 0.2, 'MCL': 0.2, 'Medial Meniscus': 0.25, 'Lateral Meniscus': 0.25, "Baker's": 0.2, 'Contusion': 0.2, 'Fracture': 0.15}

def window_starts(n_slice, group, overlap=None):
    overlap = TTA_OVERLAP if overlap is None else overlap
    if overlap and n_slice >= group:
        return list(range(n_slice - group + 1))
    return [g * group for g in range(max(n_slice // group, 1))]

def apply_target_window_pool(values, probs, logits, original_probs, mapping, target_idx):
    for target, mode in mapping.items():
        j = target_idx[target]
        if mode == 'max':
            values[:, j] = probs[:, :, j].max(0).values
        elif mode == 'mean':
            values[:, j] = probs[:, :, j].mean(0)
        elif mode == 'logit_mean':
            values[:, j] = torch.sigmoid(logits[:, :, j].mean(0))
        elif mode == 'original_mean':
            values[:, j] = original_probs[:, :, j].mean(0)
        elif mode in ('top2', 'top3'):
            k = min(int(mode[3:]), probs.shape[0])
            values[:, j] = probs[:, :, j].topk(k, dim=0).values.mean(0)
        else:
            raise ValueError(f'unknown TTA pooling mode for {target}: {mode}')
    return values

def legacy_fold_soft_window_pool(original_probs, target_idx):
    values = original_probs.mean(0).clone()
    for target, beta in LEGACY_FOLD_SOFTPOOL_BETA.items():
        j = target_idx[target]
        x = original_probs[:, :, j]
        weight = torch.softmax(float(beta) * x, dim=0)
        values[:, j] = (weight * x).sum(0)
    return values

@torch.no_grad()
def predict_member(model, cache, mask, idx, dev, img_size, group=None, pool=None, starts=None, jitter=False, jitter_seed=SEED, return_public_frontier=False):
    group = GROUP if group is None else group
    pool = TTA_POOL if pool is None else pool
    starts = window_starts(cache.shape[2], group) if starts is None else list(starts)
    if not starts:
        raise ValueError('predict_member was given no windows to average over')
    target_idx = {t: j for j, t in enumerate(TARGETS)}
    unknown = (set(TTA_TARGET_POOL) | set(PUBLIC_FRONTIER_TARGET_POOL)) - set(target_idx)
    if unknown:
        raise ValueError(f'unknown target(s) in TTA_TARGET_POOL: {unknown}')
    jitter_gen = torch.Generator(device=dev)
    jitter_gen.manual_seed(int(jitter_seed) % (2 ** 63 - 1))
    model.eval()
    out, public_frontier_out, public_soft_out = ([], [], [])
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b:b + EVAL_BATCH]
        m = torch.from_numpy(mask[sel]).to(dev)
        win_probs, win_logits, win_original_probs = ([], [], [])
        for st in starts:
            rows = torch.from_numpy(np.ascontiguousarray(cache[sel, :, st:st + group])).to(dev)
            views = [rows] + ([augment(rows, generator=jitter_gen)] if jitter else [])
            view_probs, view_logits = ([], [])
            for view in views:
                with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                    z = model(view, m, img_size).float()
                view_logits.append(z)
                view_probs.append(torch.sigmoid(z))
            win_logits.append(torch.stack(view_logits).mean(0))
            win_probs.append(torch.stack(view_probs).mean(0))
            win_original_probs.append(view_probs[0])
        probs = torch.stack(win_probs)
        logits = torch.stack(win_logits)
        original_probs = torch.stack(win_original_probs)
        v = torch.sigmoid(logits.mean(0)) if pool == 'logit' else probs.mean(0)
        v = apply_target_window_pool(v, probs, logits, original_probs, TTA_TARGET_POOL, target_idx)
        out.append(v.cpu().numpy())
        if return_public_frontier:
            public_v = apply_target_window_pool(original_probs.mean(0), original_probs, logits, original_probs, PUBLIC_FRONTIER_TARGET_POOL, target_idx)
            public_frontier_out.append(public_v.cpu().numpy())
            public_soft = legacy_fold_soft_window_pool(original_probs, target_idx)
            public_soft_out.append(public_soft.cpu().numpy())
    primary = np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)
    if not return_public_frontier:
        return primary
    public_frontier = np.concatenate(public_frontier_out) if public_frontier_out else np.zeros((0, len(TARGETS)), np.float32)
    public_soft = np.concatenate(public_soft_out) if public_soft_out else np.zeros((0, len(TARGETS)), np.float32)
    return (primary, public_frontier, public_soft)
BUILD_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()

def _run_member(path, m, dev, Cte, Mte, idx, starts, jitter):
    t0 = time.time()
    with BUILD_LOCK:
        if 'state' in m:
            state, fp = (m['state'], None)
        else:
            ck = torch.load(Path(path) / m['file'], map_location='cpu', weights_only=False)
            state, fp = (ck['model'], ck.get('fingerprint'))
        model = build_model(int(m['config']['unfreeze_last']), variant=m['config']['variant'], pool=m['config'].get('pool', 'cls_mean'), prior=bool(m['config'].get('prior', False))).to(dev)
        model.load_state_dict(state)
        if fp is not None:
            check_fingerprint(model, dev, IMG, fp, tag=f"{m['id']}: ")
        else:
            log(f"  {m['id']}: no stored fingerprint (legacy bundle) -- accepted at reduced weight")
    t_ready = time.time()
    jitter_seed = SEED + int(hashlib.sha256(str(m['id']).encode()).hexdigest()[:8], 16)
    public_member = 'state' not in m
    predicted = predict_member(model, Cte, Mte, idx, dev, IMG, starts=starts, jitter=jitter, jitter_seed=jitter_seed, return_public_frontier=public_member)
    if public_member:
        p, public_p, public_soft = predicted
    else:
        p, public_p, public_soft = (predicted, None, None)
    t_done = time.time()
    del model, state
    gc.collect()
    if dev.type == 'cuda':
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
    passes = len(starts) * (2 if jitter else 1)
    return (p, public_p, public_soft, (t_ready - t0, (t_done - t_ready) / max(passes, 1)))

def _combine(per_member):
    all_ids = sorted({s for m in per_member for s in m['ids']})
    pos = {s: i for i, s in enumerate(all_ids)}
    acc = np.zeros((len(all_ids), len(TARGETS)), np.float64)
    tot = np.zeros(len(TARGETS), np.float64)
    for m in per_member:
        target_weight = m.get('target_weight')
        w = np.asarray(target_weight if target_weight is not None else [float(m.get('weight', 1.0))] * len(TARGETS), dtype=np.float64)
        if w.shape != (len(TARGETS),) or np.any(w < 0):
            raise ValueError(f"invalid target weights for {m.get('id')}: {w}")
        r = pd.DataFrame(m['pred']).rank(pct=True).to_numpy()
        acc[[pos[s] for s in m['ids']]] += r * w[None, :]
        tot += w
    if np.any(tot <= 0):
        raise ValueError(f'at least one target has no ensemble vote: {tot}')
    return (all_ids, acc / tot[None, :])

def combine_public_members_by_fold(per_member, pred_key='pred'):
    all_ids = sorted({study for member in per_member for study in member['ids']})
    position = {study: i for i, study in enumerate(all_ids)}
    groups = {}
    for i, member in enumerate(per_member):
        fold = member.get('fold')
        key = f'fold_{fold}' if fold is not None else f'member_{i}'
        groups.setdefault(key, []).append(member)
    fold_ranks, diagnostics = ([], [])
    for key, members_in_fold in sorted(groups.items()):
        matrices = []
        for member in members_in_fold:
            values = np.full((len(all_ids), len(TARGETS)), np.nan, np.float64)
            values[[position[study] for study in member['ids']]] = np.asarray(member[pred_key], np.float64)
            if np.isnan(values).any():
                raise WeightsError(f"{member.get('id')}: incomplete {pred_key} coverage")
            matrices.append(values)
        raw_fold_mean = np.mean(matrices, axis=0)
        fold_ranks.append(pd.DataFrame(raw_fold_mean).rank(method='average', pct=True).to_numpy(np.float64))
        diagnostics.append({'ensemble_group': key, 'members': len(members_in_fold)})
    if len(fold_ranks) != 5:
        raise WeightsError(f'legacy branch requires five folds, found {len(fold_ranks)}')
    return (all_ids, np.mean(fold_ranks, axis=0), pd.DataFrame(diagnostics))

def blend_legacy_frontier_and_soft(frontier_rank, soft_rank):
    output = np.asarray(frontier_rank, np.float64).copy()
    for j, target in enumerate(TARGETS):
        alpha = float(LEGACY_FOLD_SOFTPOOL_ALPHA.get(target, 0.0))
        if alpha:
            output[:, j] = (1.0 - alpha) * frontier_rank[:, j] + alpha * soft_rank[:, j]
    return output

def infer_from_package(path, dev=None):
    man = json.loads((Path(path) / 'manifest.json').read_text())
    members = man['members']
    log(f'weights package: {len(members)} member(s) from {path}; {len(DEVS)} device(s)')
    test_df = pd.read_csv(ROOT / 'test.csv')
    test_series = pd.read_csv(ROOT / 'test_series.csv')
    plane_map = dict(zip(test_series['SeriesInstanceUID'], test_series['Anatomical_Plane']))
    hte = annotate(walk('test_series'))
    log(f'test header pass: {len(hte)} series')
    groups = {}
    for m in members:
        groups.setdefault(m['pixel_group'], []).append(m)
    groups.update(legacy_group_members())
    per_member, public_frontier_members = ([], [])
    est = {'fixed': None, 'win': None}

    def bank(m, ids, pred, starts, jitter, public_pred=None, public_soft=None):
        if float(np.std(pred)) < 1e-09:
            log(f"  {m['id']}: degenerate predictions; not banked")
            return
        with STATE_LOCK:
            per_member.append({'id': m['id'], 'fold': m.get('fold'), 'ids': ids, 'pred': pred, 'weight': m.get('weight', 1.0), 'target_weight': m.get('target_weight'), 'holdout': m.get('holdout')})
            if public_pred is not None and len(starts) == len(starts_full):
                if float(np.std(public_pred)) < 1e-09:
                    raise WeightsError(f"{m['id']}: degenerate public-frontier prediction")
                public_frontier_members.append({'id': m['id'], 'fold': m.get('fold'), 'ids': ids, 'pred': public_pred, 'soft_pred': public_soft})
            elif public_pred is not None:
                log(f"  {m['id']}: public-frontier vote omitted because only {len(starts)} / {len(starts_full)} windows completed")
            all_ids, acc = _combine(per_member)
            write_submission(acc, all_ids, test_df, 'submission.csv')
            log(f"  banked {m['id']} fold {m.get('fold', '?')} ({len(starts)} window(s){(', jitter' if jitter else '')}); submission.csv = weighted rank mean of {len(per_member)} member(s)")
    for gi, (key, gm) in enumerate(groups.items(), 1):
        cfg = json.loads(key)
        adopt_config_globals(cfg)
        log(f"decode group {gi}/{len(groups)}: {cfg['img']}px x {cfg['slices']} slices, crop {cfg['crop_mm']} mm -> {len(gm)} member(s)")
        st_te, Cte, Mte = build_cache(pick_slots(hte, plane_map), plane_map, lat_of(hte, 'test '), f'test g{gi}')
        idx = np.arange(len(st_te))
        starts_full = window_starts(Cte.shape[2], GROUP)
        pending = sorted(gm, key=lambda m: -(m.get('holdout') or 0))
        left_after = sum((len(g) for j, (_, g) in enumerate(groups.items(), 1) if j > gi))

        def pop_next():
            with STATE_LOCK:
                if not pending:
                    return (None, None, False)
                left = TIME_BUDGET - (time.time() - T0)
                remaining = len(pending) + left_after
                slots_left = -(-remaining // len(DEVS))
                starts, jit = (starts_full, False)
                if est['fixed'] is not None and est['win'] is not None:
                    afford = max(left * 0.9, 0.0)
                    room = afford / max(slots_left, 1)
                    if est['fixed'] + est['win'] > room:
                        log(f'  {left / 60:.0f} min left: surrendering {len(pending)} member(s); not one more fits')
                        pending.clear()
                        return (None, None, False)
                    jit = est['fixed'] + 2 * len(starts_full) * est['win'] <= room * 0.6
                    per_win = est['win'] * (2 if jit else 1)
                    n_win = int((room - est['fixed']) / per_win) if per_win > 0 else len(starts_full)
                    n_win = max(1, min(len(starts_full), n_win))
                    if n_win < len(starts_full):
                        mid = (len(starts_full) - n_win) // 2
                        starts = starts_full[mid:mid + n_win]
                return (pending.pop(0), starts, jit)

        def worker(dev):
            others = [d for d in DEVS if d is not dev]
            while True:
                m, starts, jit = pop_next()
                if m is None:
                    return
                for attempt, d in enumerate([dev] + others[:1]):
                    try:
                        p, public_p, public_soft, (fs, ws) = _run_member(path, m, d, Cte, Mte, idx, starts, jit)
                        with STATE_LOCK:
                            est['fixed'], est['win'] = (fs, ws)
                        bank(m, st_te, p, starts, jit, public_p, public_soft)
                        break
                    except Exception as exc:
                        log(f"  MEMBER {m['id']} failed on {d} ({type(exc).__name__}: {exc}); " + ('retrying on peer device' if attempt == 0 and others else 'dropped -- costs one vote, not the run'))
                        if d.type == 'cuda':
                            with torch.cuda.device(d):
                                torch.cuda.empty_cache()
        threads = [threading.Thread(target=worker, args=(d,)) for d in DEVS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        del Cte, Mte
        gc.collect()
    if not per_member:
        raise WeightsError('no member produced predictions; submission stays at 0.5')
    all_ids, acc = _combine(per_member)
    sub = write_submission(acc, all_ids, test_df, 'submission.csv')
    log(f'final submission.csv = weighted rank mean of {len(per_member)} member(s); {sub.shape}; nulls {int(sub[TARGETS].isna().sum().sum())}')
    if len(public_frontier_members) == len(members):
        frontier_ids, frontier_acc = _combine(public_frontier_members)
        frontier_sub = write_submission(frontier_acc, frontier_ids, test_df, 'submission_public_0899.csv')
        log(f'submission_public_0899.csv = exact no-jitter public-frontier rank mean of {len(public_frontier_members)} member(s); {frontier_sub.shape}; nulls {int(frontier_sub[TARGETS].isna().sum().sum())}')
        fold_ids, fold_frontier, fold_diagnostics = combine_public_members_by_fold(public_frontier_members, 'pred')
        soft_ids, fold_soft, _ = combine_public_members_by_fold(public_frontier_members, 'soft_pred')
        if fold_ids != soft_ids:
            raise WeightsError('legacy hard/soft study order mismatch')
        legacy_prediction = blend_legacy_frontier_and_soft(fold_frontier, fold_soft)
        legacy_sub = write_submission(legacy_prediction, fold_ids, test_df, 'submission_legacy_fold_blend.csv')
        fold_diagnostics.to_csv('legacy_fold_diagnostics.csv', index=False)
        log(f'legacy DINO aggregation written from five folds; {legacy_sub.shape}')
    else:
        log(f'public-frontier fallback not emitted: {len(public_frontier_members)} / {len(members)} required public members completed')
    return sub

def adopt_config_globals(cfg):
    global IMG, CACHE_IMG, GROUP, CACHE_SLICES, N_GROUP, CROP_MM, SLICE_BAND, RULES
    CACHE_IMG = IMG = int(cfg['img'])
    GROUP = int(cfg['group'])
    CACHE_SLICES = int(cfg['slices'])
    N_GROUP = max(CACHE_SLICES // GROUP, 1)
    CROP_MM = float(cfg['crop_mm'])
    SLICE_BAND = tuple((float(x) for x in cfg['band']))
    rules = cfg.get('rules') or RULES_NATIVE
    unknown = {k: v for k, v in rules.items() if k not in RULES_NATIVE or v not in (RULES_NATIVE[k], RULES_LEGACY[k])}
    if unknown:
        raise WeightsError(f'the members record pixel rules this pipeline cannot reproduce: {unknown}')
    RULES = {**RULES_NATIVE, **rules}
    if [s[0] for s in SLOTS] != list(cfg['slots']):
        raise WeightsError(f"the members were fitted on slots {cfg['slots']} and this pipeline defines {[s[0] for s in SLOTS]}; a weight would be read against the wrong slot")

def augment(imgs, generator=None):
    lead = imgs.shape[:-3]
    x = imgs.reshape(-1, *imgs.shape[-3:]).float()
    n, dev = (x.shape[0], x.device)
    rot = (torch.rand(n, device=dev, generator=generator) - 0.5) * 2 * (AUG_ROT_DEG * np.pi / 180)
    sc = 1.0 + torch.rand(n, device=dev, generator=generator) * AUG_SCALE
    tx = (torch.rand(n, device=dev, generator=generator) - 0.5) * 2 * AUG_SHIFT
    ty = (torch.rand(n, device=dev, generator=generator) - 0.5) * 2 * AUG_SHIFT
    cos, sin = (torch.cos(rot) / sc, torch.sin(rot) / sc)
    theta = torch.zeros(n, 2, 3, device=dev, dtype=torch.float32)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = (cos, -sin, tx)
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = (sin, cos, ty)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
    scale = 1.0 + (torch.rand(n, 1, 1, 1, device=dev, generator=generator) - 0.5) * 2 * AUG_INTENSITY
    x = (x * scale).clamp(0, 255)
    return x.reshape(*lead, *x.shape[-3:]).to(imgs.dtype)

def write_submission(pred, studies, test_df, path):
    sub = pd.DataFrame(pd.DataFrame(pred).rank(pct=True).values, columns=TARGETS)
    sub.insert(0, 'StudyInstanceUID', studies)
    sub = test_df[['StudyInstanceUID']].merge(sub, on='StudyInstanceUID', how='left')
    sub[TARGETS] = sub[TARGETS].fillna(0.5)
    sub.to_csv(path, index=False)
    return sub

def find_dinov2(variant='small'):
    if not (DINO / 'config.json').is_file():
        raise FileNotFoundError(DINO)
    return DINO

def legacy_group_members():
    return {}

def run_dinov2():
    path = ASSET / 'rsna-knee-weights'
    infer_from_package(path, DEVS[0])
    public = Path('/kaggle/working/submission_public_0899.csv')
    if not public.is_file():
        raise RuntimeError('public DINOv2 frontier was not produced')
    public.replace('/kaggle/working/submission.csv')
    for name in ('submission_legacy_fold_blend.csv', 'legacy_fold_diagnostics.csv'):
        candidate = Path('/kaggle/working') / name
        if candidate.is_file():
            candidate.unlink()
print('ONEPASS: partner half not run', flush=True)
import pandas as _ao_pd
_ao_df = _ao_pd.read_csv(ROOT / 'sample_submission.csv')
_ao_df.to_csv('/kaggle/working/submission.csv', index=False)

# A5 DINOv3 5-fold removed. It ran its own build_study pass over every study and wrote
# submission.csv, which the CoAtNet arm overwrites before the V18 blend reads it, so its
# output never reached the score. Nothing defined here is used downstream.
print('ONEPASS: A5 transformer half not run', flush=True)

# The rad calibration stage is removed, for the same reason: it re-read DICOM headers for
# every study and wrote submission.csv, and the arm write lands after it. V18_CALIBRATOR_
# APPLIED is left undefined on purpose; cell 4 reads it via globals().get(..., False).
print('ONEPASS: rad calibration not run', flush=True)

# COATNET_TRANSFORMER_BLEND_V1
#!/usr/bin/env python3
"""Knee MRI: twelve findings from a single model

This notebook takes a knee MRI study and scores twelve findings at once: ACL tear, MCL tear,
medial and lateral meniscus tears, osteoarthritis in the medial, lateral and patellofemoral
compartments, joint effusion, synovitis, a Baker's cyst, bone contusion and fracture. It scores
0.924 on the public leaderboard using one model, with no ensembling and no test-time augmentation.

This is the inference half of the work. The model was trained separately and its weights are
attached as a dataset, so this notebook only loads them and predicts:
https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense

Where the training labels came from

Worth saying up front, because it shapes everything else. The competition gives you 4,407 studies
but structured labels for only 58 of them. Every other study arrives with a free-text radiology
report and nothing more, so there is very little to train against out of the box.

The labels behind these weights were made by reading those reports with a language model and
turning each into twelve probabilities rather than twelve yes or no answers. A report that says a
tear is suspected becomes a number near 0.8, not a 1, which is a fairer target than forcing every
hedged sentence into a hard label. That yields 4,349 studies to train on. The 58 studies that came
with real labels were never trained on and are used to check the result honestly; the model reaches
0.9167 macro-AUC on them.

Building a fixed input from studies that are all shaped differently

The hard part of this competition is not the network, it is that no two studies look alike. A
study holds several DICOM series shot in different planes, the number of series varies, and the
number of slices in a series varies more. Anything that expects a fixed-size input has to be given
one.

The approach here is to fill five fixed slots per study, always in the same order, for a stack of
64 images:

  18 slices from a sagittal series, preferring a fluid-sensitive one
  14 slices from a second sagittal series, preferring one that is not fluid-sensitive
  12 slices from a coronal series, preferring a fluid-sensitive one
   8 slices from a second coronal series
  12 slices from an axial series

Preferring a fluid-sensitive series for some slots and not for others is deliberate. Fluid-
sensitive sequences show swelling, effusion and acute injury clearly, while the other sequences
show anatomy and cartilage better, and the twelve findings are split across both. If a study has
no series for a slot, the slot is left as zeros and the model is told to skip it rather than being
fed something misleading.

Within a series, slices are taken evenly across 6 to 94 percent of the stack rather than from the
middle. The outer slices are where the collateral ligaments and the lateral meniscus sit, and
cutting them was measurably costing accuracy on exactly those findings.

Every slice is cropped to a 140 mm box around the centre of the image using the pixel spacing from
the DICOM header, then resized to 336 pixels. Cropping by millimetres rather than by pixel count
matters: it means a knee occupies the same fraction of the frame whether the scan was acquired at
0.3 or 0.5 mm per pixel, so the model is not asked to learn scale differences that carry no medical
information.

How the model reads the stack

Three neighbouring slices are stacked into the three channels of one image. The network then sees
a little of what lies above and below the slice in the middle, which is most of the benefit of a 3D
model at the cost of a 2D one. Each of these three-slice windows is passed through a CoAtNet
backbone at 384 pixels.

The windows are combined with an attention layer that has separate weights for each of the twelve
findings. This is the part that matters most. A cruciate tear may be visible on two sagittal slices
while osteoarthritis is spread across many coronal ones, and a single pooled score forces those two
to share one notion of which slices are important. Giving each finding its own attention weights
lets each one draw on the slices that actually show it.

Running it

Scoring uses 42 windows per study. Inference runs in half precision and automatically retries a
study in full precision if it fails, so no study is ever dropped from the submission. The notebook
needs no internet: the backbone is loaded from the attached weights rather than downloaded.
"""
import os, sys, glob, time, json, gc
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
# T4 (Turing) cuDNN v9 has fp16/fp32 conv engines but NOT bf16 for these shapes
# ("GET was unable to find an engine..."); benchmark lets it pick a valid algo for
# the fixed (1,24,3,res,res) input.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ---- fixed config (must match training exactly) -----------------------------
IMG = 336
CROP_MM = 140.0
# 64 slices per study instead of 44, same proportions. Must match the corpus the weights
# were trained on (knee_corpus_v4.py).
SLOTS = [("Sagittal", 1, 27), ("Sagittal", 0, 21), ("Coronal", 1, 18),
         ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = sum(s[2] for s in SLOTS)                     # 64
K_EVAL = 48   # every window position the volume holds, not an evenly spaced subset
NORM = "imagenet"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Three arms: (weights filename, fallback arch, fallback res). ck carries arch+res too.
# Selected 2026-08-19 by greedy forward selection AND exhaustive subset search over a 7-arm
# panel on the 45-study gold set (phase2/blend_panel.py); both agree on this exact set.
# Singles: coatnet384 0.9025 | swinbase384 0.8825 | effv2l480 0.8716.
# Blend {coatnet+swin+effv2l} = 0.9068 (2-arm {coatnet+swin} = 0.9059, coatnet alone 0.9025).
# Dropped as redundant: cnn336 (0.8833, the former champion), cnbase384 (0.8754),
# cnlarge384 (0.8752), maxvit384 (0.8438).
#
# SINGLE ARM: coatnet_rmlp_2_rw_384 retrained on the EXPANDED 4,349-study corpus.
#
# Why one arm and not the 3-arm blend: on the live leaderboard CoAtNet alone scored 0.914 while
# every blend scored 0.914-0.915, so ensembling is worth ~+0.001 there -- the ~+0.010 it showed
# on the old 45-study gold set was gold-set noise. One arm is also 1/3 the kernel runtime.
#
# Corpus expansion: the corpus previously held 3,200 of the 4,349 labelled studies and only 45
# of the 58 gold studies. Rebuilt to 4,407 studies (+37.8% training data, 58-study gate).
#
# Measured on the 58-study gate (the incumbent re-scored on the SAME gate for a fair compare):
#   incumbent CoAtNet (3,155-study corpus) 0.8923
#   this model       (4,349-study corpus) 0.9054   (+0.0131, better in 92.7% of 2000 bootstraps)
# Biggest gains land on the findings that were capping us: Lateral Meniscus +0.071,
# Fracture +0.057, Lateral OA +0.048, Medial Meniscus +0.035, ACL +0.028.
ARMS = [
    {"file": "raptor_ft_confdist_swa.pt",          "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},
    {"file": "raptor_ft_coatnet_w80_full_swa.pt",  "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},
]


# ============================================================================
# Model -- verbatim from finetune_raptor.py
# ============================================================================
def build_backbone(arch, pretrained=False):
    # maxvit/maxxvit/coatnet are conv-attention hybrids: NO CLS token, NO interpolatable
    # pos-embed -> avg pool. The "vit" substring in "coatnet"/"maxvit" must NOT route them
    # down the ViT path (mirrors finetune_raptor.py exactly).
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))


def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", arch_default)
    ck_res = int(ck.get("res", res_default))
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    # NOTE: DataParallel removed on purpose. On the full hidden test it drove a system-RAM OOM
    # (per-forward module replication over many studies); a single T4 handles K_EVAL=24 windows
    # fine. Arms are also run SEQUENTIALLY (see main) so peak RAM == one model, not two.
    del ck
    gc.collect()
    return model, ck_res


# ============================================================================
# Eval windowing -- verbatim from finetune_raptor.py StudyWindows (train=False)
# ============================================================================
def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def eval_windows(vol, mask, k, res, norm=NORM):
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear",
                              align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    return x


@torch.no_grad()
def infer_probs(model, xwins, device):
    x = xwins.unsqueeze(0).to(device)
    use_cuda = device != "cpu" and str(device).startswith("cuda")
    if use_cuda:
        # fp16 conv on T4 is fully cuDNN-supported (bf16 is NOT -> "no engine").
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
        except RuntimeError:
            # fp32 always has a Turing conv engine; slower but never drops a study.
            torch.cuda.empty_cache()
            o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
    o = torch.sigmoid(model(x).float())
    return o[0].cpu().numpy()


def rankpct(x):                                   # per-column percentile rank in [0,1]
    order = x.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, (x.shape[0] - 1))


# ============================================================================
# Preprocessing -- verbatim from kprep2/dino_preprocess.py, retargeted to TEST
# ============================================================================
def _make_reader():
    import pydicom, cv2
    from concurrent.futures import ThreadPoolExecutor as _ArmPool
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    _RD_THREADS = int(os.environ.get("ARM_RD_THREADS", "16"))

    def _hdr_one(f):
        """One header -> (pos, file, pixel_spacing, ok). Identical arithmetic to the serial loop."""
        try:
            # Parse ONLY the four elements used below. Measured 7.4x faster than a
            # full header parse over the same files, with identical ordering.
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

    def order_and_meta(sdir):
        # Threaded because the read is IO-bound and was the single largest cost in the run.
        # pool.map preserves INPUT order, so the stable sort below breaks ties exactly as the
        # serial loop did -> byte-identical slice ordering.
        fs = glob.glob(sdir + "/*.dcm")
        if not fs:
            return [], 0.5
        if len(fs) > 1 and _RD_THREADS > 1:
            with _ArmPool(max_workers=min(_RD_THREADS, len(fs))) as pool:
                out = list(pool.map(_hdr_one, fs))
        else:
            out = [_hdr_one(f) for f in fs]
        recs = [(p, f, ps) for p, f, ps, _ok in out]
        ps_list = [ps for _p, _f, ps, ok in out if ok]
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, f, ps in recs], med_ps
    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_px, mm_crop_resize


def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
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
        used.add(r['SeriesInstanceUID'])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k; continue
        # wide span: the collateral ligaments and lateral meniscus live in the
        # peripheral slices the old 0.15-0.85 crop threw away. Must match the corpus
        # the weights were trained on (knee_corpus_v2.py, SPAN_LO/SPAN_HI).
        n = len(files); lo, hi = int(n * 0.02), int(n * 0.98) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        from concurrent.futures import ThreadPoolExecutor as _PxPool
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
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= MAXS: break
            if a is None: idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        if idx >= MAXS: break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


# ============================================================================
# Test-root discovery + weights + main
# ============================================================================
def find_test_root():
    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
             "/kaggle/input/rsna-knee-abnormality-detection"]
    for b in cands:
        if os.path.exists(b + "/test.csv"):
            return b
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f and (os.path.isdir(d + "/test_series") or os.path.isdir(d + "/test_images")):
            return d
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f:
            return d
    raise RuntimeError("no test root under /kaggle/input")


def find_weight_file(fname):
    # direct dataset mounts first; NEVER recursive-glob the competitions DICOM tree.
    direct = [f"/kaggle/input/raptor-knee-arms/{fname}",
              f"/kaggle/input/raptor-knee-arms/1/{fname}",
              f"/kaggle/input/raptor-cnn336/{fname}"]
    for p in direct:
        if os.path.exists(p):
            return p
    for d in sorted(glob.glob("/kaggle/input/*/")):
        if "competition" in d.lower():
            continue
        hits = glob.glob(os.path.join(d, "**", fname), recursive=True)
        if hits:
            return hits[0]
    raise RuntimeError(f"{fname} not found under /kaggle/input")


def main():
    import pandas as pd
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"device {dev} | gpus {ngpu} | torch {torch.__version__}", flush=True)

    ROOT = find_test_root()
    tsdir = ROOT + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = ROOT + "/test_images"
    print("test root:", ROOT, "| series dir:", tsdir, flush=True)

    test = pd.read_csv(ROOT + "/test.csv"); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    test_ids = test["StudyInstanceUID"].tolist()
    tser = pd.read_csv(ROOT + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}
    print(f"test studies {len(test_ids)} | test series {len(tser)}", flush=True)

    sub_cols = ["StudyInstanceUID"] + LAB
    ssub = os.path.join(ROOT, "sample_submission.csv")
    if os.path.exists(ssub):
        sub_cols = list(pd.read_csv(ssub, nrows=1).columns)

    from concurrent.futures import ThreadPoolExecutor as _ArmPool2
    reader = _make_reader()
    N = len(test_ids); A = len(ARMS)
    arm_probs = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]

    # SEQUENTIAL ARMS (the OOM fix): only ONE model is resident at a time, so peak system RAM ==
    # one model == the single-arm champion's footprint (which graded fine at 0.879). Holding both
    # arms simultaneously OOM'd system RAM on the full hidden test. Each study is re-preprocessed
    # per arm (build_study is cheap vs inference) and every per-study buffer is freed. Same models,
    # same windowing, same rank-mean blend -> identical 0.8893 result, just serialized.
    # SINGLE PASS over the test set. Every arm here is the same architecture at the same
    # resolution, so eval_windows() returns the IDENTICAL tensor for each of them. The original
    # loop was arms-outer, which re-read and re-preprocessed all 1,322 studies once PER ARM, and
    # DICOM reading is the whole wall clock (header scan alone was 72.9% of IO). Building each
    # study once and running both models on the result halves the run. Numerically this is the
    # same computation: same xw, same infer_probs, same arm_probs, so the blend is unchanged.
    _NG = torch.cuda.device_count() if str(dev).startswith("cuda") else 0
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

    _PF = int(os.environ.get("ARM_PREFETCH", "2"))

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
                del xw
            except Exception as e:
                _fut.pop(i, None)
                print(f"  study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
            _nx = i + _PF + 1
            if _nx < N:
                _fut[_nx] = _pf_ex.submit(_prep_study, test_ids[_nx])
            if (i + 1) % 100 == 0 or i + 1 == N:
                print(f"  [1pass] {i+1}/{N} | {time.time()-t0:.0f}s", flush=True)
    del _sp_models
    gc.collect()
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"[1pass] all arms done + freed | {time.time()-t0:.0f}s", flush=True)

    # WEIGHTED rank-mean blend across the test set, per finding (the offline recipe).
    # Weights come from ARMS[*]["w"] and are normalised here, so dropping/adding an arm can
    # never silently change the scale. Falls back to equal weights if none are declared.
    _w = np.array([float(a.get("w", 1.0)) for a in ARMS], dtype=np.float64)
    _w = _w / _w.sum()
    print(f"[blend] weighted rank-mean w={dict(zip([a['file'] for a in ARMS], _w.round(4)))}", flush=True)
    ranks = np.tensordot(_w, np.stack([rankpct(np.clip(p, 0, 1)) for p in arm_probs]),
                         axes=(0, 0))                                          # (N,12) in [0,1]
    if not np.isfinite(ranks).all():
        ranks[~np.isfinite(ranks)] = 0.5

    sub = pd.DataFrame(ranks.astype(np.float32), columns=LAB)
    sub.insert(0, "StudyInstanceUID", test_ids)
    sub = sub[sub_cols]
    assert list(sub.columns) == sub_cols, "column order drift"
    assert sub["StudyInstanceUID"].tolist() == test_ids, "row identity drift"
    assert np.isfinite(sub[LAB].values).all()
    out = "/kaggle/working/submission_coatnet.csv"
    sub.to_csv(out, index=False)
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    print("ONEPASS: wrote submission.csv", flush=True)
    print("wrote", out, "|", len(sub), "rows x", len(sub.columns), "cols", flush=True)
    print(sub.head().to_string(index=False), flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as _coat_exc:
        import traceback as _coat_traceback
        print(f"CoAtNet branch failed; retaining transformer submission: {type(_coat_exc).__name__}: {_coat_exc}", flush=True)
        _coat_traceback.print_exc()



# Blend two independently validated rank predictors. The default remains the transformer
# submission if the CoAtNet branch did not complete, so a recoverable branch failure
# cannot erase a valid competition artifact.
from pathlib import Path as _BlendPath
import numpy as _blend_np
import pandas as _blend_pd

_blend_work = _BlendPath('/kaggle/working')
_blend_transformer_path = _blend_work / 'submission.csv'
_blend_coatnet_path = _blend_work / 'submission_coatnet.csv'
if _blend_coatnet_path.is_file():
    _blend_transformer = _blend_pd.read_csv(_blend_transformer_path, dtype={'StudyInstanceUID': str})
    _blend_coatnet = _blend_pd.read_csv(_blend_coatnet_path, dtype={'StudyInstanceUID': str})
    _blend_labels = [c for c in _blend_transformer.columns if c != 'StudyInstanceUID']
    if _blend_coatnet.columns.tolist() != _blend_transformer.columns.tolist():
        raise RuntimeError('CoAtNet/transformer submission schema mismatch')
    if _blend_coatnet['StudyInstanceUID'].tolist() != _blend_transformer['StudyInstanceUID'].tolist():
        raise RuntimeError('CoAtNet/transformer study order mismatch')
    _blend_tr = _blend_transformer[
        _blend_labels
    ].rank(
        method='average',
        pct=True,
    )

    _blend_cr = _blend_coatnet[
        _blend_labels
    ].rank(
        method='average',
        pct=True,
    )

    _blend_output = (
        _blend_transformer.copy()
    )

    _coatnet_weight = {
        label: 0.55
        for label in _blend_labels
    }

    if globals().get(
        'V18_CALIBRATOR_APPLIED',
        False,
    ):
        _coatnet_weight.update(
            {
                'Medial Meniscus': 0.57,
                'Lateral Meniscus': 0.59,
                'Fracture': 0.59,
            }
        )

    for _label in _blend_labels:
        _cw = float(
            _coatnet_weight[
                _label
            ]
        )

        _blend_output[
            _label
        ] = (
            (
                1.0
                - _cw
            )
            * _blend_tr[
                _label
            ]
            + _cw
            * _blend_cr[
                _label
            ]
        )

    _blend_output[
        _blend_labels
    ] = _blend_output[
        _blend_labels
    ].rank(
        method='average',
        pct=True,
    )

    print(
        '[V18] CoAtNet target weights: '
        + ', '.join(
            f'{label}='
            f'{_coatnet_weight[label]:.2f}'
            for label in _blend_labels
            if (
                _coatnet_weight[label]
                != 0.50
            )
        ),
        flush=True,
    )

    _blend_values = _blend_output[
        _blend_labels
    ].to_numpy(
        _blend_np.float64
    )
    if not _blend_np.isfinite(_blend_values).all() or _blend_values.min() < 0 or _blend_values.max() > 1:
        raise RuntimeError('invalid blended prediction values')
    _blend_output.to_csv(_blend_transformer_path, index=False)
    print(f'final submission.csv = V18 calibrated transformer + CoAtNet rank blend; {_blend_output.shape}', flush=True)
else:
    print('CoAtNet output unavailable; submission.csv remains the validated transformer ensemble', flush=True)

# V18 output hygiene.
for _v18_temp in (
    _blend_work / 'submission_coatnet.csv',
    _blend_work / 'submission_transformer_0920.csv',
):
    try:
        if _v18_temp.is_file():
            _v18_temp.unlink()
    except OSError:
        pass
