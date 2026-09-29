#!/usr/bin/env python3
"""Build the champion kernel with the direct DICOM reader swapped in.

Takes knee-mri-dual-gpu-decode (raptor_ft_confdist_swa + raptor_ft_coatnet_w80_full_swa, the 0.941
/ 29.1 min entry) and replaces only the two functions that touch DICOM bytes. Everything else -
arms, weights, windowing, blend - is untouched, so the score is unchanged by construction and only
the runtime moves. Both replacements keep the original pydicom implementation as a fallback.
"""
import io, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "variants", "push_twogpu2")
SRC_NB = os.path.join(SRC_DIR, "knee-mri-dual-gpu-decode.ipynb")
RD_THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = leave the champion default (16)
SLUG = "knee-mri-raptor-direct-dicom-read"
TITLE = "Knee MRI - Raptor Direct DICOM Read"
OUT_DIR = os.path.join(HERE, "variants", "push_fastdcm")

FASTDCM = io.open(os.path.join(HERE, "fastdcm.py"), encoding="utf-8").read()

OLD_HDR_SIG = "    def _hdr_one(f):"
OLD_PX = '''    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a
'''

NEW_HDR = '''    def _hdr_one(f):
        """One header -> (pos, file, pixel_spacing, ok). Identical arithmetic to the serial loop."""
        try:
            # Direct explicit-VR-LE scan of the same four elements. The competition files are all
            # uncompressed explicit VR little endian, so the tags can be taken from a single 4 KB
            # pread. Verified byte-identical to the pydicom parse (ordering key, pixel spacing,
            # ok-flag) over a real series; falls back to pydicom for anything it declines.
            _t0 = _fd_clock()
            m = _fd_read_meta(f)
            if m is None:
                _fd_note('meta', False, _fd_clock() - _t0)
                return _hdr_one_pydicom(f)
            iop, ipp = m.iop, m.ipp
            if iop and ipp and len(iop) == 6:
                r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(m.inst or 0)
            ps = float(m.spacing[0]) if m.spacing else 0.5
            _fd_note('meta', True, _fd_clock() - _t0)
            return (pos, f, ps, True)
        except Exception:
            return _hdr_one_pydicom(f)

'''

NEW_PX = '''    def read_px(f):
        # PixelData is the final element and uncompressed, so the pixels are the file's tail and
        # come back with one seek+read. When slope==1 and intercept==0 the float64 multiply inside
        # apply_modality_lut is the identity and uint16 -> float32 is exact (65535 < 2**24), so the
        # array is the same one, built with fewer copies. Anything unusual falls back to pydicom.
        try:
            _t0 = _fd_clock()
            # one open serves both: the header is the head of the same buffer the pixels end
            m, a = _fd_read_one(f)
            if m is None or a is None or m.has_modlut:
                _fd_note('px', False, _fd_clock() - _t0)
                return _read_px_pydicom(f)
            if m.has_rescale and not (m.slope == 1.0 and m.inter == 0.0):
                a = (a.astype(np.float64) * m.slope + m.inter).astype(np.float32)
            else:
                a = a.astype(np.float32)
            if m.photo == 'MONOCHROME1':
                a = a.max() - a
            _fd_note('px', True, _fd_clock() - _t0)
            return a
        except Exception:
            return _read_px_pydicom(f)

'''

nb = json.load(io.open(SRC_NB, encoding="utf-8"))
target = None
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "def _make_reader():" in "".join(cell["source"]):
        target = i
        break
if target is None:
    sys.exit("could not find the _make_reader cell")

src = "".join(nb["cells"][target]["source"])
if OLD_HDR_SIG not in src or OLD_PX not in src:
    sys.exit("reader text did not match the expected champion source")

# keep the originals as fallbacks, then install the fast versions
src = src.replace(OLD_HDR_SIG, "    def _hdr_one_pydicom(f):", 1)
src = src.replace(OLD_PX, OLD_PX.replace("    def read_px(f):", "    def _read_px_pydicom(f):", 1), 1)
# insert new definitions immediately before order_and_meta so both fallbacks already exist
anchor = "    def order_and_meta(sdir):"
if anchor not in src:
    sys.exit("order_and_meta anchor missing")
src = src.replace(anchor, NEW_HDR + NEW_PX + anchor, 1)
if "    return order_and_meta, read_px, mm_crop_resize" not in src:
    sys.exit("reader return anchor missing")
src = src.replace("    return order_and_meta, read_px, mm_crop_resize", '    def _timed_order(sdir):\n        # wall clock, not cumulative: this is the number that moves when threads change\n        _t0 = _fd_clock()\n        try:\n            return order_and_meta(sdir)\n        finally:\n            with _FD_LOCK:\n                _FD["t_order"] += _fd_clock() - _t0\n                _FD["n_order"] += 1\n\n    return _timed_order, read_px, mm_crop_resize', 1)
if RD_THREADS:
    old_t = '_RD_THREADS = int(os.environ.get("ARM_RD_THREADS", "16"))'
    if old_t not in src:
        sys.exit("thread-count anchor missing")
    src = src.replace(old_t, '_RD_THREADS = int(os.environ.get("ARM_RD_THREADS", "%d"))' % RD_THREADS, 1)

nb["cells"][target]["source"] = src.splitlines(keepends=True)

# fastdcm module, exposed under _fd_* names, in its own cell just before the reader
COUNTERS = """

# --- diagnostics: prove which path actually ran, and price the reader on Kaggle's disk ---
import atexit as _fd_atexit, threading as _fd_threading
from time import perf_counter as _fd_clock

_fd_read_meta = read_meta
_fd_read_pixels = read_pixels
_fd_read_one = read_one
_FD = {"meta_fast": 0, "meta_fb": 0, "px_fast": 0, "px_fb": 0, "t_meta": 0.0, "t_px": 0.0,
       "t_order": 0.0, "n_order": 0}
_FD_LOCK = _fd_threading.Lock()


def _fd_note(kind, fast, dt):
    with _FD_LOCK:
        _FD[kind + ("_fast" if fast else "_fb")] += 1
        _FD["t_" + kind] += dt


@_fd_atexit.register
def _fd_report():
    print("[fastdcm] meta fast=%d fallback=%d | px fast=%d fallback=%d | cum: meta %.1fs px %.1fs"
          " | ORDER-PASS WALL %.2fs over %d series | threads=%s"
          % (_FD["meta_fast"], _FD["meta_fb"], _FD["px_fast"], _FD["px_fb"],
             _FD["t_meta"], _FD["t_px"], _FD["t_order"], _FD["n_order"],
             os.environ.get("ARM_RD_THREADS", "16")), flush=True)
"""

mod = FASTDCM + COUNTERS
nb["cells"].insert(target, {"cell_type": "code", "execution_count": None,
                            "metadata": {}, "outputs": [], "source": mod.splitlines(keepends=True)})

# atexit fires after Kaggle stops capturing output, so report from a final cell instead
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                    "source": ["_fd_report()\n"]})

os.makedirs(OUT_DIR, exist_ok=True)
out_nb = os.path.join(OUT_DIR, SLUG + ".ipynb")
json.dump(nb, io.open(out_nb, "w", encoding="utf-8"), ensure_ascii=False)

meta = json.load(io.open(os.path.join(SRC_DIR, "kernel-metadata.json"), encoding="utf-8"))
meta["id"] = "dreaddevelopment/" + SLUG
meta["title"] = TITLE
meta["code_file"] = SLUG + ".ipynb"
meta.pop("id_no", None)
json.dump(meta, io.open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8"), indent=1)

chk = "".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
print("built:", out_nb)
for name, cond in [
    ("fastdcm embedded",        "def parse_header(buf)" in chk),
    ("_fd_read_meta alias",     "_fd_read_meta = read_meta" in chk),
    ("diagnostics wired",       "[fastdcm] meta fast=" in chk and "_fd_note('px', True" in chk),
    ("pydicom hdr fallback",    "def _hdr_one_pydicom(f):" in chk),
    ("pydicom px fallback",     "def _read_px_pydicom(f):" in chk),
    ("fast _hdr_one installed", "m = _fd_read_meta(f)" in chk),
    ("fast read_px installed",  "m, a = _fd_read_one(f)" in chk),
    ("read_one aliased",        "_fd_read_one = read_one" in chk),
    ("arms untouched",          "raptor_ft_coatnet_w80_full_swa.pt" in chk and "raptor_ft_confdist_swa.pt" in chk),
    ("K_EVAL untouched",        "K_EVAL = 48" in chk),
    ("final report cell",       chk.rstrip().endswith("_fd_report()")),
]:
    print("  %-26s %s" % (name, "OK" if cond else "FAIL"))
