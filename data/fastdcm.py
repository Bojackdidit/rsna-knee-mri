"""Direct DICOM reader for the RSNA knee test set.

The competition files are Explicit VR Little Endian (1.2.840.10008.1.2.1), uncompressed,
with PixelData as the final element. So both the ordering tags and the pixels can be taken
with plain preads and a small byte scan, with no pydicom Dataset construction at all.

Falls back to pydicom for any file whose transfer syntax is not explicit VR LE, or whose
layout does not match the expected trailing-PixelData form. Returns identical values to
pydicom by construction - verified in verify_fastdcm.py.
"""
import os
import numpy as np

_EXPLICIT_LE = b"1.2.840.10008.1.2.1"
# VRs that carry a 2-byte reserved field then a 4-byte length
_LONG_VR = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN", b"OD", b"OL", b"OV", b"UC", b"UR"}

TAG_ROWS = (0x0028, 0x0010)
TAG_COLS = (0x0028, 0x0011)
TAG_BITS = (0x0028, 0x0100)
TAG_REPR = (0x0028, 0x0103)
TAG_SLOPE = (0x0028, 0x1053)
TAG_INTER = (0x0028, 0x1052)
TAG_PSPACE = (0x0028, 0x0030)
TAG_IPP = (0x0020, 0x0032)
TAG_IOP = (0x0020, 0x0037)
TAG_INST = (0x0020, 0x0013)
TAG_PIXEL = (0x7FE0, 0x0010)
TAG_PHOTO = (0x0028, 0x0004)
TAG_MODLUT = (0x0028, 0x3000)


def _u16(b, o):
    return b[o] | (b[o + 1] << 8)


def _u32(b, o):
    return b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)


def parse_header(buf):
    """Walk explicit-VR-LE elements in buf. Returns (tags, pixel_data_offset, ok).

    pixel_data_offset is the absolute file offset of the PixelData VALUE if the tag was
    reached inside buf, else None.
    """
    if len(buf) < 140 or buf[128:132] != b"DICM":
        return None, None, False
    o = 132
    n = len(buf)
    # file meta group (0002) is always explicit VR LE; read its length to find dataset start
    ts = None
    dataset_start = None
    while o + 8 <= n:
        grp = _u16(buf, o)
        elm = _u16(buf, o + 2)
        vr = buf[o + 4:o + 6]
        if vr in _LONG_VR:
            if o + 12 > n:
                break
            ln = _u32(buf, o + 8)
            vo = o + 12
        else:
            ln = _u16(buf, o + 6)
            vo = o + 8
        if grp == 0x0002 and elm == 0x0000:
            grp_len = int.from_bytes(buf[vo:vo + ln], "little")
            dataset_start = vo + ln + grp_len
        if grp == 0x0002 and elm == 0x0010:
            ts = bytes(buf[vo:vo + ln]).rstrip(b"\x00 ")
        if dataset_start is not None and o >= dataset_start:
            break
        o = vo + ln
        if dataset_start is not None and o >= dataset_start:
            break
    if ts is None or ts != _EXPLICIT_LE or dataset_start is None:
        return None, None, False

    tags = {}
    o = dataset_start
    while o + 8 <= n:
        grp = _u16(buf, o)
        elm = _u16(buf, o + 2)
        vr = buf[o + 4:o + 6]
        if not (0x41 <= vr[0] <= 0x5A and 0x41 <= vr[1] <= 0x5A):
            return tags, None, False  # not explicit VR where we expected it
        if vr in _LONG_VR:
            if o + 12 > n:
                break
            ln = _u32(buf, o + 8)
            vo = o + 12
        else:
            ln = _u16(buf, o + 6)
            vo = o + 8
        key = (grp, elm)
        if key == TAG_PIXEL:
            return tags, vo, True
        if vo + ln <= n:
            tags[key] = bytes(buf[vo:vo + ln])
        o = vo + ln
    return tags, None, False


def _dec_str(v):
    return v.decode("ascii", "ignore").strip("\x00 ")


def _floats(v):
    s = _dec_str(v)
    return [float(x) for x in s.split("\\")] if s else []


def _us(v):
    return int.from_bytes(v[:2], "little")


class Slice:
    __slots__ = ("rows", "cols", "bits", "pixrepr", "slope", "inter",
                 "spacing", "ipp", "iop", "inst", "pixel_offset", "path", "size",
                 "photo", "has_modlut", "has_rescale")


def read_meta(path, nbytes=4096):
    """Read ordering/geometry tags with a single small pread. Returns Slice or None."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        buf = fh.read(nbytes)
    tags, pxo, ok = parse_header(buf)
    if tags is None:
        return None
    s = Slice()
    s.path = path
    s.size = size
    try:
        s.rows = _us(tags[TAG_ROWS]) if TAG_ROWS in tags else None
        s.cols = _us(tags[TAG_COLS]) if TAG_COLS in tags else None
        s.bits = _us(tags[TAG_BITS]) if TAG_BITS in tags else None
        s.pixrepr = _us(tags[TAG_REPR]) if TAG_REPR in tags else 0
        s.slope = _floats(tags[TAG_SLOPE])[0] if TAG_SLOPE in tags else 1.0
        s.inter = _floats(tags[TAG_INTER])[0] if TAG_INTER in tags else 0.0
        s.spacing = _floats(tags[TAG_PSPACE]) if TAG_PSPACE in tags else []
        s.ipp = _floats(tags[TAG_IPP]) if TAG_IPP in tags else []
        s.iop = _floats(tags[TAG_IOP]) if TAG_IOP in tags else []
        s.inst = int(_dec_str(tags[TAG_INST])) if TAG_INST in tags and _dec_str(tags[TAG_INST]) else None
        s.photo = _dec_str(tags[TAG_PHOTO]) if TAG_PHOTO in tags else ""
        s.has_modlut = TAG_MODLUT in tags
        s.has_rescale = (TAG_SLOPE in tags) and (TAG_INTER in tags)
    except Exception:
        return None
    s.pixel_offset = pxo
    return s


def read_pixels(meta):
    """Read the pixel block. PixelData is the final element, so it is the file's tail."""
    if meta is None or not meta.rows or not meta.cols or meta.bits not in (8, 16):
        return None
    nbytes = meta.rows * meta.cols * (meta.bits // 8)
    off = meta.size - nbytes
    if off < 0:
        return None
    if meta.pixel_offset is not None and meta.pixel_offset != off:
        return None  # trailing elements after PixelData; fall back
    with open(meta.path, "rb") as fh:
        fh.seek(off)
        raw = fh.read(nbytes)
    if len(raw) != nbytes:
        return None
    dt = ("<i2" if meta.pixrepr else "<u2") if meta.bits == 16 else ("i1" if meta.pixrepr else "u1")
    return np.frombuffer(raw, dtype=dt).reshape(meta.rows, meta.cols)


def read_one(path):
    """Metadata + pixels from a SINGLE open/read.

    read_meta followed by read_pixels opens the file twice. These files are ~1.8 MB and the header
    is the first ~1.5 KB of the same buffer, so one read serves both. Returns (meta, array) or
    (meta_or_None, None) when the caller should fall back.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) != size:
        return None, None
    tags, pxo, ok = parse_header(buf[:8192])
    if tags is None:
        return None, None
    s = Slice()
    s.path, s.size = path, size
    try:
        s.rows = _us(tags[TAG_ROWS]) if TAG_ROWS in tags else None
        s.cols = _us(tags[TAG_COLS]) if TAG_COLS in tags else None
        s.bits = _us(tags[TAG_BITS]) if TAG_BITS in tags else None
        s.pixrepr = _us(tags[TAG_REPR]) if TAG_REPR in tags else 0
        s.slope = _floats(tags[TAG_SLOPE])[0] if TAG_SLOPE in tags else 1.0
        s.inter = _floats(tags[TAG_INTER])[0] if TAG_INTER in tags else 0.0
        s.spacing = _floats(tags[TAG_PSPACE]) if TAG_PSPACE in tags else []
        s.ipp = _floats(tags[TAG_IPP]) if TAG_IPP in tags else []
        s.iop = _floats(tags[TAG_IOP]) if TAG_IOP in tags else []
        s.inst = int(_dec_str(tags[TAG_INST])) if TAG_INST in tags and _dec_str(tags[TAG_INST]) else None
        s.photo = _dec_str(tags[TAG_PHOTO]) if TAG_PHOTO in tags else ""
        s.has_modlut = TAG_MODLUT in tags
        s.has_rescale = (TAG_SLOPE in tags) and (TAG_INTER in tags)
    except Exception:
        return None, None
    s.pixel_offset = pxo
    if not s.rows or not s.cols or s.bits not in (8, 16):
        return s, None
    nbytes = s.rows * s.cols * (s.bits // 8)
    off = size - nbytes
    if off < 0 or (pxo is not None and pxo != off):
        return s, None
    dt = ("<i2" if s.pixrepr else "<u2") if s.bits == 16 else ("i1" if s.pixrepr else "u1")
    # memoryview: slicing bytes would copy the whole 1.8 MB block before numpy sees it
    return s, np.frombuffer(memoryview(buf)[off:off + nbytes], dtype=dt).reshape(s.rows, s.cols)
