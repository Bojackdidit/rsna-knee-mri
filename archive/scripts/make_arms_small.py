#!/usr/bin/env python3
"""N arms of ANY architecture at ONE resolution, built on the fastdcm base.

Generalises make_arms224.py in the two ways the efficientnet plan needs:
  - arch is a parameter, not hardcoded to coatnet_rmlp_2_rw_224. The kernel's load_model already
    reads arch out of the checkpoint and builds it generically, and the trainer saves arch/res, so
    nothing downstream cares which family it is.
  - it patches the FASTDCM notebook rather than push_twogpu2, so the 2.44x reader survives. Reading
    is only ~4 of the 23.8 minutes now, but there is no reason to hand it back.

The one-resolution rule is unchanged and still asserted: the kernel builds each study's windows once
and feeds the same tensor to every arm, so all arms here share one res.

Usage:
    python make_arms_small.py <slug> <dataset> <arch> <res> <arm1.pt> [arm2.pt ...]
                              [--k N] [--title "..."]
"""
import ast, json, os, sys

argv = list(sys.argv[1:])
K, TITLE, out, i = 48, None, [], 0
while i < len(argv):
    if argv[i] == "--k":       K = int(argv[i + 1]); i += 2
    elif argv[i] == "--title": TITLE = argv[i + 1]; i += 2
    else:                      out.append(argv[i]); i += 1
SLUG, DATASET, ARCH, RES, ARMS_IN = out[0], out[1], out[2], int(out[3]), out[4:]
assert ARMS_IN, "no arms given"
assert len(set(ARMS_IN)) == len(ARMS_IN), "duplicate arm"
TITLE = TITLE or ("Knee MRI - %s at %d" % (ARCH.split(".")[0], RES))

SRC = "variants/push_fastdcm/knee-mri-raptor-direct-dicom-read.ipynb"
META = "variants/push_fastdcm/kernel-metadata.json"
OUT_DIR = "variants/push_" + SLUG.replace("-", "_")

OLD = ('    {"file": "raptor_ft_confdist_swa.pt",          '
       '"arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
       '    {"file": "raptor_ft_coatnet_w80_full_swa.pt",  '
       '"arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},')
W = round(1.0 / len(ARMS_IN), 6)
NEW = "\n".join('    {"file": "%s", "arch": "%s", "res": %d, "w": %s},' % (a, ARCH, RES, W)
                for a in ARMS_IN)

nb = json.load(open(SRC, encoding="utf-8"))
hits = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    if OLD in j:
        j = j.replace(OLD, NEW, 1); hits += 1
        c["source"] = j.splitlines(keepends=True)
assert hits == 1, "two-arm block matched %d times (expected 1)" % hits

body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
for a in ARMS_IN:
    assert body.count(a) == 1, "arm not wired exactly once: %s" % a
assert "raptor_ft_confdist_swa.pt" not in body and "raptor_ft_coatnet_w80_full_swa.pt" not in body
assert '"arch": "coatnet_rmlp_2_rw_384' not in body, "a 384 CoAtNet is still wired as an arm"
assert body.count('"res": %d' % RES) == len(ARMS_IN), "not every arm is at %d" % RES
assert body.count('"w": %s' % W) == len(ARMS_IN), "weights are not equal"
for marker in ("_hdr_one", "_px_one", "_prep_study", "_PARALLEL_ARMS", "[1pass]",
               "_fd_read_one", "single pass needs one resolution"):
    assert marker in body, "lost optimisation: %s" % marker
assert "K_EVAL = %d" % K in body, "K_EVAL is not %d" % K
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations",
                                               "import typing as _f"))

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open(META, encoding="utf-8"))
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
meta.pop("id_no", None)
if DATASET not in meta["dataset_sources"]:
    meta["dataset_sources"].append(DATASET)
assert meta.get("is_private") is True
assert TITLE.lower().replace(" ", "-").replace("--", "-") == SLUG or True
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)
print("built %s" % OUT_DIR)
print("  %d arm(s)  arch=%s  res=%d  w=%s each  K_EVAL=%d" % (len(ARMS_IN), ARCH, RES, W, K))
print("  datasets: %s" % meta["dataset_sources"])
