#!/usr/bin/env python3
"""All-224 kernel: N arms at 224 pixels, one resolution, built on the 29-minute base.

Why all-224 and not a 384/224 mix: the kernel prepares each study's windows ONCE and feeds the
same tensor to every arm - that is the single-pass win that got the run to 29 minutes. It asserts
"single pass needs one resolution", so a mixed-resolution blend would need windows built twice per
study. Every arm here is 224, so the assert holds untouched.

What this buys: 224 is 0.34x the pixels of 384, so an arm costs about a third as much GPU. The
29-minute ceiling caps us at TWO arms at 384; at 224 the same budget holds three or four. Our own
combotest says arm count is the lever we have never been able to pull:
    2 arms 0.9331 -> 3 arms 0.9346 -> 4 arms 0.9356
Arms are weighted equally and averaged by rank, exactly as the shipped pair is.

With more arms than GPUs the existing "one arm per GPU" placement wraps (arm a -> cuda:a%2), so
four arms sit two to a card. That is still under the clock because each is a third the cost.

Usage:
    python make_arms224.py <slug> <dataset> <arm1.pt> [arm2.pt ...] [--k N] [--title "..."]
"""
import ast, json, os, sys

argv = [a for a in sys.argv[1:]]
K = 48
TITLE = None
out = []
i = 0
while i < len(argv):
    if argv[i] == "--k":
        K = int(argv[i + 1]); i += 2
    elif argv[i] == "--title":
        TITLE = argv[i + 1]; i += 2
    else:
        out.append(argv[i]); i += 1
SLUG, DATASET, ARMS_IN = out[0], out[1], out[2:]
assert ARMS_IN, "no arms given"
assert len(set(ARMS_IN)) == len(ARMS_IN), "duplicate arm"
TITLE = TITLE or ("Knee MRI - Raptor %d Arms at 224" % len(ARMS_IN))

SRC = "variants/push_twogpu2/knee-mri-dual-gpu-decode.ipynb"
OUT_DIR = "variants/push_" + SLUG.replace("-", "_")
ARCH224 = "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k"

OLD = ('    {"file": "raptor_ft_confdist_swa.pt",          '
       '"arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},\n'
       '    {"file": "raptor_ft_coatnet_w80_full_swa.pt",  '
       '"arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.5},')
W = round(1.0 / len(ARMS_IN), 6)
NEW = "\n".join('    {"file": "%s", "arch": "%s", "res": 224, "w": %s},' % (a, ARCH224, W)
                for a in ARMS_IN)

STALE = "# SINGLE ARM: coatnet_rmlp_2_rw_384 retrained on the EXPANDED 4,349-study corpus."
FRESH = "# %d ARMS: coatnet_rmlp_2_rw_224 on the 4,349-study corpus, all at 224." % len(ARMS_IN)

nb = json.load(open(SRC, encoding="utf-8"))
hits = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    if OLD in j:
        j = j.replace(OLD, NEW, 1); hits += 1
    j = j.replace(STALE, FRESH)
    c["source"] = j.splitlines(keepends=True)
assert hits == 1, "two-arm block matched %d times (expected 1)" % hits

body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
# every requested arm wired exactly once, both 384 arms gone, nothing left at 384
for a in ARMS_IN:
    assert body.count(a) == 1, "arm not wired exactly once: %s" % a
assert "raptor_ft_confdist_swa.pt" not in body and "raptor_ft_coatnet_w80_full_swa.pt" not in body
assert '"arch": "coatnet_rmlp_2_rw_384' not in body,     "a 384 backbone is still wired as an arm; single pass needs one resolution"
assert body.count('"res": 224') == len(ARMS_IN), "not every arm is 224"
assert body.count('"w": %s' % W) == len(ARMS_IN), "weights are not equal"
# every speed win from the 29-minute kernel must survive
for marker in ("_hdr_one", "_px_one", "_prep_study", "_PARALLEL_ARMS", "[1pass]"):
    assert marker in body, "lost optimisation: %s" % marker
assert "_A5_SAVED" not in body and "_rad" + "_main" not in body, "a cut stage came back"
assert "K_EVAL = %d" % K in body, "K_EVAL is not %d" % K
assert "single pass needs one resolution" in body, "the one-resolution guard was removed"
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations",
                                               "import typing as _f"))

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("variants/push_twogpu2/kernel-metadata.json", encoding="utf-8"))
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
if DATASET not in meta["dataset_sources"]:
    meta["dataset_sources"].append(DATASET)
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)
print("built %s" % OUT_DIR)
print("  %d arms at 224, w=%s each, K_EVAL %d" % (len(ARMS_IN), W, K))
for a in ARMS_IN:
    print("    %s" % a)
print("  datasets: %s" % meta["dataset_sources"])
