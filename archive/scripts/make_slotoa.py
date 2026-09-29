#!/usr/bin/env python3
"""Give the OA findings a coronal/axial-restricted second pass.

Measured on the 58-study gold gate, our arm beats the partner half on 10 of 12 findings but loses
exactly two:
    Lateral OA   ours 0.8201  partner 0.8627
    PF OA        ours 0.8417  partner 0.8842
The kernel's own SLOT_PRIOR_TABLE says why those two are special: 'Lateral OA': (1, 4, 5) and
'Medial OA': (1, 4, 5) are CORONAL slots, and 'PF OA': (0, 2, 5) includes the AXIAL slot. The
partner half conditions on that explicitly (learned slot embeddings + per-finding slot priors at
strength 0.55). Our arm does not: build_study concatenates all five slots into one flat stack and
the attention-MIL head pools over the bag with no slot prior.

⚠ Hypothesis, not established: the head CAN distinguish sequences from appearance (axial and
sagittal look nothing alike), so the arm is not literally blind - it simply has no explicit prior.
This variant tests whether supplying one closes the gap.

Cost: exactly ONE extra pass over the arm, not a per-finding fan-out, and only the three OA findings
are overwritten. Slot identity is free - build_study fills the stack slot by slot in a fixed order
(Sagittal-FS 27, Sagittal 21, Coronal-FS 18, Coronal 12, Axial 18), so a window's sequence is
determined by its centre index.
"""
import ast, difflib, json, os

SRC = "champ/knee-mri-blend-even-full.ipynb"
OUT_DIR = "variants/push_slotoa"
SLUG = "knee-mri-coronal-axial-oa-pass"
TITLE = "Knee MRI - Coronal Axial OA Pass"

HELPER = '''
# --- slot-restricted evaluation -------------------------------------------------------------
# build_study fills `vol` slot by slot in SLOTS order, k slices each, so slot identity is a pure
# function of the slice index. These are the coronal + axial index ranges.
def _slot_index_ranges():
    out, i = {}, 0
    for name, _fluid, k in SLOTS:
        out.setdefault(name, []).append((i, i + k))
        i += k
    return out

def _coronal_axial_mask(D):
    keep = np.zeros(D, bool)
    for name, spans in _slot_index_ranges().items():
        if name in ("Coronal", "Axial"):
            for a, b in spans:
                keep[a:min(b, D)] = True
    return keep

def eval_windows_coronal_axial(vol, mask, k, res, norm=NORM):
    """Same as eval_windows but centres are drawn only from the coronal and axial slots."""
    D = vol.shape[0]
    m2 = (np.asarray(mask).astype(bool) & _coronal_axial_mask(D)).astype(np.uint8)
    if m2.sum() < 3:
        return eval_windows(vol, mask, k, res, norm)
    return eval_windows(vol, m2, k, res, norm)

OA_TARGETS = ("Lateral OA", "Medial OA", "PF OA")

'''

OLD_ARM = """            try:
                vol, mask = build_study(sid, SER, tsdir, reader)
                xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM)
                arm_probs[a][i] = infer_probs(model, xw, dev)
                del vol, mask, xw"""
NEW_ARM = """            try:
                vol, mask = build_study(sid, SER, tsdir, reader)
                xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM)
                _p = infer_probs(model, xw, dev)
                # One extra pass restricted to the coronal and axial slots, used ONLY for the three
                # osteoarthritis findings the kernel's own slot priors place on those sequences.
                xw2 = eval_windows_coronal_axial(vol, mask, k=K_EVAL, res=res, norm=NORM)
                _p2 = infer_probs(model, xw2, dev)
                for _t in OA_TARGETS:
                    _p[LAB.index(_t)] = _p2[LAB.index(_t)]
                arm_probs[a][i] = _p
                del vol, mask, xw, xw2"""

nb = json.load(open(SRC, encoding="utf-8"))
hits = {"helper": 0, "arm": 0}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    j = "".join(c["source"])
    if "def eval_windows(vol, mask, k, res, norm=NORM):" in j and not hits["helper"]:
        j = j.replace("def eval_windows(vol, mask, k, res, norm=NORM):",
                      HELPER.strip("\n") + "\n\ndef eval_windows(vol, mask, k, res, norm=NORM):", 1)
        hits["helper"] += 1
    if OLD_ARM in j:
        j = j.replace(OLD_ARM, NEW_ARM, 1); hits["arm"] += 1
    c["source"] = j.splitlines(keepends=True)
assert hits == {"helper": 1, "arm": 1}, hits
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))
print("all cells parse; anchors:", hits)

os.makedirs(OUT_DIR, exist_ok=True)
json.dump(nb, open(os.path.join(OUT_DIR, SLUG + ".ipynb"), "w", encoding="utf-8"))
meta = json.load(open("champ/kernel-metadata.json", encoding="utf-8"))
meta.pop("id_no", None)
meta.update({"id": "dreaddevelopment/" + SLUG, "title": TITLE, "code_file": SLUG + ".ipynb"})
assert meta.get("is_private") is True
json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w"), indent=1)
old = "".join("".join(c["source"]) for c in json.load(open(SRC, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
new = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
d = [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++","---")]
print(f"private OK, {len(d)} changed lines -> {OUT_DIR}")
