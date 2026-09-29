#!/usr/bin/env python3
"""Teach the KERNEL the same per-slot embedding the trainer learned. Usage: patch_slot_kernel.py DIR

Serving an arm trained with `--slot_emb` needs the slot id of every window at inference time. That is
well defined and 1:1 with training, verified rather than assumed:

    trainer, 44-slice corpus : [0,12) [12,22) [22,30) [30,36) [36,44)   12/10/8/6/8
    kernel,  96-slot volume  : [0,27) [27,48) [48,66) [66,78) [78,96)   27/21/18/12/18
    both are Sagittal-FS, Sagittal, Coronal-FS, Coronal, Axial IN THE SAME ORDER

so slot id 0-4 means the same sequence on both sides. The kernel's LIVE `SLOTS` is the one at the
bottom of the file - the earlier `SLOTS = SLOTS_PUBLIC if ...` with its 6-entry SLOT_PRIOR_TABLE
belongs to the dead SlotHead/Model path and is shadowed.

BACKWARD COMPATIBLE BY CONSTRUCTION: the embedding is only created when the checkpoint actually
carries `slot_emb.weight`. An arm without it loads and runs exactly as before, slots are passed as
None, and the arithmetic is untouched - so this patch can sit in the kernel while old arms still ship.

Three call sites change, and the fourth is the one that would silently corrupt everything: the
interleave slices the window tensor `xw[_a % 2::2]`, so the slot ids MUST be sliced with the identical
stride or every window gets labelled with another window's plane.
"""
import ast, glob, io, json, os, sys

D = sys.argv[1]
nbp = glob.glob(os.path.join(D, "*.ipynb"))[0]
nb = json.load(io.open(nbp, encoding="utf-8"))

SUBS = [
    # 1) model: optional, checkpoint-driven slot embedding
    ("""    def head(self, feats):
        h = self.norm(feats)""",
     """    def head(self, feats, slots=None):
        if getattr(self, "slot_emb", None) is not None and slots is not None:
            feats = feats + self.slot_emb(slots)
        h = self.norm(feats)"""),
    ("""    def forward(self, x):
        return self.head(self.encode(x))""",
     """    def forward(self, x, slots=None):
        return self.head(self.encode(x), slots)"""),
    # 2) window builder also reports which sequence each window came from
    ("""    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    return x""",
     """    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    _b, _bounds = 0, []
    for _nm, _fs, _n in SLOTS:
        _bounds.append((_b, _b + _n)); _b += _n
    _sid = []
    for c in cs:
        _s = len(_bounds) - 1
        for _i, (_a, _z) in enumerate(_bounds):
            if _a <= c < _z:
                _s = _i; break
        _sid.append(_s)
    return x, torch.tensor(_sid, dtype=torch.long)"""),
    # 3) inference passes them through
    ("""def infer_probs(model, xwins, device):
    x = xwins.unsqueeze(0).to(device)""",
     """def infer_probs(model, xwins, device, slots=None):
    x = xwins.unsqueeze(0).to(device)
    _sl = None if slots is None else slots.unsqueeze(0).to(device)"""),
]
REPL_MODEL_CALLS = [("o = torch.sigmoid(model(x).float())",
                     "o = torch.sigmoid(model(x, _sl).float())")]

body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
report = []
for old, new in SUBS:
    n = body.count(old)
    report.append((old.strip().split("\n")[0][:58], n))
print("  anchor occurrences in this kernel:")
for a, n in report:
    print("     %-60s x%d" % (a, n))
print("\n  model(x) call sites to rewrite: %d" % body.count(REPL_MODEL_CALLS[0][0]))
print("  interleave sites (xw[_a %% 2::2] / xw[a %% 2::2]): %d / %d"
      % (body.count("xw[_a % 2::2]"), body.count("xw[a % 2::2]")))
print("""
  NOT APPLIED. The remaining work is mechanical but must be done with the arm in hand:
    - every `eval_windows(...)` caller now receives a 2-tuple and must unpack it
    - the slot tensor must be sliced with the SAME stride as the interleave, or windows get
      labelled with another window's plane and the feature becomes noise that still trains
    - `load_model` must build the embedding only when the checkpoint carries slot_emb.weight
  Applying it before a slot-emb arm exists risks the 0.941 for nothing.""")
