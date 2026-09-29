#!/usr/bin/env python3
"""Compare two gate dumps and REFUSE if they are not actually comparable.
Usage: _cmp_dumps.py A.npz B.npz [--expect-differ slots]

Exists because three separate checks tonight returned a number that looked like a result and was
not: a 3-series arm scored on 5 series, an override silently defeated by the checkpoint, and two
"different" probes that ran the identical configuration and agreed perfectly. Identical output from
configurations that were supposed to differ is a FAILURE SIGNAL, not a finding.
"""
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score

A, B = sys.argv[1], sys.argv[2]
expect = None
if "--expect-differ" in sys.argv:
    expect = sys.argv[sys.argv.index("--expect-differ") + 1]

def load(p):
    z = np.load(p, allow_pickle=True)
    cfg = json.loads(str(z["cfg"])) if "cfg" in z.files else None
    return z, cfg

za, ca = load(A)
zb, cb = load(B)
fail = []
if ca is None or cb is None:
    fail.append("a dump has no config stamp - regenerate it; the number cannot be interpreted")
else:
    print("  A %s" % ca)
    print("  B %s" % cb)
    if expect:
        if ca.get(expect) == cb.get(expect):
            fail.append("%s is IDENTICAL in both (%r) - the comparison did not run" % (expect, ca.get(expect)))
    diff = [k for k in ca if ca[k] != cb.get(k)]
    print("  differing keys: %s" % (diff or "NONE"))
    if not diff:
        fail.append("the two runs used the SAME configuration - any agreement is meaningless")

ia, ib = za["ids"].astype(str), zb["ids"].astype(str)
common = sorted(set(ia) & set(ib))
if not common:
    fail.append("no shared studies - not comparable")
else:
    def sc(z, ids):
        idx = {u: i for i, u in enumerate(z["ids"].astype(str))}
        k = [idx[u] for u in ids]
        P, Y = z["pred"][k], z["truth"][k]
        a = [roc_auc_score((Y[:, j] >= 0.5).astype(int), P[:, j]) for j in range(Y.shape[1])
             if len(set(((Y[:, j] >= 0.5).astype(int)).tolist())) > 1]
        return float(np.mean(a))
    va, vb = sc(za, common), sc(zb, common)
    print("  on %d shared studies:  A %.4f   B %.4f   delta %+.4f" % (len(common), va, vb, vb - va))
    if abs(va - vb) < 1e-9 and not fail:
        fail.append("scores are bit-identical - almost certainly the same run twice")

print()
if fail:
    print("REFUSED - this comparison is not valid:")
    for f in fail:
        print("  * %s" % f)
    sys.exit(1)
print("comparison is valid")
