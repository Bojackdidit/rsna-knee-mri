#!/usr/bin/env python3
"""Numerically verify the generated serve variants against the champion head. CPU only.

Extracts the REAL RaptorClassifier source out of each generated notebook (not a re-typed copy),
gives it a fake backbone, and checks:
  1. the refactored head reproduces the champion head exactly when no override is set;
  2. the pooling variant changes ONLY the findings named in POOL_MODE;
  3. the slot-prior variant shifts attention toward the intended planes, with correct broadcasting;
  4. the window-hygiene variant drops zero and slot-straddling centres and nothing else.
"""
import json, os, re, sys
import numpy as np
import torch

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SLOT_BUDGETS_REF = [27, 21, 18, 12, 18]
MAXS_REF = 96


def load_ns(nb_path):
    """Exec just the pieces we need out of the notebook, with a stub backbone."""
    src = "".join("".join(c["source"]) for c in json.load(open(nb_path, encoding="utf-8"))["cells"]
                  if c["cell_type"] == "code")
    ns = {"torch": torch, "np": np, "nn": torch.nn, "SLOTS": [(None, None, b) for b in SLOT_BUDGETS_REF],
          "MAXS": MAXS_REF, "TARGETS": LAB, "LAB": LAB}
    # globals block
    m = re.search(r"# ---- serve-side pooling / routing.*?return len\(SLOT_BUDGETS\) - 1", src, re.S)
    assert m, "globals block not found in " + nb_path
    exec(compile(m.group(0), nb_path + "#globals", "exec"), ns)
    # the CoAtNet RaptorClassifier (the one with clsW), not the legacy SlotHead model
    m2 = re.search(r"class RaptorClassifier\(nn\.Module\):.*?\n(?=def load_model)", src, re.S)
    assert m2, "RaptorClassifier not found in " + nb_path
    exec(compile(m2.group(0), nb_path + "#model", "exec"), ns)
    # _eval_centers
    m3 = re.search(r"def _eval_centers\(mask, D, k\):.*?\n(?=\n*def eval_windows)", src, re.S)
    assert m3, "_eval_centers not found"
    exec(compile(m3.group(0), nb_path + "#centers", "exec"), ns)
    return ns, src


def build(ns, seed=0):
    torch.manual_seed(seed)
    F = 64
    class FakeBB(torch.nn.Module):
        num_features = F
        def forward(self, x):
            return torch.randn(x.shape[0], F)
    m = ns["RaptorClassifier"](FakeBB(), F_dim=F)
    m.eval()
    return m


def champion_head(m, feats, ns):
    h = m.norm(feats)
    a = torch.softmax(m.att(h), dim=1)
    pooled = torch.einsum("bkn,bkf->bnf", a, h)
    return (pooled * m.clsW).sum(-1) + m.clsb


def main():
    root = "variants"
    V = {
        "push_knee_mri_per_finding_pooling": "knee-mri-per-finding-pooling.ipynb",
        "push_knee_mri_oa_plane_prior": "knee-mri-oa-plane-prior.ipynb",
        "push_knee_mri_window_hygiene": "knee-mri-window-hygiene.ipynb",
        "push_knee_mri_pooling_and_window_hygiene": "knee-mri-pooling-and-window-hygiene.ipynb",
    }
    torch.manual_seed(1)
    B, K, F = 3, 24, 64
    feats = torch.randn(B, K, F)
    slots = torch.tensor([i * 5 // K for i in range(K)], dtype=torch.long).clamp(0, 4)
    fails = 0

    for d, nbn in V.items():
        p = os.path.join(root, d, nbn)
        ns, src = load_ns(p)
        m = build(ns)
        pm, sp = ns["POOL_MODE"], ns["SLOT_PRIOR"]
        print("\n=== %s" % d)
        print("    POOL_MODE  : %s" % ({LAB[k]: v for k, v in pm.items()} if pm else None))
        print("    SLOT_PRIOR : %s" % ("set" if sp is not None else None))

        with torch.no_grad():
            champ = champion_head(m, feats, ns)
            got = m.head(feats, slots)
            # parity with overrides disabled
            sv_pm, sv_sp = ns["POOL_MODE"], ns["SLOT_PRIOR"]
            import builtins
            m_mod = sys.modules[m.__class__.__module__] if m.__class__.__module__ in sys.modules else None
            ns["POOL_MODE"], ns["SLOT_PRIOR"] = None, None
            m.__class__.head.__globals__["POOL_MODE"] = None
            m.__class__.head.__globals__["SLOT_PRIOR"] = None
            parity = m.head(feats, slots)
            d0 = (parity - champ).abs().max().item()
            print("    parity (overrides off) max|diff| = %.3e  %s"
                  % (d0, "OK" if d0 < 1e-4 else "FAIL"))
            fails += d0 >= 1e-4
            m.__class__.head.__globals__["POOL_MODE"] = sv_pm
            m.__class__.head.__globals__["SLOT_PRIOR"] = sv_sp
            ns["POOL_MODE"], ns["SLOT_PRIOR"] = sv_pm, sv_sp
            got = m.head(feats, slots)

        delta = (got - champ).abs().max(0).values
        changed = {LAB[i] for i in range(12) if delta[i].item() > 1e-6}
        if sv_pm:
            want = {LAB[k] for k in sv_pm}
            ok = changed == want
            print("    changed findings == POOL_MODE set : %s" % ("OK" if ok else "FAIL"))
            if not ok:
                print("      changed=%s\n      want   =%s" % (sorted(changed), sorted(want)))
                fails += 1
        elif sv_sp is not None:
            want = {"Medial OA", "Lateral OA", "PF OA"}
            ok = changed == want
            print("    changed findings == the 3 OA findings : %s" % ("OK" if ok else "FAIL"))
            if not ok:
                print("      changed=%s" % sorted(changed))
                fails += 1
            # and the shift must be TOWARD the intended slots
            h = m.norm(feats)
            a0 = torch.softmax(m.att(h), dim=1)
            a1 = torch.softmax(m.att(h) + sv_sp.index_select(1, slots).t().unsqueeze(0), dim=1)
            for lab, tgt in [("Medial OA", (2, 3)), ("Lateral OA", (2, 3)), ("PF OA", (4, 0))]:
                n = LAB.index(lab)
                sel = torch.tensor([s.item() in tgt for s in slots])
                before = a0[:, sel, n].sum(1).mean().item()
                after = a1[:, sel, n].sum(1).mean().item()
                arrow = "OK" if after > before else "FAIL"
                print("      %-12s attention on target planes %.3f -> %.3f  %s"
                      % (lab, before, after, arrow))
                fails += after <= before
        else:
            ok = len(changed) == 0
            print("    head unchanged (window-only variant) : %s" % ("OK" if ok else "FAIL"))
            fails += not ok

        # ---- window hygiene ----
        ec = ns["_eval_centers"]
        mask = np.ones(MAXS_REF, np.uint8)
        mask[66:78] = 0                       # Coronal non-fluid absent: the ~26% case
        cs = ec(mask, MAXS_REF, 48)
        zero_hits = [c for c in cs if mask[c] == 0 or mask[c - 1] == 0 or mask[c + 1] == 0]
        bounds, acc = [], 0
        for b in SLOT_BUDGETS_REF:
            bounds.append((acc, acc + b)); acc += b
        def slot(c):
            for s, (a_, b_) in enumerate(bounds):
                if a_ <= c < b_: return s
            return 4
        straddle = [c for c in cs if slot(c - 1) != slot(c) or slot(c + 1) != slot(c)]
        hygienic = "window-hygiene" in nbn or "window_hygiene" in d
        print("    centres: %d | touching-absent-slice: %d | straddling: %d  (%s)"
              % (len(cs), len(zero_hits), len(straddle),
                 "hygiene ON" if hygienic else "champion behaviour"))
        if hygienic:
            ok = (len(zero_hits) == 0 and len(straddle) == 0)
            print("    hygiene removes all bad centres : %s" % ("OK" if ok else "FAIL"))
            fails += not ok
        else:
            if len(zero_hits) == 0 and len(straddle) == 0:
                print("    NOTE unpatched kernel shows no bad centres here - check the fixture")

    print("\n%s" % ("ALL CHECKS PASS" if fails == 0 else "%d CHECK(S) FAILED" % fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
