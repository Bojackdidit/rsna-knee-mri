#!/usr/bin/env python3
"""Post-train a finished arm on the REAL labels. Usage: _gold_finetune.py ARM.pt OUT.pt [HOLDOUT]

The 58 gold studies are the only hard ground truth in the whole corpus, and the trainer currently
EXCLUDES them from training (`train_ids = [u for u in ids if ... u not in goldset]`) to use them as a
validation set we have already decided not to trust. So real labels sit idle while the model learns
entirely from soft labels distilled out of radiology reports.

Why it should move the score: AUC is a ranking metric. If the distilled labels systematically
mis-rank a finding — call it late, or confuse two findings that read similarly in a report — the model
inherits that bias and nothing in training can correct it. The golds are the only signal that can.

Why HEAD-ONLY: 58 studies against a full backbone is an invitation to overfit. The head here is small
(LayerNorm + a 2-layer attention MLP + a 12x F_dim classifier), the backbone is frozen, the LR is
low, and a holdout of the golds is kept to early-stop on. That makes this a re-ranking of features the
arm already learned, not a re-learning of the features.

Reports the holdout AUC before and after so the change is measured, not assumed.
"""
import os, sys, json, copy
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import finetune_box as F

ARM = sys.argv[1]
OUT = sys.argv[2]
HOLD = int(sys.argv[3]) if len(sys.argv) > 3 else 14
EPOCHS = int(os.environ.get("GF_EPOCHS", "40"))
LR = float(os.environ.get("GF_LR", "2e-4"))
SEED = int(os.environ.get("GF_SEED", "0"))

DATA, LAB, RSNA = F.DATA, F.LAB, F.RSNA
ids = np.load(os.path.join(DATA, "all_ids.npy"), allow_pickle=True).astype(str)
id2row = {u: i for i, u in enumerate(ids)}
tr = pd.read_csv(os.path.join(RSNA, "train.csv"))
tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
gold_ids = [u for u in gold_df.index if u in set(ids)]
labels = {u: gold_df.loc[u, LAB].values.astype(np.float32) for u in gold_ids}
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(gold_ids))
hold = [gold_ids[i] for i in perm[:HOLD]]
fit = [gold_ids[i] for i in perm[HOLD:]]
print("gold studies %d -> fine-tune on %d, hold out %d" % (len(gold_ids), len(fit), len(hold)), flush=True)

ck = torch.load(ARM, map_location="cpu", weights_only=False)
arch, res = ck.get("arch"), int(ck.get("res", 384))
K = int(os.environ.get("GF_K", "24"))
bb = F.build_backbone(arch, pretrained=False)
model = F.RaptorClassifier(bb, F_dim=bb.num_features)
model.load_state_dict(ck["model"], strict=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev)
print("loaded %s | %s @ %d | k_eval %d" % (os.path.basename(ARM), arch, res, K), flush=True)

# freeze the backbone: this re-ranks features the arm already has, it does not relearn them
for p in model.backbone.parameters():
    p.requires_grad = False
head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
print("trainable head params: %s" % ("{:,}".format(sum(p.numel() for p in head))), flush=True)


def loader(sub, train):
    ds = F.StudyWindows(DATA, sub, id2row, labels, res, K, train=train, aug=train, norm="imagenet")
    return DataLoader(ds, batch_size=2, shuffle=train, num_workers=6, collate_fn=F.collate)


def evaluate(dl):
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for x, y in dl:
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = torch.sigmoid(model(x.to(dev)).float())
            P.append(o.cpu().numpy()); Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    a = []
    for j in range(Y.shape[1]):
        yb = (Y[:, j] >= 0.5).astype(int)
        if len(set(yb.tolist())) > 1:
            a.append(roc_auc_score(yb, P[:, j]))
    return float(np.mean(a)), len(a)


hl, fl = loader(hold, False), loader(fit, True)
base, nj = evaluate(hl)
print("holdout AUC BEFORE post-training: %.4f (%d findings scorable)" % (base, nj), flush=True)

opt = torch.optim.AdamW(head, lr=LR, weight_decay=1e-4)
lossf = torch.nn.BCEWithLogitsLoss()
best, best_state, bad = base, copy.deepcopy(model.state_dict()), 0
for ep in range(EPOCHS):
    model.train()
    model.backbone.eval()          # keep frozen BN/stats fixed
    tot = 0.0
    for x, y in fl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            loss = lossf(model(x).float(), y)
        loss.backward(); opt.step(); tot += float(loss)
    au, _ = evaluate(hl)
    flag = ""
    if au > best:
        best, bad = au, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        flag = " *"
    else:
        bad += 1
    print("  ep%-2d loss %.4f holdout %.4f%s" % (ep, tot / max(len(fl), 1), au, flag), flush=True)
    if bad >= 8:
        print("  early stop (8 epochs without improvement)", flush=True)
        break

print("\nholdout AUC  before %.4f  ->  after %.4f  (%+.4f)" % (base, best, best - base), flush=True)
if best <= base:
    print("NO GAIN - not writing a checkpoint. Post-training on golds does not help this arm.", flush=True)
    sys.exit(0)
out = dict(ck)
out["model"] = best_state
out["gold_finetuned"] = {"before": base, "after": best, "n_fit": len(fit), "n_hold": len(hold),
                         "lr": LR, "head_only": True}
torch.save(out, OUT)
print("wrote %s" % OUT, flush=True)
