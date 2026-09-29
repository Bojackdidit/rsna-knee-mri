"""
A screening rig: many cheap experiments a day instead of one expensive one.

Scott, who leads this competition and does not work in machine learning, has trained about 900
models. We have trained roughly thirty. He is not out-thinking us - he is sampling the space thirty
times more densely, using a small model he can train quickly so he can keep iterating. Our runs cost
five to eleven hours each, so most of our ideas are never tested at all, and the ones that are get a
single shot.

Measured cost of going fast: ConvNeXt at 224 trains at 415 s/epoch against 1,210 at 384, so 2.9x more
experiments per hour, for 0.008 of gold. For SCREENING that trade is free, because what matters is
whether idea A beats idea B, not the absolute number.

⛔ This rig is a FILTER, never evidence for shipping. A result here earns a full-scale run, nothing
more. Geometry has already confounded one architecture comparison on this competition, and a winner
at 224 that loses at 384 is exactly the failure mode to expect.

Usage:  python _fastrig.py plan.json      # each entry is a dict of finetune_box args to override
"""
import json, subprocess, sys, time, os

BASE = dict(arch="convnext_base.fb_in22k_ft_in1k_384", res=224, ckpt="timm", norm="imagenet",
            labels="/workspace/knee/labels_confdist.parquet", bs=2, k=24, k_eval=24,
            epochs=8, bb_lr=3e-5, head_lr=1e-3, workers=4, seed=4242,
            aug="geo,erase,blur,noise", ema=0.9998,
            group_csv="/workspace/knee/study_fingerprint.csv", group_folds=5, group_fold=0,
            # The rig runs on the 44-slice corpus, which the trainer REFUSES by default because it is
            # 2.18x coarser than the kernel serves and caps arms at 0.928. That guard exists because
            # ignoring it once cost this project weeks. Opting in explicitly is correct HERE and only
            # here: the rig compares experiments against each other on one corpus and never ships.
            expect_depth=96)
FLAGS = {"grad_ckpt", "train_gold", "slot_emb"}          # presence-only arguments


def cmd(over, tag):
    a = dict(BASE); a.update(over)
    out = ["python", "-u", "finetune_box.py"]
    for k, v in a.items():
        if k in FLAGS:
            if v: out.append(f"--{k}")
        else:
            out += [f"--{k}", str(v)]
    out += ["--grad_ckpt", "--tag", tag]
    return out


def main(planfile):
    plan = json.load(open(planfile))
    print(f"{len(plan)} experiments queued, ~{BASE['epochs']} epochs each at res {BASE['res']}")
    results = []
    for i, exp in enumerate(plan):
        tag = exp.pop("tag", f"FR{i:02d}")
        t0 = time.time()
        log = f"run_{tag}.log"
        with open(log, "w") as f:
            subprocess.run(cmd(exp, tag), stdout=f, stderr=subprocess.STDOUT,
                           env={**os.environ, "KNEE_DATA_ROOT": "/workspace/knee/corpus_v11"})
        best = ""
        for line in open(log, encoding="utf-8", errors="replace"):
            if "BEST GOLD" in line:
                best = line.strip().split()[-1]
        results.append((tag, best, round((time.time() - t0) / 60, 1), json.dumps(exp)))
        print(f"  {tag:10s} {best:>8s}  {results[-1][2]:5.1f} min  {results[-1][3]}", flush=True)
        with open("fastrig_results.tsv", "a") as f:
            f.write("\t".join(str(x) for x in results[-1]) + "\n")
    print("\nranked:")
    for tag, best, mins, over in sorted(results, key=lambda r: r[1], reverse=True):
        print(f"  {best:>8s}  {tag:10s} {over}")


if __name__ == "__main__":
    main(sys.argv[1])
