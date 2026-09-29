#!/usr/bin/env python3
"""Who in the RSNA Knee competition is running Dread Development's data and code.

Enumerates every public notebook in the competition, then asks Kaggle which of them attach each of
the public Raptor datasets. That gives the dependency graph without downloading anything. The
dependent subset is then pulled and read, so each one can be sorted into: says where it came from,
attaches the data but never mentions it, or carries the inference code with no mention at all.

Writes audit_ecosystem.json and a readable summary to stdout.
"""
import csv, io, json, os, re, subprocess, sys, time

COMP = "rsna-knee-abnormality-detection"
OWNER = "dreaddevelopment"
OUT = os.path.dirname(os.path.abspath(__file__))
PULL = os.path.join(OUT, "eco_pull")
os.makedirs(PULL, exist_ok=True)


def kaggle(args, tries=3):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for i in range(tries):
        p = subprocess.run(["kaggle"] + args, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env)
        if p.returncode == 0:
            return p.stdout
        time.sleep(2 + 3 * i)
    return p.stdout or ""


def rows(out):
    out = "\n".join(l for l in out.splitlines() if not l.startswith("Warning"))
    try:
        return list(csv.DictReader(io.StringIO(out)))
    except Exception:
        return []


def list_paged(extra, cap=40):
    got, seen = [], set()
    for page in range(1, cap + 1):
        r = rows(kaggle(["kernels", "list"] + extra + ["--page-size", "100",
                                                       "--page", str(page), "--csv"]))
        r = [x for x in r if x.get("ref")]
        if not r:
            break
        new = [x for x in r if x["ref"] not in seen]
        for x in new:
            seen.add(x["ref"])
        got += new
        if len(r) < 100:
            break
    return got


print("enumerating every public notebook in the competition ...", flush=True)
allk = list_paged(["--competition", COMP])
print("  %d public notebooks" % len(allk), flush=True)

print("listing the public Raptor datasets ...", flush=True)
ds = [d for d in rows(kaggle(["datasets", "list", "-m", "--page-size", "100", "--csv"]))
      if d.get("ref", "").startswith(OWNER + "/")]
knee_ds = [d["ref"] for d in ds
           if re.search(r"knee|raptor", d["ref"], re.I) and "umud" not in d["ref"].lower()
           and "solar" not in d["ref"].lower()]
print("  %d candidate datasets" % len(knee_ds), flush=True)

dep = {}
for ref in knee_ds:
    users = list_paged(["--dataset", ref])
    dep[ref] = [u["ref"] for u in users]
    print("  %-46s %3d notebooks" % (ref, len(users)), flush=True)

users = sorted({k for v in dep.values() for k in v})
own = {k["ref"] for k in allk if k["ref"].startswith(OWNER + "/")}
users_ext = [u for u in users if not u.startswith(OWNER + "/")]
print("\n%d distinct notebooks attach at least one Raptor dataset (%d of them ours)"
      % (len(users), len(users) - len(users_ext)), flush=True)

comp_refs = {k["ref"] for k in allk}
in_comp = [u for u in users_ext if u in comp_refs]
out_comp = [u for u in users_ext if u not in comp_refs]
print("  %d are attached to this competition, %d are not" % (len(in_comp), len(out_comp)), flush=True)

print("\npulling source for the dependent notebooks ...", flush=True)
CREDIT = re.compile(r"dreaddevelopment|dread\s*development|johnathan\s*wagner|bojackdidit", re.I)
RAPTOR = re.compile(r"raptor", re.I)
# distinctive strings from the published inference notebook
SIG = [re.compile(p, re.I) for p in [
    r"attention\s*pool", r"k_eval", r"eval_windows", r"raptor_ft_", r"coatnet_rmlp_2_rw_384",
]]

recs = []
for i, ref in enumerate(sorted(users_ext)):
    d = os.path.join(PULL, ref.replace("/", "_"))
    if not os.path.isdir(d) or not os.listdir(d):
        os.makedirs(d, exist_ok=True)
        kaggle(["kernels", "pull", ref, "-p", d, "-m"])
    txt = ""
    for fn in os.listdir(d):
        p = os.path.join(d, fn)
        if fn.endswith(".ipynb"):
            try:
                nb = json.load(open(p, encoding="utf-8"))
                txt += "".join("".join(c.get("source", [])) for c in nb.get("cells", []))
            except Exception:
                pass
        elif fn.endswith(".py") or fn.endswith(".r") or fn.endswith(".R"):
            txt += open(p, encoding="utf-8", errors="replace").read()
    meta = {}
    mp = os.path.join(d, "kernel-metadata.json")
    if os.path.exists(mp):
        try:
            meta = json.load(open(mp, encoding="utf-8"))
        except Exception:
            pass
    srcs = [s for s in meta.get("dataset_sources", []) if s.startswith(OWNER + "/")]
    ksrc = [s for s in meta.get("kernel_sources", []) if s.startswith(OWNER + "/")]
    msrc = [s for s in meta.get("model_sources", []) if s.startswith(OWNER + "/")]
    recs.append({
        "ref": ref, "author": ref.split("/")[0], "title": meta.get("title", ""),
        "in_competition": ref in comp_refs,
        "raptor_datasets": srcs, "raptor_kernels": ksrc, "raptor_models": msrc,
        "names_him": bool(CREDIT.search(txt)),
        "says_raptor": bool(RAPTOR.search(txt)),
        "code_signature": sum(1 for s in SIG if s.search(txt)),
        "chars": len(txt),
    })
    if i % 20 == 0:
        print("  %d/%d" % (i, len(users_ext)), flush=True)

json.dump({"total_public_notebooks": len(allk), "dependency": dep, "records": recs},
          open(os.path.join(OUT, "audit_ecosystem.json"), "w"), indent=1)
print("\nwrote audit_ecosystem.json with %d dependent notebooks" % len(recs), flush=True)
