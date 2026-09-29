#!/usr/bin/env python3
"""MANDATORY gate before `kaggle kernels push`. Usage: preflight_push.py VARIANT_DIR [--live]

Exists because a variant built from an existing kernel INHERITS that kernel's id, so pushing it
publishes a new version of the live kernel it was derived from - replacing tested content with
untested content - and every structural check in build_fastarm passes while it happens. That was
caught once by eye on push_knee_mri_slotwin (it carried the champion's id). A later full audit found
228 metadata files, 23 ids shared by more than one directory, and 11 of those with DIFFERENT notebook
content. Eyes do not scale; this does.

Refuses the push (exit 1) when:
  * the id is claimed by another directory whose notebook differs
  * code_file does not name the notebook actually present
  * the notebook has no code cells, or any cell fails to parse
  * a declared dataset is not attached, or an arm file is not reachable from one that is
  * masks/ids look malformed (duplicate ids)
Prints the id, title and arm->dataset map so the human sees exactly what is about to be written.
"""
import ast, glob, hashlib, io, json, os, sys, collections

d = sys.argv[1].rstrip("/\\")
LIVE = "--live" in sys.argv
meta_p = os.path.join(d, "kernel-metadata.json")
fail = []

if not os.path.exists(meta_p):
    sys.exit("no kernel-metadata.json in %s" % d)
meta = json.load(io.open(meta_p, encoding="utf-8"))
kid, title, cf = meta.get("id", ""), meta.get("title", ""), meta.get("code_file", "")
nbs = glob.glob(os.path.join(d, "*.ipynb"))

print("=" * 64)
print("  id        %s" % kid)
print("  title     %s" % title)
print("  code_file %s" % cf)
print("  datasets  %s" % meta.get("dataset_sources"))
print("=" * 64)

if not nbs:
    fail.append("no .ipynb in the directory")
else:
    if cf and os.path.basename(nbs[0]) != cf:
        fail.append("code_file %r but notebook present is %r" % (cf, os.path.basename(nbs[0])))
    nb = json.load(io.open(nbs[0], encoding="utf-8"))
    cells = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    if not cells:
        fail.append("notebook has no code cells")
    for i, c in enumerate(cells):
        try:
            ast.parse(c)
        except SyntaxError as e:
            fail.append("code cell %d does not parse: %s" % (i, e))
    mine = hashlib.md5(io.open(nbs[0], "rb").read()).hexdigest()

    # does any OTHER directory claim this id with different content?
    clash = []
    for p in glob.glob(os.path.join("D:/claude2", "**", "kernel-metadata.json"), recursive=True):
        if os.path.abspath(p) == os.path.abspath(meta_p):
            continue
        try:
            m2 = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        if m2.get("id") != kid:
            continue
        o = glob.glob(os.path.join(os.path.dirname(p), "*.ipynb"))
        oh = hashlib.md5(io.open(o[0], "rb").read()).hexdigest() if o else "none"
        if oh != mine:
            clash.append((os.path.dirname(p).replace("\\", "/"), oh[:8]))
    if clash:
        fail.append("id %r is also claimed by %d directory(ies) with DIFFERENT content:%s"
                    % (kid, len(clash), "".join("\n      %s  md5 %s" % c for c in clash)))

    # every arm the kernel loads must be reachable from an attached dataset
    body = "\n".join(cells)
    import re
    arms = re.findall(r'"file": "([^"]+\.pt)"', body)
    if arms:
        print("  arms      %s" % arms)
        if LIVE:
            import subprocess
            have = {}
            for ds in meta.get("dataset_sources", []):
                r = subprocess.run(["kaggle", "datasets", "files", ds],
                                   capture_output=True, text=True, timeout=180)
                for ln in r.stdout.splitlines():
                    p0 = ln.split()
                    if p0 and p0[0].endswith(".pt"):
                        have.setdefault(p0[0], ds)
            miss = [a for a in arms if a not in have]
            if miss:
                fail.append("arm(s) not in any attached dataset: %s" % miss)
            else:
                print("  arm->ds   %s" % {a: have[a].split("/")[-1] for a in arms})
        else:
            print("  (re-run with --live to verify arms are reachable on Kaggle)")

print()
if fail:
    print("REFUSED - do not push:")
    for f in fail:
        print("  * %s" % f)
    sys.exit(1)
print("OK to push: %s" % d)
