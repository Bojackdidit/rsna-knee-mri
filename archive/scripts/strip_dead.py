#!/usr/bin/env python3
"""Remove every function the kernel never reaches. Usage: strip_dead.py SRC_DIR OUT_DIR

Audit of the base notebook: 59 top-level functions, **29 reachable, 30 dead - 635 of 1,849 lines,
34.3%**. All of it is the inherited DINOv2 / legacy-second-family pipeline: `run_dinov2` is never
called, and everything only it reached (`infer_from_package` at 110 lines, `build_cache` at 71,
`predict_member` at 47, `read_slot` at 51, ...) is dead with it. That dead half is what dragged in the
third-party mounts and the foreign framing.

Reachability is computed properly: roots are the module-level statements and class bodies, then the
call graph is walked from there. A function is removed only if nothing reachable calls it.

⚠ The proof is not the analysis - it is that the rebuilt kernel produces a BYTE-IDENTICAL submission.
Anything referenced dynamically (by string, getattr) would not be caught by an AST walk, so the
commit-run diff is the real check and must be run before this ships.
"""
import ast, io, json, os, sys, glob

SRC, OUT = sys.argv[1], sys.argv[2]
src_nb = glob.glob(os.path.join(SRC, "*.ipynb"))[0]
nb = json.load(io.open(src_nb, encoding="utf-8"))

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
code = "\n".join("".join(c["source"]) for c in code_cells)
tree = ast.parse(code.replace("from __future__ import annotations", "import typing as _f"))

funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def names_in(node):
    out = set()
    for x in ast.walk(node):
        if isinstance(x, ast.Call):
            f = x.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
        elif isinstance(x, ast.Name):
            out.add(x.id)
        elif isinstance(x, ast.Constant) and isinstance(x.value, str):
            # a name mentioned in a string could be a dynamic reference - treat as live
            for fn in funcs:
                if fn in x.value:
                    out.add(fn)
    return out


roots = set()
for n in tree.body:
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)):
        continue
    roots |= names_in(n)

live, stack = set(), [r for r in roots if r in funcs]
while stack:
    f = stack.pop()
    if f in live:
        continue
    live.add(f)
    for c in names_in(funcs[f]):
        if c in funcs and c not in live:
            stack.append(c)

dead = {f for f in funcs if f not in live}
print("functions %d | reachable %d | dead %d" % (len(funcs), len(live), len(dead)))

# rebuild each code cell without the dead defs, cell by cell so line numbers stay local
removed_lines = 0
for c in code_cells:
    s = "".join(c["source"])
    try:
        t = ast.parse(s.replace("from __future__ import annotations", "import typing as _f"))
    except SyntaxError:
        continue
    drops = []
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in dead:
            # n.lineno points at the `def`, NOT at any decorator above it - take the earliest
            # decorator line or the decorator would be left dangling and break the parse.
            start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
            drops.append((start - 1, n.end_lineno))
    if not drops:
        continue
    lines = s.split("\n")
    keep = [True] * len(lines)
    for a, b in drops:
        for i in range(a, min(b, len(lines))):
            keep[i] = False
        removed_lines += b - a
    new = "\n".join(l for l, k in zip(lines, keep) if k)
    c["source"] = new.splitlines(keepends=True)

body = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]).replace("from __future__ import annotations", "import typing as _f"))
print("removed %d lines (%d -> %d, -%.1f%%)"
      % (removed_lines, len(code.split("\n")), len(body.split("\n")),
         100.0 * removed_lines / len(code.split("\n"))))
for t in ("tonylica", "romantamrazov", "prvsiyan", "dinov3", "DINOsaur", "Head and shoulders"):
    if t in body:
        print("  STILL PRESENT: %s x%d" % (t, body.count(t)))

os.makedirs(OUT, exist_ok=True)
name = os.path.basename(src_nb)
json.dump(nb, io.open(os.path.join(OUT, name), "w", encoding="utf-8"))
meta = json.load(io.open(os.path.join(SRC, "kernel-metadata.json"), encoding="utf-8"))
json.dump(meta, io.open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", os.path.join(OUT, name))
