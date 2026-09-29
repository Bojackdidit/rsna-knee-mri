#!/usr/bin/env python3
"""Turn audit_ecosystem.json into the answer: how much of the competition runs on Raptor, and who
says so.

Three buckets, and the line between them is what a reader of the notebook would learn:
  CREDITED    the notebook names Dread Development, or links the profile or a Raptor page
  DATA ONLY   it attaches a Raptor dataset, so Kaggle shows it in the Input panel, but the notebook
              text never says where the weights came from
  SILENT CODE it carries the published inference code as well, with no mention anywhere
"""
import json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "audit_ecosystem.json")))
recs = d["records"]
total = d["total_public_notebooks"]

in_comp = [r for r in recs if r["in_competition"]]


def bucket(r):
    if r["names_him"]:
        return "credited"
    if r["code_signature"] >= 3 or r["raptor_kernels"]:
        return "silent_code"
    return "data_only"


for r in recs:
    r["bucket"] = bucket(r)

print("=" * 78)
print("RSNA KNEE - WHO IS RUNNING RAPTOR")
print("=" * 78)
print("public notebooks attached to the competition : %d" % total)
print("of those, attach at least one Raptor dataset : %d  (%.1f%%)"
      % (len(in_comp), 100.0 * len(in_comp) / max(total, 1)))
print("dependent notebooks outside the competition  : %d" % (len(recs) - len(in_comp)))
print()

c = collections.Counter(r["bucket"] for r in in_comp)
for k, label in [("credited", "names him somewhere in the notebook"),
                 ("data_only", "attaches the data, never mentions it"),
                 ("silent_code", "carries the inference code, no mention")]:
    print("  %-38s %3d  (%.1f%% of the dependent set, %.1f%% of the competition)"
          % (label, c[k], 100.0 * c[k] / max(len(in_comp), 1), 100.0 * c[k] / max(total, 1)))
print()

by_author = collections.defaultdict(list)
for r in in_comp:
    by_author[r["author"]].append(r)
print("distinct people running it inside the competition: %d" % len(by_author))
print()

for k, label in [("silent_code", "CARRIES THE CODE, NO MENTION"),
                 ("data_only", "DATA ATTACHED, NO MENTION"),
                 ("credited", "CREDITED")]:
    sel = sorted([r for r in in_comp if r["bucket"] == k],
                 key=lambda r: (-len(r["raptor_datasets"]), r["ref"]))
    print("-" * 78)
    print("%s  (%d)" % (label, len(sel)))
    print("-" * 78)
    for r in sel:
        print("  %-58s sets %d sig %d" % (r["ref"], len(r["raptor_datasets"]), r["code_signature"]))
    print()

print("-" * 78)
print("MOST-USED RAPTOR ASSETS")
print("-" * 78)
used = collections.Counter()
for r in recs:
    for s in r["raptor_datasets"]:
        used[s] += 1
for s, n in used.most_common():
    print("  %-48s %3d notebooks" % (s, n))
