#!/usr/bin/env python3
"""Append one batch of extracted labels to the language CSV. Verifies count and range."""
import json, sys, io, os, csv
LAB=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
     "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
lang=sys.argv[1]
ids=json.load(open("_batch_ids.json",encoding="utf-8"))
rows=json.load(open("_batch_labels.json",encoding="utf-8"))
assert len(rows)==len(ids), "got %d label rows for %d reports" % (len(rows),len(ids))
for r in rows:
    assert len(r)==12, "row has %d values, need 12" % len(r)
    assert all(0.0<=v<=1.0 for v in r), "value out of [0,1]: %s" % r
p="opuslabels/%s.csv"%lang
new = not os.path.exists(p)
with io.open(p,"a",encoding="utf-8",newline="") as f:
    w=csv.writer(f)
    if new: w.writerow(["id"]+LAB)
    for i,r in zip(ids,rows): w.writerow([i]+["%.2f"%v for v in r])
n=sum(1 for _ in csv.DictReader(io.open(p,encoding="utf-8")))
print("committed %d rows -> %s (total now %d)" % (len(rows), p, n))
