#!/usr/bin/env python3
"""Emit one batch of reports for extraction, and record what has already been done.

Checkpointed: every completed batch is appended to opuslabels/<lang>.csv immediately, so a
compaction or crash costs at most the batch in flight, never the whole language.
"""
import json, sys, io, os, csv
lang=sys.argv[1]; start=int(sys.argv[2]); n=int(sys.argv[3])
d=json.load(open("_by_language.json",encoding="utf-8"))[lang]
done=set()
p="opuslabels/%s.csv"%lang
if os.path.exists(p):
    for r in csv.DictReader(io.open(p,encoding="utf-8")): done.add(r["id"])
todo=[x for x in d if x["id"] not in done]
print("LANG %s | total %d | done %d | remaining %d" % (lang,len(d),len(done),len(todo)), file=sys.stderr)
out=todo[start:start+n]
with io.open("_batch_out.txt","w",encoding="utf-8") as f:
    for i,x in enumerate(out):
        f.write("### %d | %s\n%s\n\n" % (i, x["id"][-14:], x["report"].replace("\n"," ")))
json.dump([x["id"] for x in out], io.open("_batch_ids.json","w",encoding="utf-8"))
print("emitted %d reports -> _batch_out.txt" % len(out), file=sys.stderr)
