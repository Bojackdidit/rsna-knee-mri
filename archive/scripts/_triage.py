import csv, glob, io, json, os
FLAG = [("6843879156","Effusion"),("6255028614","ACL"),("6728385881","Baker's"),
        ("5381023947","Medial Meniscus"),("6428547156","Baker's"),("9895503197","Baker's"),
        ("3173225758","Baker's"),("7222109323","Baker's"),("6724730305","Lateral Meniscus")]
rep={}
for lang,rows in json.load(io.open("_by_language.json",encoding="utf-8")).items():
    for r in rows: rep[r["id"]]=(lang,r["report"])
lab={}
for p in glob.glob("opuslabels/*.csv"):
    for row in csv.DictReader(io.open(p,encoding="utf-8")): lab[row["id"]]=row
for suf,fin in FLAG:
    fid=[i for i in rep if i.endswith(suf)]
    if not fid: print("NOT FOUND",suf); continue
    i=fid[0]; lang,t=rep[i]
    print("="*100)
    print("%s  [%s]  %s = %s" % (suf, lang, fin, lab.get(i,{}).get(fin,"?")))
    print(t[:1400].replace("\n"," "))
