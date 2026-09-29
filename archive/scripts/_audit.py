#!/usr/bin/env python3
"""Cross-check every extracted label against unambiguous statements in its own report.

The 4,349 training studies have no ground truth, so nothing catches a mis-set label except reading
it again. This does the re-reading mechanically: for each finding it looks for phrases that are
unambiguous in the source language ("ACL intact", "rupture of the ACL") and flags any row where the
number contradicts the report's own words. It also flags reports that are not knee MRIs at all.

Deliberately no \b word boundaries anywhere: escape handling mangled them repeatedly and a
backspace character silently disabled the whole negation guard. Plain substrings are enough here.
"""
import csv, glob, io, json, os, re

LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
       "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

NEG = ["няма", "без", "geen", "kein", "keine", "ohne", "niet", "no ", "not ", "не се", "липсва",
       "не се установява",
       "\u03c7\u03c9\u03c1\u03b9\u03c2\u0020\u03b5\u03bd\u03b4\u03b1\u03c1\u03b8\u03c1\u03b9\u03ba\u03b7",
       "\u03c7\u03c9\u03c1\u03b9\u03c2\u0020\u03b5\u03bd\u03b4\u03bf\u03b1\u03c1\u03b8\u03c1\u03b9\u03ba\u03b7",
       "\u03b4\u03b5\u03bd\u0020\u03c0\u03b1\u03c1\u03b1\u03c4\u03b7\u03c1\u03b5\u03b9\u03c4\u03b1\u03b9\u0020\u03b5\u03bd\u03b4\u03b1\u03c1\u03b8\u03c1\u03b9\u03ba\u03b7",
       "\u03c7\u03c9\u03c1\u03b9\u03c2\u0020\u03b1\u03be\u03b9\u03bf\u03bb\u03bf\u03b3\u03b7\u03c3\u03b9\u03bc\u03b7",
       "\u03b4\u03b5\u03bd\u0020\u03b1\u03bd\u03b1\u03b3\u03bd\u03c9\u03c1\u03b9\u03b6\u03b5\u03c4\u03b1\u03b9",
       "\u03b1\u03be\u03b9\u03bf\u03bb\u03bf\u03b3\u03b7\u03c3\u03b9\u03bc\u03b7",
       "\u03bc\u03bf\u03bb\u03b9\u03c2\u0020\u03c5\u03c0\u03bf\u03c3\u03b7\u03bc\u03b1\u03b9\u03bd\u03b5\u03c4\u03b1\u03b9",
       "\u03c9\u03c1\u03b9\u03c2\u0020\u03b5\u03bd\u03b4\u03b1\u03c1\u03b8\u03c1\u03b9\u03ba\u03b7", "\u03c9\u03c1\u03b9\u03c2\u0020\u03b5\u03bd\u03b4\u03bf\u03b1\u03c1\u03b8\u03c1\u03b9\u03ba\u03b7", "\u03c9\u03c1\u03b9\u03c2\u0020\u03b1\u03be\u03b9\u03bf\u03bb\u03bf\u03b3\u03b7", "\u03c9\u03c1\u03b9\u03c2\u0020\u03b1\u03be\u03b9\u03bf\u03c3\u03b7\u03bc\u03b5\u03b9\u03c9\u03c4\u03b7", "\u03bc\u03bf\u03bb\u03b9\u03c2\u0020\u03b4\u03b9\u03b1\u03ba\u03c1\u03b9\u03bd\u03b5\u03c4\u03b1\u03b9",
       "trace ", "physiologic", "bez ", "nema ", "ne nalazi se", "ne vidi se", "ne prikazuje se", "minimal", "minimlan", "nesignifikantan", "tek "]

import unicodedata
def fold(t):
    t = t.replace("µ", "μ").replace("∆", "δ").replace("·", ":").replace("ı", "i")
    t = "".join(c for c in unicodedata.normalize("NFD", t.lower())
                if not unicodedata.combining(c))
    return " ".join(t.split())


def negated(before, after):
    b = before[-55:].lower()
    if any(w in b for w in NEG): return True
    a = after[:16].lower().lstrip(" :=-")
    return a.startswith(("geen", "kein", "none", "negatief", "б.о.", "not seen", "not visualized", "not torn", "absent", "negative"))

def hit(pattern, text):
    text = fold(text)
    for m in re.finditer(fold(pattern), text, re.I):
        if not negated(text[:m.start()], text[m.end():]):
            return True
    return False

RULES = [
 (0, "intacte voorste kruisband|kruisbanden intact|kruisbanden en laterale banden intact"
     "|Kreuzb\u00e4nder intakt|Intakte Kreuzb\u00e4nder|intaktes VKB|VKB und HKB intakt"
     "|\u043a\u0440\u044a\u0441\u0442\u043d\u0438\u0442\u0435 \u0432\u0440\u044a\u0437\u043a\u0438 \u0441\u0430 \u0438\u043d\u0442\u0430\u043a\u0442\u043d\u0438"
     "|\u043a\u0440\u044a\u0441\u0442\u043d\u0438\u0442\u0435 \u0432\u0440\u044a\u0437\u043a\u0438 \u0441\u0430 \u0441\u044a\u0441 \u0437\u0430\u043f\u0430\u0437\u0435\u043d\u0430",
     "ruptuur van de voorste kruisband|volledige scheur van de voorste kruisband"
     "|VKB-Ruptur|Ruptur des VKB|VKB Ruptur"
     "|\u0440\u0443\u043f\u0442\u0443\u0440\u0430 \u043d\u0430 \u043f\u0440\u0435\u0434\u043d\u0430"
     "|\u0441\u043a\u044a\u0441\u0432\u0430\u043d\u0435 \u043d\u0430 \u043f\u0440\u0435\u0434\u043d\u0430\u0442\u0430"),
 (2, "intacte mediale meniscus|Intakter Innenmeniskus",
     "Riss des Innenmeniskus|Korbhenkelriss des Innenmeniskus"
     "|scheur.{0,30}(?:hoorn|corpus).{0,25}mediale meniscus"
     "|mediale meniscus.{0,40}scheur"),
 (3, "intacte laterale meniscus|Intakter Au\u00dfenmeniskus",
     "Riss des Au\u00dfenmeniskus|scheur.{0,25}laterale meniscus"),
 (7, "Kein Gelenkerguss|geen hydrops"
     "|\u043d\u044f\u043c\u0430 \u0434\u0430\u043d\u043d\u0438 \u0437\u0430 \u0441\u0442\u0430\u0432\u0435\u043d \u0438\u0437\u043b\u0438\u0432",
     "Ausgedehnter Gelenkerguss|H\u00e4marthros|forse hydrops"
     "|\u043d\u0430\u043b\u0438\u0447\u0438\u0435 \u043d\u0430 \u0441\u0442\u0430\u0432\u0435\u043d \u0438\u0437\u043b\u0438\u0432"),
 (9, "Bakercyste: geen",
     "[Bb]akercyste|Baker-Zyste|\u0411\u0435\u043a\u0435\u0440\u043e\u0432\u0430 \u043a\u0438\u0441\u0442\u0430"),

 # Greek. Text and patterns are folded (accents stripped, micro sign mapped to mu)
 (0, "(?:\u03c6\u03c5\u03c3\u03b9\u03bf\u03bb\u03bf\u03b3\u03b9\u03ba|\u03b5\u03bd\u03c4\u03bf\u03c2\u0020\u03c4\u03bf\u03c5\u0020\u03c6\u03c5\u03c3\u03b9\u03bf\u03bb\u03bf\u03b3\u03b9\u03ba\u03bf\u03c5|\u03b4\u03b5\u03bd\u0020\u03c0\u03b1\u03c1\u03b1\u03c4\u03b7\u03c1\u03bf\u03c5\u03bd\u03c4\u03b1\u03b9|\u03b4\u03b5\u03bd\u0020\u03b1\u03bd\u03b1\u03b3\u03bd\u03c9\u03c1\u03b9\u03b6\u03bf\u03bd\u03c4\u03b1\u03b9|\u03b4\u03b5\u03bd\u0020\u03c3\u03b7\u03bc\u03b5\u03b9\u03c9\u03bd\u03bf\u03bd\u03c4\u03b1\u03b9)[^.]{0,90}(?:\u03c7\u03b9\u03b1\u03c3\u03c4\u03bf\u03b9|\u03c7\u03b9\u03b1\u03c3\u03c4\u03c9\u03bd)|(?:\u03c7\u03b9\u03b1\u03c3\u03c4\u03bf\u03b9|\u03c7\u03b9\u03b1\u03c3\u03c4\u03c9\u03bd)[^.]{0,90}(?:\u03c6\u03c5\u03c3\u03b9\u03bf\u03bb\u03bf\u03b3\u03b9\u03ba|\u03b5\u03bd\u03c4\u03bf\u03c2\u0020\u03c4\u03bf\u03c5\u0020\u03c6\u03c5\u03c3\u03b9\u03bf\u03bb\u03bf\u03b3\u03b9\u03ba\u03bf\u03c5|\u03b4\u03b5\u03bd\u0020\u03c0\u03b1\u03c1\u03b1\u03c4\u03b7\u03c1\u03bf\u03c5\u03bd\u03c4\u03b1\u03b9|\u03b4\u03b5\u03bd\u0020\u03b1\u03bd\u03b1\u03b3\u03bd\u03c9\u03c1\u03b9\u03b6\u03bf\u03bd\u03c4\u03b1\u03b9|\u03b4\u03b5\u03bd\u0020\u03c3\u03b7\u03bc\u03b5\u03b9\u03c9\u03bd\u03bf\u03bd\u03c4\u03b1\u03b9)",
     "\u03c1\u03b7\u03be[^.]{0,30}(?:\u03c0\u03c1\u03bf\u03c3\u03b8\u03b9\u03bf\u03c5\u0020\u03c7\u03b9\u03b1\u03c3\u03c4\u03bf\u03c5|\u03c0\u03c1\u03bf\u03c3\u03b8\u03b9\u03bf\u03c2\u0020\u03c7\u03b9\u03b1\u03c3\u03c4\u03bf\u03c2|\u03c0\u03c1\u03bf\u03c3\u03b8\u03b9\u03bf\u0020\u03c7\u03b9\u03b1\u03c3\u03c4\u03bf)|(?:\u03c0\u03c1\u03bf\u03c3\u03b8\u03b9\u03bf\u03c5\u0020\u03c7\u03b9\u03b1\u03c3\u03c4\u03bf\u03c5|\u03c0\u03c1\u03bf\u03c3\u03b8\u03b9\u03bf\u03c2\u0020\u03c7\u03b9\u03b1\u03c3\u03c4\u03bf\u03c2|\u03c0\u03c1\u03bf\u03c3\u03b8\u03b9\u03bf\u0020\u03c7\u03b9\u03b1\u03c3\u03c4\u03bf)[^.]{0,120}\u03c1\u03b7\u03be"),
 (7, "\u03c7\u03c9\u03c1\u03b9\u03c2\u0020\u03b5\u03bd\u03b4\u03b1\u03c1\u03b8\u03c1\u03b9\u03ba\u03b7\u0020\u03c3\u03c5\u03bb\u03bb\u03bf\u03b3\u03b7|\u03b4\u03b5\u03bd\u0020\u03c0\u03b1\u03c1\u03b1\u03c4\u03b7\u03c1\u03b5\u03b9\u03c4\u03b1\u03b9\u0020\u03b5\u03bd\u03b4\u03b1\u03c1\u03b8\u03c1\u03b9\u03ba\u03b7\u0020\u03c3\u03c5\u03bb\u03bb\u03bf\u03b3\u03b7",
     "\u03c3\u03c5\u03bb\u03bb\u03bf\u03b3\u03b7\u0020\u03c5\u03b3\u03c1\u03bf\u03c5|\u03c5\u03b4\u03c1\u03b1\u03c1\u03b8\u03c1\u03bf|\u03b1\u03b9\u03bc\u03b1\u03c1\u03b8\u03c1\u03bf"),
 (9, "zzz-no-such-phrase-zzz",
     "\u03ba\u03c5\u03c3\u03c4\u03b7\u0020\u0062\u0061\u006b\u0065\u0072"),

 # Croatian
 (0, "prednji krizni ligament urednog|prednji i straznji krizni ligament[a-z]{0,2} su primjerenih|prednjeg i straznjeg kriznog ligamenta|krizni ligamenti su primjerenih|krizni ligamenti odrzani",
     "ruptur[^.]{0,20}prednj[a-z]{0,2} krizn|prednji krizni ligament se ne prikazuje"),
 (2, "medijalni menisk bez znakova|medijalni menisk uredne",
     "ruptur[^.]{0,60}medijalnog menis"),
 (3, "lateralni menisk bez znakova|lateralni menisk uredne",
     "ruptur[^.]{0,60}lateralnog menis"),
 (7, "bez izljeva|bez signifikantnog izljeva|nema znacajnijeg izljeva|bez znacajnog izljeva|bez vidljivog izljeva",
     "izljev(?![^.]{0,40}(?:burz|ovojnic|cist|insercij|hvatist|iliotibijaln|tetiv|gastrocnem))"),
 (9, "zzz-none-zzz",
     "baker cista|bakerova cista|poplitealna cista|baker ciste"),

 # Turkish
 (0, "(?<!arka )capraz ve yan baglar normal|(?<!arka )capraz baglar normal|on capraz bag ve arka capraz bag[^.]{0,70}normal",
     "on capraz bag[^.]{0,100}(?:yirti[kg]|ruptur)|anterior capraz bag[^.]{0,100}(?:yirti[kg]|ruptur)|anterior capraz bag secilemedi|acl[^.]{0,60}yirti[kg]"),
 (2, "medial meniskus normal|medyal ve lateral meniskus normal|medial ve lateral meniskus normal",
     "med[iy]al meniskus(?:(?!lateral)[^.]){0,110}yirti[kg](?![^.]{0,25}(?:icermiyor|eslik etmiyor|yoktur|izlenmedi|saptanma))|med[iy]al meniskus[^.]{0,60}kova sapi"),
 (3, "lateral meniskus normal|medyal ve lateral meniskus normal|medial ve lateral meniskus normal",
     "lateral meniskus(?:(?!med[iy]al)[^.]){0,110}yirti[kg](?![^.]{0,25}(?:icermiyor|eslik etmiyor|yoktur|izlenmedi|saptanma))|lateral meniskus[^.]{0,60}kova sapi"),
 (7, "sivi artisi saptanmamistir|sivi miktari normal|patolojik sivi artisi yoktur",
     "efuzyon(?![^.]{0,25}(?:saptanma|yoktur|izlenmedi|izlenmemistir))"
     "|sivi miktari[^.]{0,25}artmis"
     "|eklem[^.]{0,60}sivi artisi(?![^.]{0,25}(?:saptanma|yoktur|izlenmedi|izlenmemistir))"
     "|sivi artisi[^.]{0,40}eklem(?![^.]{0,25}(?:saptanma|yoktur|izlenmedi|izlenmemistir))"
     "|suprapatellar burs[^.]{0,45}(?:sivi artisi|efuzyon)(?![^.]{0,25}(?:saptanma|yoktur|izlenmedi|izlenmemistir))"),
 (9, "zzz-none-zzz", "baker kisti"),

 # Spanish
 (0, "ligamentos cruzados y colaterales dentro de limites normales|ligamentos cruzados y colaterales sin alteraciones"
     "|ligamentos cruzados y colaterales de grosor y senal normal|ligamentos cruzados y colaterales continuos"
     "|ligamentos cruzados y colateral lateral sin alteraciones|lca continuo|ligamentos conservados y colaterales",
     "rotura completa de lca|rotura del lca|rotura de lca|desgarro parcial del lca|desgarro del lca"
     "|rotura parcial del lca|lca[^.]{0,60}(?:rotura|desgarro)"),
 (1, "ligamentos cruzados y colaterales dentro de limites normales|ligamentos cruzados y colaterales sin alteraciones"
     "|ligamentos colaterales de grosor y senal normales|ligamentos cruzados y colaterales de grosor y senal normal",
     "esguince grado [i1][i1]* *-* *[i1v]* *del componente superficial|ligamento colateral medial[^.]{0,120}perdida parcial de la continuidad"
     "|rotura[^.]{0,40}ligamento colateral medial"),
 (2, "menisco medial de morfologia y senal conservada|meniscos de morfologia y senal conservada"
     "|meniscos de morfologia y senal normal|menisco medial de morfologia normal"
     "|meniscos impresionan de morfologia y senal conservada|meniscos de morfologia y senal conservada"
     "|menisco interno sin desgarro|meniscos impresionan de morfologia y senal conservada",
     "rotura de menisco interno|rotura de menisco medial|rotura[^.]{0,90}del menisco medial"
     "|rotura[^.]{0,90}menisco medial(?![^.]{0,20}sin)"),
 (3, "menisco lateral de morfologia y senal conservada|el menisco lateral es de morfologia y senal conservada"
     "|menisco lateral de morfologia y senal normal|meniscos de morfologia y senal conservada"
     "|meniscos de morfologia y senal normal|menisco lateral impresiona de morfologia y senal conservada",
     "rotura de menisco externo|rotura menisco externo|rotura[^.]{0,90}del menisco lateral"
     "|rotura[^.]{0,90}menisco lateral(?![^.]{0,20}sin)"),
 (7, "no hay derrame articular|no hay evidente derrame articular|no hay significativo derrame articular"
     "|no hay signos de derrame articular|no hay signos que sugieran derrame|sin derrame articular"
     "|no hay signos que sugieran significativo derrame",
     "derrame articular|derrame con sinovitis|derrame leve|derrame[.,]"),
 (9, "no hay quistes popliteos|no hay quiste popliteo|no se observa quiste popliteo|sin quistes popliteos",
     "(?<!ni )(?<!o )(?:quiste de baker|quiste popliteo|quistes popliteos)"),

 # English
 (0, "anterior cruciate ligament [(]acl[)]: normal|cruciate ligaments are intact|intact anterior cruciate ligament"
     "|(?<!ab)normal acl|acl: intact|acl is intact|acl intact|acl normal|acl: unremarkable"
     "|contour of (?:the )?(?:anterior cruciate ligament|acl)[^.]{0,40}within normal"
     "|cruciate ligaments[)]: normal|cruciate and collateral ligaments",
     "tear of the anterior cruciate ligament|acl tear|tear of acl|acl is ruptured|acl rupture"
     "|anterior cruciate ligament tear|acl(?:(?!intact|normal|pcl)[^.]){0,30}(?:complete|partial|high.?grade) tear"
     "|(?:complete|partial|high.?grade) tear of (?:the )?(?:anterior cruciate ligament|acl)"),
 (2, "medial meniscus: normal in morphology|medial meniscus is not torn|normal medial and lateral menisci"
     "|intact menisci|medial meniscus:  *intact|the medial meniscus is intact|medial meniscus:  *preserved"
     "|medial meniscus; preserved|menisci are intact|medial compartment [(]meniscus, collateral ligament, cartilage[)]: normal"
     "|medial meniscus is not torn|medial compartment,? (?:the )?meniscus is not torn|medial meniscus of normal|medial meniscus normal|no medial meniscal tear",
     "medial meniscus tear|tear(?:(?!without|not torn|not tear|no tear|no meniscal|no definite|no obvious|no evidence|intact)[^.]){0,40}medial meniscus(?!:? *(?:intact|normal|preserved|unremarkable|no |not |without|0))|medial meniscus(?:(?!without|not torn|not tear|no tear|no meniscal|no definite|no obvious|no evidence|intact)[^.]){0,60}tear"),
 (3, "lateral meniscus: normal in morphology|lateral meniscus is not torn|normal medial and lateral menisci"
     "|intact menisci|lateral meniscus:  *intact|the lateral meniscus is intact|lateral meniscus:  *preserved"
     "|menisci are intact|lateral compartment [(]meniscus, collateral ligament complex, cartilage[)]: normal"
     "|lateral meniscus normal|no lateral meniscal tear",
     "lateral meniscus tear|tear(?:(?!without|not torn|not tear|no tear|no meniscal|no definite|no obvious|no evidence|intact)[^.]){0,40}lateral meniscus(?!:? *(?:intact|normal|preserved|unremarkable|no |not |without|0))|lateral meniscus(?:(?!without|not torn|not tear|no tear|no meniscal|no definite|no obvious|no evidence|intact)[^.]){0,60}tear"),
 (7, "no joint effusion|no knee effusion|no evidence of knee effusion|no significant knee joint effusion"
     "|no effusion|no evidence of effusion|no significant effusion|no significant knee effusion",
     "(?<!normal )(?:joint effusion|knee effusion)(?! *[:;])(?![^.]{0,12}none)|lipohemarthrosis|effusion is present"),
 (9, "no baker'?s cyst|no baker cyst|no popliteal cyst|baker cyst: not seen|no significant knee joint effusion or popliteal cyst",
     "(?<!without )(?:baker'?s cyst|baker cyst|popliteal cyst)(?! *[:;])"),
]

KNEE = re.compile("knie|knee|genou|ménisq|croisé|meniscus|meniskus|menisk|menisc|kruisband|kreuzb|tibia|femur"
                  "|femoro|quadriz|gonarthr|patel|rodilla|cruzado|rótula|fémur"
                  "|diz |dizi|çapraz|koljen|križn|мениск"
                  "|колянна|кръстна|пател"
                  "|\u03bc\u03b7\u03bd\u03b9\u03c3\u03ba|\u03b3\u03bf\u03bd\u03b1\u03c4|\u03c7\u03b9\u03b1\u03c3\u03c4|\u03b5\u03c0\u03b9\u03b3\u03bf\u03bd|\u03ba\u03bd\u03b7\u03bc", re.I)

# The referral header states what the clinician SUSPECTS ("echografisch vermoedelijk mediale
# meniscus scheur"), which is routinely the opposite of what the scan shows. Audit the report's own
# findings only - everything from the first findings heading onward.
HEAD = re.compile("FINDINGS:|IMAGING FINDINGS:|Findings:|Hallazgos:|Resultados:|Bevindingen:|Befund:|Beschreibung:|Mediales Kompartiment:|\u041c\u0420 \u043d\u0430\u0445\u043e\u0434\u043a\u0430:", re.I)

def findings(t):
    m = HEAD.search(t)
    return t[m.end():] if m else t


rep = {}
for lang, rows in json.load(io.open("_by_language.json", encoding="utf-8")).items():
    for r in rows: rep[r["id"]] = r["report"]

flags, nonknee, nofind, n = [], [], [], 0
for p in sorted(glob.glob("opuslabels/*.csv")):
    for row in csv.DictReader(io.open(p, encoding="utf-8")):
        n += 1
        t = rep.get(row["id"])
        if t is None: continue
        if not KNEE.search(t):
            nonknee.append((os.path.basename(p), row["id"][-12:], t[:60])); continue
        t = findings(t)
        # A report that says 'sin anomalias' HAS findings text - it reports normal.
        if len(t.strip()) < 40 and not re.search('sin anomal|normal', t, re.I):
            # report carries no findings at all; the labels can only be priors, not readings
            nofind.append((os.path.basename(p), row["id"][-12:])); continue
        for idx, intact_re, torn_re in RULES:
            v = float(row[LAB[idx]])
            t_hit = hit(torn_re, t)
            if hit(intact_re, t) and not t_hit and v >= 0.60:
                flags.append((os.path.basename(p), row["id"][-10:], LAB[idx], v, "report says INTACT"))
            elif t_hit and v <= 0.25:
                flags.append((os.path.basename(p), row["id"][-10:], LAB[idx], v, "report says PRESENT"))

print("audited %d labelled rows" % n)
print("NON-KNEE reports: %d" % len(nonknee))
for f in nonknee: print("   %s %s | %s" % f)
print("NO FINDINGS TEXT (labels are priors, not readings): %d" % len(nofind))
for f in nofind: print("   %s %s" % f)
print("CONTRADICTIONS: %d  (%.1f%% of %d checked cells)" % (len(flags), 100.0*len(flags)/max(1,n*len(RULES)), n*len(RULES)))
for f in flags[:25]:
    print("   %-13s %s %-17s %.2f  <- %s" % f)
