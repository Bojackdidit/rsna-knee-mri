#!/usr/bin/env python3
                      
"""Where a score stands on the board: teams strictly above, and our own published rank. Usage: board_position.py [--score S] [--selftest]"""
import argparse
import contextlib
import io
import re
import sys

COMP = "rsna-knee-abnormality-detection"
DP = 3                            

def walk(api, floor=0.940, max_pages=60):
    """Every score down to `floor`, paging correctly. Returns a descending list."""
    scores, tok, pages = [], None, 0
    while pages < max_pages:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            page = api.competition_leaderboard_view(COMP, page_size=100, page_token=tok)
        if not page:
            break
        scores += [float(getattr(r, "score", 0) or 0) for r in page]
        pages += 1
        m = re.search(r"Next Page Token = (\S+)", buf.getvalue())
        tok = m.group(1) if m else None
        if tok is None or scores[-1] < floor:
            break
    return scores

def download_board():
    """The WHOLE board in one call: CSV with a published Rank column and TeamMemberUserNames.

    This is the preferred path. Returns a list of dicts, or None if the download fails.
    """
    import csv, glob, os, subprocess, tempfile, zipfile
    d = tempfile.mkdtemp()
    r = subprocess.run(["kaggle", "competitions", "leaderboard", COMP, "--download", "-p", d],
                       capture_output=True, text=True)
    zs = glob.glob(os.path.join(d, "*.zip"))
    for z in zs:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
    cs = glob.glob(os.path.join(d, "*publicleaderboard*.csv"))
    if not cs:
        return None
    with io.open(cs[0], encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))

def position(scores, score):
    """Teams strictly above, teams tied on the DISPLAYED score, and the rank range."""
    eps = 0.5 * 10 ** -DP
    above = sum(1 for s in scores if s > score + 1e-9)
    tied = sum(1 for s in scores if abs(s - score) < eps)
    return {"above": above, "tied": tied,
            "rank_best": above + 1, "rank_worst": above + tied,
            "walked": len(scores)}

def selftest():
    """The helper must report a RANGE when the score is shared, and never the best case alone."""
    board = [0.965, 0.960, 0.955, 0.952, 0.951, 0.951, 0.951, 0.950, 0.949]
    p = position(board, 0.951)
    ok = True
    checks = [
        ("teams strictly above 0.951 is 4", p["above"] == 4),
        ("teams tied at 0.951 is 3", p["tied"] == 3),
        ("rank range is 5..7, not just 5", (p["rank_best"], p["rank_worst"]) == (5, 7)),
        ("range is WIDER than one when tied", p["rank_worst"] > p["rank_best"]),
    ]
                                                                          
    q = position(board, 0.952)
    checks.append(("a unique score gives a single rank",
                   q["rank_best"] == q["rank_worst"] == 4))
                                                                                                
    checks.append(("best-case rank != worst-case when tied, so quoting one is an error",
                   p["rank_best"] != p["rank_worst"]))
    for name, got in checks:
        print("  %-62s %s" % (name, "OK" if got else "FAIL"))
        ok = ok and got
    print("\nSELFTEST: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=float, default=0.951)
    ap.add_argument("--floor", type=float, default=0.940)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    rows = download_board()
    if rows is None:
        print("  --download failed; falling back to the paginated walk (rank will be a RANGE only)")
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        p = position(walk(api, floor=a.floor), a.score)
        print("  TEAMS STRICTLY ABOVE US : %d" % p["above"])
        print("  our rank is a RANGE     : %d - %d  (fallback path)" % (p["rank_best"], p["rank_worst"]))
        return 0

    rk = list(rows[0].keys())[0]                                                 
    us = next((r for r in rows
               if "dreaddevelopment" in str(r.get("TeamMemberUserNames", "")).lower()), None)
    sc = float(us["Score"]) if us else a.score
    above = sum(1 for r in rows if float(r["Score"]) > sc + 1e-9)
    tied = sum(1 for r in rows if abs(float(r["Score"]) - sc) < 0.5 * 10 ** -DP)

    print("  full board downloaded in ONE call: %d teams" % len(rows))
    print()
    if us:
        print("  OUR PUBLISHED RANK      : %s of %d      <- an exact number, not a range" %
              (us[rk], len(rows)))
        print("  our score / submissions : %s / %s" % (us["Score"], us["SubmissionCount"]))
    else:
        print("  OUR ROW NOT FOUND by TeamMemberUserNames - do not guess; investigate before quoting")
    print("  TEAMS STRICTLY ABOVE US : %d        <- best metric for tracking EROSION" % above)
    print("  teams tied on %.3f      : %d" % (sc, tied))
    print()
    print("  top 3:")
    for r in rows[:3]:
        print("    %-4s %-28s %s" % (r[rk], r["TeamName"][:28], r["Score"]))
    print()
    print("  Do NOT extrapolate a trajectory from this - wrong twice on this competition.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
