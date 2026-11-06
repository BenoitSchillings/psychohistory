"""One source of truth: every headline number quoted in the prose (outputs/narrative.md, inlined in the report; the
methodology's headline) must equal the current results (outputs/forecast.json, outputs/country_forecasts.csv). The
report quoted 49% great-power war and $36k from a narrative written on 1 November while the headline said 45% and
$34k (debate, 6 November 2026).

Each check is a pattern with numbered groups and the values the current results give; --fix writes those values
into the text. A pattern that no longer matches is reported (the prose changed: update the check), never skipped.
python -P scripts/number_parity.py [--fix]      (exit 1 on any mismatch or missing pattern; run before a release)"""
import csv
import json
import os
import re
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)


def values():
    fc = json.load(open(os.path.join(ROOT, "outputs", "forecast.json")))
    A, H = fc["ai"], fc["history"]
    u = A["uncertain"]
    p = lambda x: f"{100 * x:.0f}"
    k = lambda x: f"{x / 1000:.0f}"
    rows = list(csv.DictReader(open(os.path.join(ROOT, "outputs", "country_forecasts.csv"))))
    num = lambda r, c: float(r[c]) if r.get(c) not in (None, "") else -1.0
    names = {r["code"]: r["name"] for r in rows}

    def top(col, cond=lambda r: True, n=3):
        rs = sorted((r for r in rows if cond(r)), key=lambda r: -num(r, col))[:n]
        return [(r["name"], p(num(r, col))) for r in rs]
    peace = lambda r: r["in_conflict"] in ("0", "0.0")
    us = next(r for r in rows if r["code"] == "USA")
    return {
        "gpw": p(u["great_power_war_by"][-1]), "gpw_noai": p(A["none"]["great_power_war_by"][-1]),
        "gpw_1914": p(A["alliances_1914"]["great_power_war_by"][-1]),
        "gpw_cow": p(u["great_power_war_cow_by"][-1]), "gpw_cow_noai": p(A["none"]["great_power_war_cow_by"][-1]),
        "gpw_cow_1914": p(A["alliances_1914"]["great_power_war_cow_by"][-1]),
        "gdp_med": k(u["world_gdp_q"][2][-1]), "gdp_noai": k(A["none"]["world_gdp_q"][2][-1]),
        "gdp_lo": k(u["world_gdp_q"][0][-1]), "gdp_hi": k(u["world_gdp_q"][-1][-1]),
        "gpw_2030": p(A["2030"]["great_power_war_by"][-1]), "gpw_cow_2030": p(A["2030"]["great_power_war_cow_by"][-1]),
        "conf_2065": p(u["world_conflict_q"][2][-1]), "conf_lo": p(u["world_conflict_q"][0][-1]),
        "conf_hi": p(u["world_conflict_q"][-1][-1]), "conf_now": p(H["world_conflict"][-1]),
        "p_record": p(fc["world"]["p_exceed_hist_max"]),
        "nuke": p(u["risks"]["nuclear_use_p"]), "nuke_large": f"{100 * u['risks']['nuclear_large_p']:.1f}",
        "dem_2065": f"{u['world_dem_q'][2][-1]:.2f}", "dem_noai": f"{A['none']['world_dem_q'][2][-1]:.2f}",
        "arr": [p(u["arrival_share_by"][y]) for y in ("2027", "2030", "2040", "2065")],
        "war_top": top("civil_conflict_5y", peace), "coup_top": top("coup_5y"),
        "autoc_top": top("autocratization_5y"), "us_aut": p(num(us, "autocratization_5y")),
    }


def checks(V):
    """(file, label, pattern, expected values for its groups in order)"""
    N = "outputs/narrative.md"
    pair = lambda lst: [x for nm, v in lst for x in (re.escape(nm), v)]
    return [
        (N, "great-power war (headline, no AI, 1914 alliances)",
         r"About (\d+)% of worlds see a war of at least 1,000 battle deaths a year between great powers by 2065 \((\d+)% with no AI shock, (\d+)% with 1914-style alliances\)",
         [V["gpw"], V["gpw_noai"], V["gpw_1914"]]),
        (N, "great-power war, traditional majors",
         r"Among the traditional majors alone the figure is (\d+)% \((\d+)% without AI, (\d+)% with 1914-style alliances\)",
         [V["gpw_cow"], V["gpw_cow_noai"], V["gpw_cow_1914"]]),
        (N, "median world GDP per person 2065 (headline, no AI, band)",
         r"Median world GDP per person in 2065 is \$(\d+)k, against \$(\d+)k if nothing changes, with a 90% band from \$(\d+)k to \$(\d+)k",
         [V["gdp_med"], V["gdp_noai"], V["gdp_lo"], V["gdp_hi"]]),
        (N, "great-power war (wars between states section)",
         r"happens by 2065 in (\d+)% of headline worlds counting India \((\d+)% among the traditional majors\); with 1914-style alliances, in (\d+)% \((\d+)%\)",
         [V["gpw"], V["gpw_cow"], V["gpw_1914"], V["gpw_cow_1914"]]),
        (N, "great-power war with AI in 2030",
         r"rises from (\d+)% without AI to (\d+)% with arrival in 2030 \((\d+)% to (\d+)% among the traditional majors\)",
         [V["gpw_noai"], V["gpw_2030"], V["gpw_cow_noai"], V["gpw_cow_2030"]]),
        (N, "large nuclear exchange (short version)", r"large exchange between great powers in (\d\.\d)%", [V["nuke_large"]]),
        (N, "large nuclear exchange (section)", r"large exchange between great powers happens in about (\d\.\d)%", [V["nuke_large"]]),
        (N, "civil conflict 2065 and band",
         r"is about (\d+)% in 2065, with a 90% band from (\d+)% to (\d+)%; in about three worlds in four \((\d+)%\)",
         [V["conf_2065"], V["conf_lo"], V["conf_hi"], V["p_record"]]),
        (N, "AI arrival shares", r"by 2027 in (\d+)% of worlds, by 2030 in (\d+)%, by 2040 in (\d+)% and by 2065 in (\d+)%", V["arr"]),
        (N, "nuclear use (short version)", r"nuclear weapon is used in a war in about (\d+)% of worlds", [V["nuke"]]),
        (N, "nuclear use (section)", r"nuclear weapon is used in a war by 2065 in about (\d+)% of worlds", [V["nuke"]]),
        (N, "democracy 2065", r"world democracy index ends near (0\.\d+) in 2065 \((0\.\d+) with no AI shock", [V["dem_2065"], V["dem_noai"]]),
        (N, "US autocratization by 2030", r"the five-year model gives a (\d+)% chance of another fall that large by 2030", [V["us_aut"]]),
        (N, "new civil war, top three", r"New civil war: ([^(]+) \((\d+)%\), ([^(]+) \((\d+)%\), ([^(]+) \((\d+)%\)\.", pair(V["war_top"])),
        (N, "coups, top three", r"Coups: ([^(]+) \((\d+)%\), ([^(]+) \((\d+)%\), ([^(]+) \((\d+)%\)\.", pair(V["coup_top"])),
        (N, "autocratization, top three", r"Autocratization: ([^(]+) \((\d+)%\), ([^(]+) \((\d+)%\), ([^(]+) \((\d+)%\)\.", pair(V["autoc_top"])),
    ]


def main():
    fix = "--fix" in sys.argv
    V = values()
    bad = 0
    texts = {}
    for f, label, pat, exp in checks(V):
        path = os.path.join(ROOT, f)
        t = texts.setdefault(f, open(path).read())
        m = re.search(pat, t)
        if not m:
            print(f"MISSING  {f}: {label} (pattern not found: the prose changed, update scripts/number_parity.py)")
            bad += 1
            continue
        got = list(m.groups())
        want = [re.sub(r"\\(.)", r"\1", e) for e in exp]
        if got == want:
            print(f"ok       {f}: {label} = {got}")
            continue
        print(f"MISMATCH {f}: {label}: text {got}, results {want}")
        bad += 1
        if fix:
            s, e0 = m.span()
            seg = t[s:e0]
            for i in range(len(got), 0, -1):          # replace groups right to left to keep offsets
                a, b = m.span(i)
                seg = seg[:a - s] + want[i - 1] + seg[b - s:]
            texts[f] = t = t[:s] + seg + t[e0:]
    if fix:
        for f, t in texts.items():
            open(os.path.join(ROOT, f), "w").write(t)
    r = subprocess.run([sys.executable, "-P", os.path.join(ROOT, "scripts", "headline_sync.py"), "--check"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
    ok = r.returncode == 0
    print(("ok       " if ok else "MISMATCH ") + "docs/methodology.md headline (scripts/headline_sync.py --check)")
    bad += not ok
    print(f"{bad} problem(s)" + (" (fixed in the text; re-run to confirm)" if fix and bad else ""))
    sys.exit(1 if bad and not fix else 0)


if __name__ == "__main__":
    main()
