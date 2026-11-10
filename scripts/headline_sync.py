"""The headline paragraph of docs/methodology.md, written from the current forecast (outputs/forecast.json) with each
number's known hindcast bias next to it, so the methodology cannot quote a stale forecast (it still gave the
29 October numbers on 5 November). The paragraph sits between <!-- headline --> and <!-- /headline -->.

python -P scripts/headline_sync.py          # rewrite the paragraph
python -P scripts/headline_sync.py --check  # exit 1 if the paragraph is out of date (run before a release)"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DOC = os.path.join(os.path.dirname(__file__), "..", "docs", "methodology.md")
A, B = "<!-- headline -->", "<!-- /headline -->"


def paragraph():
    import json
    from psychohistory import report as R
    v = R._summary_vars()
    fc = json.loads((R.OUT / "forecast.json").read_text())
    u = fc["ai"]["uncertain"]
    c_lo, c_hi = (f"{100 * u['world_conflict_q'][i][-1]:.0f}%" for i in (0, -1))
    ver, date = R.VERSION
    nf = "the nuclear-pair war factor, whose range moves it a lot"
    fr = R.OUT / "uncertainty_ranges.json"
    if fr.exists():
        g = json.loads(fr.read_text()).get("great_power_war", {}).get("drivers", {}).get("nuclear_war_factor")
        if g:
            nf = (f"the nuclear-pair war factor, whose range alone moves great-power war from "
                  f"{100 * g['low_fifth']:.0f}% to {100 * g['high_fifth']:.0f}%")
    gpw_any = fc["ai"].get("alliances_any_scope", {}).get("great_power_war_by", [None])[-1]
    any_s = f" ({100 * gpw_any:.0f}% if allies joined any war wherever it was fought, the rule before November 2026)" if gpw_any else ""
    return (f"The current headline (version {ver}, {date}): median world GDP per person in 2065 is {v['gdp_med']} "
            f"(90% band {v['gdp_lo']}–{v['gdp_hi']}; {v['gdp_noai']} with no AI shock). In hindcasts its 90% bands held "
            f"about 92% of the time at ten years, but countries' growth drifts apart 1.3–1.9 times too much since the "
            f"1990s, so single-country bands are probably too wide. About {v['conf_2065']} of countries are in civil "
            f"conflict in 2065 (band {c_lo}–{c_hi}), against {v['conf_now']} in 2025, and the 1990 record is broken in "
            f"{v['p_record']} of worlds; simulated from 2000–2012 the share at war came out about 6% low, while the safest "
            f"countries got too many new wars (two or three expected in 15 years from 1995–2009 origins, none happened). "
            f"A war between the traditional great powers of at least 1,000 battle deaths a year happens by 2065 in "
            f"{v['gpw_cow']} of worlds; counting India as a great power raises it to {v['gpw']}, mostly through a "
            f"China–India war{any_s}, and nuclear weapons are used in {v['nuke']}. These depend "
            f"heavily on assumptions, above all {nf}, and on a war model fitted to 1946–2024; the simulator missed the 2010s rise in wars between states two- to "
            f"threefold. The worst 1% of worlds on each measure are traced to their causes (see Disaster worlds).")


def main():
    doc = open(DOC).read()
    new = f"{A}\n{paragraph()}\n{B}"
    if A in doc:
        cur = doc[doc.index(A):doc.index(B) + len(B)]
    else:
        # first run: replace the hand-written headline paragraph
        m = re.search(r"^The current headline .*$", doc, flags=re.M)
        if not m:
            sys.exit("no headline paragraph found")
        cur = m.group(0)
    if "--check" in sys.argv:
        ok = cur == new
        print("headline up to date" if ok else "headline out of date: run scripts/headline_sync.py")
        sys.exit(0 if ok else 1)
    open(DOC, "w").write(doc.replace(cur, new))
    print(paragraph())


if __name__ == "__main__":
    main()
