"""Builds outputs/seldon_report.html: a self-contained interactive report from outputs/forecast.json.

The narrative (outputs/narrative.md, optional) is written separately and inlined.
"""
from __future__ import annotations

import html
import json
import re

import pandas as pd

from .data import RAW, ROOT

OUT = ROOT / "outputs"
TEMPLATE = ROOT / "psychohistory" / "report_template.html"
# the report's release number: bump on every published change (the private copy's own revision counter can run ahead)
VERSION = ("54.1", "6 November 2026")


def _world_topology():
    topo = json.loads((RAW / "countries-110m.json").read_text())
    iso = pd.read_csv(RAW / "iso3166.csv", dtype=str)
    num_to_a3 = dict(zip(iso["country-code"], iso["alpha-3"]))
    for g in topo["objects"]["countries"]["geometries"]:
        g["id"] = num_to_a3.get(str(g.get("id")), g.get("properties", {}).get("name"))
    return topo


def _md_to_html(md: str) -> str:
    """minimal markdown: ## headings, paragraphs, - bullets, **bold**, *italic*."""
    out, items = [], []

    def inline(s):
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
        # links to sections of the page only: [text](#id)
        return re.sub(r"\[([^\]]+)\]\((#[A-Za-z0-9_-]+)\)", r'<a href="\2">\1</a>', s)

    def flush():
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for block in md.strip().split("\n\n"):
        lines = block.strip().splitlines()
        if not lines:
            continue
        if lines[0].startswith("### "):
            flush()
            out.append(f"<h4>{inline(lines[0][4:])}</h4>")
            lines = lines[1:]
        elif lines[0].startswith("## "):
            flush()
            out.append(f"<h3>{inline(lines[0][3:])}</h3>")
            lines = lines[1:]
        if lines and all(l.lstrip().startswith("- ") for l in lines):
            items.extend(l.lstrip()[2:] for l in lines)
            flush()
        elif lines:
            flush()
            out.append(f"<p>{inline(' '.join(lines))}</p>")
    flush()
    return "\n".join(out)


def _anon_panel(text: str) -> str:
    """Show the analyst panel without naming the models behind it."""
    for name, label in (("Claude", "Analyst A"), ("Gemini", "Analyst B")):
        text = text.replace(f'"{name}"', f'"{label}"').replace(name, "the analyst")
    return text


def _summary_vars():
    """numbers for outputs/summary_template.md ("The future in one page"), read from the current results so the
    summary cannot go stale when the forecast is rerun"""
    import csv
    fc = json.loads((OUT / "forecast.json").read_text())
    u, n, H, W = fc["ai"]["uncertain"], fc["ai"]["none"], fc["history"], fc["world"]
    k = lambda v: f"${v / 1000:.0f}k"
    p = lambda v: f"{100 * v:.0f}%"
    v = {"gdp_now": k(H["world_gdp"][-1]), "gdp_med": k(u["world_gdp_q"][2][-1]), "gdp_lo": k(u["world_gdp_q"][0][-1]),
         "gdp_hi": k(u["world_gdp_q"][-1][-1]), "gdp_noai": k(n["world_gdp_q"][2][-1]),
         "conf_now": p(H["world_conflict"][-1]), "conf_2065": p(u["world_conflict_q"][2][-1]), "p_record": p(W["p_exceed_hist_max"]),
         "gpw": p(u["great_power_war_by"][-1]), "gpw_cow": p(u["great_power_war_cow_by"][-1]),
         "nuke": p(u["risks"]["nuclear_use_p"]), "nuke_large": f"{100 * u['risks']['nuclear_large_p']:.1f}%",
         "ai_cat": f"{100 * (u['risks'].get('ai_catastrophe_p') or 0):.1f}%",
         "dem_now": f"{H['world_dem'][-1]:.2f}", "dem_2065": f"{u['world_dem_q'][2][-1]:.2f}",
         "dem_lo": f"{u['world_dem_q'][0][-1]:.2f}", "dem_hi": f"{u['world_dem_q'][-1][-1]:.2f}"}
    att = OUT / "uncertainty_attribution.json"
    if att.exists():
        v["ai_share"] = p(json.loads(att.read_text())["log_world_gdp_2065"]["groups"]["AI and robot timing"])
    rows = list(csv.DictReader(open(OUT / "country_forecasts.csv")))
    num = lambda r, c: float(r[c]) if r.get(c) not in (None, "") else -1.0
    top = lambda c, cond=lambda r: True: sorted((r for r in rows if cond(r)), key=lambda r: -num(r, c))[:3]
    and_ = lambda xs: ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else "".join(xs)
    v["war_top"] = and_([r["name"] for r in top("civil_conflict_5y", lambda r: r["in_conflict"] in ("0", "0.0"))])
    v["coup_top"] = and_([r["name"] for r in top("coup_5y")])
    us = next(r for r in rows if r["code"] == "USA")
    v["us_aut"] = p(num(us, "autocratization_5y"))
    # the next twelve months (monthly ledger, direct 12-month model) and what moves the 2065 war numbers
    lf = OUT / "ledger_summary.json"
    if lf.exists():
        led = json.loads(lf.read_text())
        names = {r["code"]: r["name"] for r in rows}
        top3 = list(led.get("annual_new", {}).items())[:3]
        v["ledger_top"] = and_([f"{names.get(c, c)} ({p(x)})" for c, x in top3])
    ur = OUT / "uncertainty_ranges.json"
    if ur.exists():
        g = json.loads(ur.read_text()).get("great_power_war", {}).get("drivers", {}).get("nuclear_war_factor")
        if g:
            v["gpw_nuc_lo"], v["gpw_nuc_hi"] = p(g["low_fifth"]), p(g["high_fifth"])
    sk = OUT / "skill_horizon.json"
    v["skill_note"] = ""
    if sk.exists():
        S = json.loads(sk.read_text())["by_horizon"]
        pos = [int(h) for h, r in S.items() if r.get("conflict", {}).get("ci90", [0])[0] > 0]
        v["skill_note"] = (f" Against a fair naive forecast, its country-level war forecasts keep measurable skill up to "
                           f"{max(pos)} years ahead in hindcasts." if pos else "")
    c21 = OUT / "climate_2100.json"
    if c21.exists():
        sc = json.loads(c21.read_text())["scenarios"]
        L = lambda s_, y, q="median": f"{-sc[s_]['loss_pct'][str(y)][q]:.0f}%"
        v.update({"clim_245_2065": L("ssp245", 2065), "clim_245_2100": L("ssp245", 2100), "clim_245_2100_hi": L("ssp245", 2100, "p5"),
                  "clim_585_2100": L("ssp585", 2100), "clim_tech_2100": L("ssp245_tech", 2100)})
        hd = lambda s_, y: f"{sc[s_]['heat_deaths']['cumulative_m_' + str(y)]:.0f} million"
        v.update({"heat_245_2065": hd("ssp245", 2065), "heat_245_2100": hd("ssp245", 2100), "heat_585_2100": hd("ssp585", 2100)})
    ts = OUT / "tech_scenarios.json"
    if ts.exists():
        t = json.loads(ts.read_text())
        a, b = t["A_uncertain_none"], t["A_uncertain_tech"]
        r = lambda key: f"{abs(100 * (b[key] / a[key] - 1)):.0f}%"
        v.update({"tech_gdp": f"{100 * (b['world_gdp_median'] / a['world_gdp_median'] - 1):.1f}%", "tech_oil": r("oil_median"),
                  "tech_grain": r("food_price_median"), "tech_sahel": r("gdp_sahel_median")})
    return v


def _worldbuilder():
    """"Build your own world": for every combination of five reader choices, the 2065 outcomes of the simulated worlds
    that match it (outputs/uncertainty_sims.npz, 48,000 worlds); sparse combinations widen by dropping the least
    important choices first. No fitted model: each answer is a set of actual simulated worlds."""
    import itertools
    import numpy as np
    f = OUT / "uncertainty_sims.npz"
    if not f.exists():
        return None
    z = np.load(f)
    need = ["p_ai_arrival", "p_dmg_family", "p_nuclear_war_factor", "p_firm_capture", "y_log_world_gdp_2065"]
    if any(k not in z.files for k in need):
        return None
    ai = z["p_ai_arrival"]
    ai_bins = [(2025, 2030), (2030, 2035), (2035, 2045), (2045, 2065.5), (2065.5, 1e9)]
    c_ai = np.full(len(ai), -1)
    for i, (a, b) in enumerate(ai_bins):
        c_ai[(ai >= a) & (ai < b)] = i
    c_cl = np.clip(np.round(z["p_dmg_family"]).astype(int), 0, 2)
    ter = lambda v: np.digitize(v, np.quantile(v, [1 / 3, 2 / 3]))
    c_nu, c_fc = ter(z["p_nuclear_war_factor"]), ter(z["p_firm_capture"])
    tk = [k for k in z.files if k.startswith("p_tech_")]
    n_t = np.sum([z[k] <= 2045 for k in tk], 0) if tk else np.zeros(len(ai))
    c_te = ter(n_t + 1e-6 * np.arange(len(n_t)) / len(n_t))
    C = np.stack([c_ai, c_cl, c_nu, c_fc, c_te], 1)
    Y = {"gdp": np.exp(z["y_log_world_gdp_2065"]), "conflict": z["y_world_conflict_2065"], "dem": z["y_world_democracy_2065"],
         "gpw": z["y_great_power_war"], "nuke": z["y_nuclear_use"] if "y_nuclear_use" in z.files else None}
    drop_order = [3, 2, 4, 1]                     # widen by dropping: rent capture, nuclear, breakthroughs, climate
    cells = {}
    for combo in itertools.product(range(5), range(3), range(3), range(3), range(3)):
        keep, used = [0, 1, 2, 3, 4], []
        while True:
            m = np.all(C[:, keep] == np.array(combo)[keep], 1)
            if m.sum() >= 60 or len(keep) == 1:
                break
            keep = [k_ for k_ in keep if k_ != drop_order[len(used)]]
            used.append(drop_order[len(used)])
        q = lambda v: [round(float(x), 4) for x in np.quantile(v[m], [0.05, 0.5, 0.95])]
        cells["".join(map(str, combo))] = {"n": int(m.sum()), "kept": len(keep), "gdp": [round(float(x)) for x in np.quantile(Y["gdp"][m], [0.05, 0.5, 0.95])],
                                           "conflict": q(Y["conflict"]), "dem": q(Y["dem"]), "gpw": round(float(Y["gpw"][m].mean()), 4),
                                           "nuke": round(float(Y["nuke"][m].mean()), 4) if Y["nuke"] is not None else None}
    return {"n": int(len(ai)), "cells": cells, "tech_cuts": [float(x) for x in np.quantile(n_t, [1 / 3, 2 / 3])]}


def _release_changes():
    """headline numbers of this release next to the previous one (outputs/release_history.json, one entry per version)"""
    hist_f = OUT / "release_history.json"
    hist = json.loads(hist_f.read_text()) if hist_f.exists() else {}
    v = _summary_vars()
    keys = ["gdp_med", "gdp_noai", "conf_2065", "dem_2065", "gpw", "nuke", "clim_245_2100", "heat_245_2100", "tech_gdp"]
    cur = {k: v.get(k) for k in keys if v.get(k) is not None}
    fc = json.loads((OUT / "forecast.json").read_text())
    ed = fc["ai"]["uncertain"]["risks"].get("excess_deaths_q")
    if ed:
        cur["excess_deaths"] = f"{ed[2] / 1e6:.0f} million"
    hist[VERSION[0]] = {"date": VERSION[1], "headline": cur}
    hist_f.write_text(json.dumps(hist, indent=1))
    vk = lambda k: tuple(int(x) for x in str(k).split("."))          # versions and patch versions (54, 54.1)
    prev = [k for k in hist if vk(k) < vk(VERSION[0])]
    if not prev:
        return None
    pk = max(prev, key=vk)
    return {"version": VERSION[0], "prev": pk, "prev_date": hist[pk]["date"], "now": cur, "before": hist[pk]["headline"]}


def build(path=None) -> str:
    fc = json.loads((OUT / "forecast.json").read_text())
    narrative = OUT / "narrative.md"
    review = OUT / "review_response.md"
    page = TEMPLATE.read_text().replace("<!--__VERSION__-->", "Version %s · %s" % VERSION)
    page = page.replace("/*__DATA__*/null", json.dumps(fc, separators=(",", ":")))
    page = page.replace("/*__TOPO__*/null", json.dumps(_world_topology(), separators=(",", ":")))
    panel = OUT / "llm_panel.json"
    page = page.replace("/*__PANEL__*/null", _anon_panel(panel.read_text()) if panel.exists() else "null")
    extra = {}
    # (the narrated single worlds, outputs/chronicles.json, are no longer shown (6 November 2026): one draw read as a
    # story, with coups in Canada and Iceland that come from the simulator's safe-country tail, misleads more than it
    # explains; the tail itself is an open item in docs/world_system.md)
    for key, fn in [("score_2025", "score_2025.json"), ("crisis_explorer", "crisis_explorer.json"),
                    ("us_history", "us_history.json"), ("seshat", "seshat.json"), ("calibration_top", "calibration_top.json"),
                    ("monthly", "monthly_2025_vintage.json"), ("ledger", "ledger_summary.json"),
                    ("crowd", "crowd_benchmark.json"), ("gulf", "gulf_2026.json"), ("leaders", "leader_benchmark.json"),
                    ("crowd_live", "crowd_live.json"), ("predcomp", "predcomp_leaderboard.json"),
                    ("predcomp_pgm", "predcomp_pgm_leaderboard.json"), ("views_archive", "views_archive_benchmark.json"),
                    ("scorecard", "prospective_scorecard.json"), ("climate", "climate_risk.json")]:
        if (OUT / fn).exists():
            extra[key] = json.loads((OUT / fn).read_text())
    wb = _worldbuilder()
    if wb:
        extra["worldbuilder"] = wb
    ch = _release_changes()
    if ch:
        extra["changes"] = ch
    ts = OUT / "tech_scenarios.json"
    if ts.exists():
        from .tech import TECHS
        extra["tech"] = {"scen": json.loads(ts.read_text()),
                         "registry": {k: [v[0], v[1], sorted(v[4])] for k, v in TECHS.items()}}
    ua, ur = OUT / "uncertainty_attribution.json", OUT / "uncertainty_ranges.json"
    if ua.exists():
        att = json.loads(ua.read_text())
        extra["uncertainty"] = {"n": next(iter(att.values()), {}).get("n"),
                                "attribution": {k: {"assumptions_all": v["assumptions_all"], "groups": v.get("groups", {})} for k, v in att.items()},
                                "ranges": json.loads(ur.read_text()) if ur.exists() else {}}
    dis = OUT / "disaster" / "rca.json"
    if dis.exists():
        d = json.loads(dis.read_text())
        extra["disaster"] = {k: d[k] for k in ("n_worlds", "years", "sets", "outcomes_all", "picked", "knockouts") if k in d}
    if "us_history" in extra:
        extra["us_history"].pop("violence_rows", None)
    for k in ("crowd", "leaders"):
        if k in extra:
            extra[k].pop("questions", None)
    page = page.replace("/*__EXTRA__*/null", json.dumps(extra, separators=(",", ":")))
    us = OUT / "us_states.json"
    page = page.replace("/*__US__*/null", us.read_text() if us.exists() else "null")
    page = page.replace("/*__USTOPO__*/null", (RAW / "us" / "states-10m.json").read_text())
    md = narrative.read_text() if narrative.exists() else ""
    top, _, rest = md.partition("<!--more-->")
    if not rest:                                    # no split marker: everything goes in the detailed commentary
        top, rest = "", top
    top = top.replace("## The short version", "", 1)   # the section heading already says it
    summ = OUT / "summary_template.md"                    # "The future in one page" (top of the report): {name}
    if summ.exists():                                     # placeholders are filled from the results (_summary_vars)
        txt = summ.read_text().format_map(_summary_vars())
        (OUT / "summary.md").write_text(txt)              # (the filled copy, for reading and diffs)
        page = page.replace("<!--__SUMMARY__-->", _md_to_html(txt))
    else:
        page = page.replace("<!--__SUMMARY__-->", "")
    page = page.replace("<!--__NARRATIVE_TOP__-->", _md_to_html(top) if top.strip() else "")
    page = page.replace("<!--__NARRATIVE__-->", _md_to_html(rest) if rest.strip() else
                        "<p class='muted'>No narrative written yet (outputs/narrative.md).</p>")
    dn = OUT / "disaster_notes.md"
    page = page.replace("<!--__DISASTER__-->", _md_to_html(dn.read_text()) if dn.exists() else "")
    imp = OUT / "improvement.md"
    page = page.replace("<!--__IMPROVE__-->", _md_to_html(imp.read_text()) if imp.exists() else "")
    page = page.replace("<!--__REVIEW__-->", _md_to_html(review.read_text()) if review.exists() else "")
    # the artifact host wraps the page in its own document skeleton; the local copy needs one
    (OUT / "seldon_report_artifact.html").write_text(page)
    path = path or OUT / "seldon_report.html"
    path.write_text('<!doctype html>\n<html lang="en"><meta charset="utf-8">\n'
                    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + page + "\n</html>\n")
    return str(path)
