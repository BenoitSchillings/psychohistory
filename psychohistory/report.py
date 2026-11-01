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
# bump on every published change to the report (version 39 follows the 38 earlier published revisions)
VERSION = ("44", "1 November 2026")


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
        return re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)

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
    for key, fn in [("score_2025", "score_2025.json"), ("crisis_explorer", "crisis_explorer.json"), ("chronicles", "chronicles.json"),
                    ("us_history", "us_history.json"), ("seshat", "seshat.json"), ("calibration_top", "calibration_top.json"),
                    ("monthly", "monthly_2025_vintage.json"), ("ledger", "ledger_summary.json"),
                    ("crowd", "crowd_benchmark.json"), ("gulf", "gulf_2026.json"), ("leaders", "leader_benchmark.json"),
                    ("crowd_live", "crowd_live.json"), ("predcomp", "predcomp_leaderboard.json"),
                    ("predcomp_pgm", "predcomp_pgm_leaderboard.json"), ("views_archive", "views_archive_benchmark.json"),
                    ("scorecard", "prospective_scorecard.json"), ("climate", "climate_risk.json")]:
        if (OUT / fn).exists():
            extra[key] = json.loads((OUT / fn).read_text())
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
