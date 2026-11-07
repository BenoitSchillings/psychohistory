"""Companion pages for the public site: the methodology (docs/methodology.md) and the data source tree
(raw sources from scripts/draw_dependencies.py, the files under data/raw, each folder's README summary).
python -P scripts/build_site_pages.py  -> outputs/netlify_site/{methodology,data_sources,ethics}.html
"""
import os
import re
import sys
from html import escape

import markdown

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
from draw_dependencies import E, N  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
SITE = os.path.join(ROOT, "outputs", "netlify_site")
MAX_FILES = 25  # list a folder's files up to this many, else summarise by type

CSS = """
:root{--bg:#f7f7f5;--card:#ffffff;--ink:#1f2328;--ink2:#57606a;--muted:#6e7781;--line:#d8dee4;--accent:#1d5fbf;--code:#eef1f4}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1115;--card:#171a21;--ink:#e6e8eb;--ink2:#b0b7c0;--muted:#8b949e;--line:#2d333b;--accent:#6ea8fe;--code:#22272e}}
:root[data-theme="dark"]{--bg:#0f1115;--card:#171a21;--ink:#e6e8eb;--ink2:#b0b7c0;--muted:#8b949e;--line:#2d333b;--accent:#6ea8fe;--code:#22272e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 Inter,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:920px;margin:0 auto;padding:32px 16px 80px}
a{color:var(--accent)}
nav.top{font-size:14px;color:var(--muted);margin-bottom:24px}
nav.top a{margin-right:16px}
h1{font-size:30px;line-height:1.2;margin:0 0 8px}
h2{font-size:21px;margin:40px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:17px;margin:24px 0 6px}
p,li{color:var(--ink)}
.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600}
code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:.88em}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:8px 14px}
details details{margin:6px 0;border-radius:6px}
summary{cursor:pointer;font-weight:600}
summary .meta{font-weight:400;color:var(--muted);font-size:14px;margin-left:8px}
ul.files{list-style:none;padding-left:4px;margin:6px 0;font-size:14px}
ul.files li{padding:1px 0}
ul.files .sz{color:var(--muted);margin-left:8px}
.feeds{font-size:14px;color:var(--ink2)}
.figure{background:#fff;border:1px solid var(--line);border-radius:8px;padding:8px;overflow-x:auto}
.figure img{display:block;max-width:none;width:1400px}
footer{margin-top:48px;font-size:14px;color:var(--muted)}
"""

NAV = ('<nav class="top"><a href="tldr.html">In two minutes</a><a href="index.html">Report</a><a href="methodology.html">Methodology</a>'
       '<a href="data_sources.html">Data source tree</a><a href="ethics.html">Purpose and ethics</a></nav>')
CREDIT = "<footer>Study conceived and directed by Benoit Schillings.</footer>"


def page(title, body, desc):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(title)}</title><meta name="description" content="{escape(desc)}">'
            f'<style>{CSS}</style></head><body><main>{NAV}{body}{CREDIT}</main></body></html>')


def size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def readme_summary(folder):
    """First prose paragraph of a folder's README, as plain text."""
    path = os.path.join(folder, "README.md")
    if not os.path.exists(path):
        return ""
    text = open(path, encoding="utf-8", errors="replace").read()
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para and not para.startswith(("#", "|", "```", "*", "-")):
            para = re.sub(r"`([^`]*)`", r"\1", re.sub(r"\*\*([^*]*)\*\*", r"\1", " ".join(para.split())))
            para = para.replace(ROOT + "/", "").replace(os.path.expanduser("~"), "~")
            return para if len(para) < 420 else para[:400].rsplit(" ", 1)[0] + " …"
    return ""


def walk(folder):
    """(files, subfolders, total bytes, file count) for one directory, recursively."""
    files, subs, total, count = [], [], 0, 0
    for name in sorted(os.listdir(folder), key=str.lower):
        path = os.path.join(folder, name)
        if name.startswith(".") or os.path.islink(path):
            continue
        if os.path.isdir(path):
            sub = walk(path)
            subs.append((name, sub))
            total += sub[2]
            count += sub[3]
        else:
            b = os.path.getsize(path)
            files.append((name, b))
            total += b
            count += 1
    return files, subs, total, count


def render_dir(name, node, depth=0, path=""):
    files, subs, total, count = node
    full = os.path.join(RAW, path)
    desc = readme_summary(full) if path else ""
    out = [f'<details{" open" if depth == 0 and count < 15 else ""}><summary>{escape(name)}/'
           f'<span class="meta">{count:,} files · {size(total)}</span></summary>']
    if desc:
        out.append(f'<p class="muted">{escape(desc)}</p>')
    for sname, sub in subs:
        out.append(render_dir(sname, sub, depth + 1, os.path.join(path, sname)))
    if files:
        if len(files) <= MAX_FILES:
            out.append('<ul class="files">' + "".join(
                f'<li><code>{escape(f)}</code><span class="sz">{size(b)}</span></li>' for f, b in files) + "</ul>")
        else:
            by = {}
            for f, b in files:
                ext = os.path.splitext(f)[1].lower() or "(none)"
                n, t = by.get(ext, (0, 0))
                by[ext] = (n + 1, t + b)
            out.append('<ul class="files">' + "".join(
                f'<li>{n:,} <code>{escape(e)}</code> files<span class="sz">{size(t)}</span></li>'
                for e, (n, t) in sorted(by.items(), key=lambda kv: -kv[1][1])) + "</ul>")
    out.append("</details>")
    return "".join(out)


def data_sources():
    from psychohistory.report import VERSION as RVER
    feeds = {}
    for src, dst, *_ in E:
        feeds.setdefault(src, []).append(N[dst][1])
    rows = "".join(
        f"<tr><td><b>{escape(t)}</b></td><td>{escape(d)}</td><td class='feeds'>{escape(', '.join(feeds.get(k, [])))}</td></tr>"
        for k, (layer, t, d, *_) in N.items() if layer == 0)
    top_files, subs, total, count = walk(RAW)
    tree = [render_dir(name, sub, 0, name) for name, sub in subs]
    if top_files:
        tree.append(render_dir("top-level series", (top_files, [], sum(b for _, b in top_files), len(top_files)), 1, ""))
    body = (
        "<h1>Data source tree</h1>"
        f"<p>Every input to the Seldon forecasting engine: {sum(v[0] == 0 for v in N.values())} source families, "
        f"{count:,} files and {size(total)} of raw data. All sources are public; downloads were parsed as data, never executed.</p>"
        "<h2>How data flows</h2><p class='muted'>Raw sources (left) are processed into panels, fitted into models and estimates, "
        "run through the simulator, validated, and written to the published results. Scroll sideways on small screens.</p>"
        '<div class="figure"><a href="data_dependencies.svg"><img src="data_dependencies.svg" alt="Data dependency diagram"></a></div>'
        f"<h2>Source families</h2><table><thead><tr><th>Source</th><th>What it measures</th><th>Feeds</th></tr></thead><tbody>{rows}</tbody></table>"
        f"<h2>Files</h2><p class='muted'>The raw data directory as used for report version {RVER[0]} ({RVER[1]}). "
        f"Folders with more than {MAX_FILES} files are summarised by file type.</p>" + "".join(tree))
    return page("Data source tree", body, "Every data source behind the Seldon simulation of humanity, 2026-2065.")


def methodology():
    md = open(os.path.join(ROOT, "docs", "methodology.md"), encoding="utf-8").read()
    html = markdown.markdown(md, extensions=["tables"])
    return page("Seldon methodology", html, "How the Seldon world crisis forecasting engine works and how it is tested.")


def ethics():
    md = open(os.path.join(ROOT, "docs", "ethics.md"), encoding="utf-8").read()
    html = markdown.markdown(md, extensions=["tables"])
    return page("Purpose and ethics", html, "Why Seldon publishes named-country forecasts, what they are for, and how harm is limited.")


def tldr():
    """the short top layer (reader feedback, 7 November 2026): forecasts, explorations by AI timing, track record, the
    assumptions that decide the long run and the known biases with their direction; every number from the results"""
    import csv
    import json
    sys.path.insert(0, ROOT)
    from psychohistory.report import VERSION
    O = os.path.join(ROOT, "outputs")
    J = lambda f: json.load(open(os.path.join(O, f)))
    fc, rng, cr, sk = J("forecast.json"), J("uncertainty_ranges.json"), J("credence.json"), J("skill_horizon.json")
    led, crowd = J("ledger_summary.json"), J("crowd_benchmark.json")
    hist = J("release_history.json")
    A = fc["ai"]
    pc = lambda x: f"{100 * x:.0f}%"
    k = lambda x: f"${x / 1000:.0f}k"
    rows = list(csv.DictReader(open(os.path.join(O, "country_forecasts.csv"))))
    name = {r["code"]: r["name"] for r in rows}
    num = lambda r, c: float(r[c]) if r.get(c) not in (None, "") else -1.0

    def top(col, cond, n=6):
        rs = sorted((r for r in rows if cond(r)), key=lambda r: -num(r, col))[:n]
        return "".join(f"<tr><td>{i + 1}</td><td>{escape(r['name'])}</td><td>{pc(num(r, col + '_lo'))}–{pc(num(r, col + '_hi'))}</td>"
                       f"<td class='muted'>{pc(num(r, col))}</td></tr>" for i, r in enumerate(rs))
    peace = lambda r: r["in_conflict"] in ("0", "0.0")
    nxt = list(led["annual_new"].items())[:5]
    civ = list(led.get("civilian", {}).get("new", {}).items())[:5]
    g = lambda key: A[key]["world_gdp_q"]
    gdp_rows = "".join(f"<tr><td>{lab}</td><td><b>{k(g(key)[2][-1])}</b></td><td>{k(g(key)[0][-1])}–{k(g(key)[4][-1])}</td></tr>"
                       for key, lab in (("none", "Near-free AI never arrives"), ("2035", "It arrives in 2035"),
                                        ("2030", "It arrives in 2030"), ("2027", "It arrives in 2027")))
    gp = cr["great_power_war"]
    s15 = sk["by_horizon"]["15"]
    D = lambda o, d: rng[o]["drivers"][d]
    ver = [v["headline"]["gpw"] for _, v in sorted(hist.items(), key=lambda kv: tuple(int(x) for x in kv[0].split(".")))]
    oc = crowd.get("open", {})
    body = f"""
<h1>Seldon in two minutes</h1>
<p class="muted">Version {VERSION[0]}, {VERSION[1]}. The short version of the <a href="index.html">full report</a>; how it works and how it is tested is in the <a href="methodology.html">methodology</a>.</p>
<p>Seldon simulates the next forty years for 175 countries, ten thousand times, from eighty years of history. Read it less as an oracle than as an instrument: near the present it makes forecasts that are scored as reality arrives; far out it shows which beliefs about AI, climate and nuclear escalation actually move the outcome.</p>

<h2>Forecasts: scored as reality arrives</h2>
<p><b>Where fighting could start in the next twelve months</b> (countries with none in the past year): {", ".join(f"{escape(name.get(c, c))} {pc(v)}" for c, v in nxt)}. Violence against civilians or between militias: {", ".join(f"{escape(name.get(c, c))} {pc(v)}" for c, v in civ)}.</p>
<h3>New civil war, 2026–2030</h3>
<p class="muted">Ranked, with the range across the model families; the single estimate is shown last and small, because for small states it is the least certain part.</p>
<table><tr><th>#</th><th>Country</th><th>Range</th><th>Estimate</th></tr>{top("civil_conflict_5y", peace)}</table>
<h3>Coup attempt, 2026–2030</h3>
<table><tr><th>#</th><th>Country</th><th>Range</th><th>Estimate</th></tr>{top("coup_5y", lambda r: True)}</table>
<p><b>Track record.</b> Trained on data to 2024, it ranked 2025's one new civil war (South Sudan) 3rd of 142 countries. Its monthly fatality layer ranks 2nd of 21 in the 2023/24 ViEWS challenge. In hindcasts its country forecasts beat fair naive forecasts at every horizon up to 15 years ({"+" + pc(s15["conflict"]["skill"])} on who is at war, {"+" + pc(s15["new_conflict"]["skill"])} on new wars at 15 years). Forecasting crowds beat it on near-term questions about wars between states (Brier {oc.get("brier_crowd", 0):.3f} against {oc.get("brier_model", 0):.3f}); where a crowd covers a question, trust the crowd. Every forecast is frozen and scored as UCDP publishes.</p>

<h2>Explorations to 2065: what follows from stated assumptions</h2>
<p>The long run hinges on when AI makes intellectual work nearly free. The worlds split by that, so the pooled median ({k(A["uncertain"]["world_gdp_q"][2][-1])}) describes few actual worlds. World output per person in 2065 (median, 90% band):</p>
<table><tr><th>If…</th><th>Median</th><th>90% band</th></tr>{gdp_rows}</table>
<p><b>A war between great powers by 2065</b> happens in {pc(A["uncertain"]["great_power_war_by"][-1])} of the headline worlds, but read it as <b>somewhere between {pc(gp["range_across_assumptions_5_95"][0])} and {pc(gp["range_across_assumptions_5_95"][1])}</b>: that is the spread across draws of the assumptions. The number has also moved between model versions, from 69% in October to {ver[-1]} today, through bug fixes and design choices; model choices are part of the uncertainty. Counting only the traditional great powers, {pc(A["uncertain"]["great_power_war_cow_by"][-1])}. Nuclear weapons are used in {pc(cr["nuclear_use"]["overall"])} of worlds ({pc(cr["nuclear_use"]["range_across_assumptions_5_95"][0])}–{pc(cr["nuclear_use"]["range_across_assumptions_5_95"][1])}).</p>

<h2>The four assumptions that decide the long run</h2>
<table><tr><th>Assumption (history cannot test it)</th><th>Moves</th><th>From lowest to highest fifth of its range</th></tr>
<tr><td>When near-free AI arrives</td><td>World output per person, 2065</td><td>{k(D("log_world_gdp_2065", "ai_arrival")["low_fifth"])} (early) to {k(D("log_world_gdp_2065", "ai_arrival")["high_fifth"])} (late)</td></tr>
<tr><td>Which estimate of climate damage is right</td><td>World output per person, 2045</td><td>{k(D("log_world_gdp_2045", "dmg_family")["low_fifth"])} to {k(D("log_world_gdp_2045", "dmg_family")["high_fifth"])}</td></tr>
<tr><td>How readily nuclear-armed rivals fight</td><td>Great-power war by 2065</td><td>{pc(D("great_power_war", "nuclear_war_factor")["low_fifth"])} to {pc(D("great_power_war", "nuclear_war_factor")["high_fifth"])}</td></tr>
<tr><td>How often allies join a war</td><td>Great-power war by 2065</td><td>{pc(D("great_power_war", "cascade_q")["low_fifth"])} to {pc(D("great_power_war", "cascade_q")["high_fifth"])}</td></tr></table>

<h2>Known biases, and which way they push</h2>
<ul>
<li><b>Wars between states: too low.</b> It missed the 2010s rise two- to threefold, and it gave the 2025 Thailand–Cambodia war under 1% and the 2026 US–Iran war 0.1%. Its rivalry data end in 2010, so recent rivalries are partly invisible.</li>
<li><b>The safest countries: too high.</b> It expected two or three new civil wars among the 20 safest countries in 15-year windows since 1995; none happened.</li>
<li><b>World recessions: too rare.</b> Simulated world output falls in under 2% of years, against 3–10% recorded.</li>
<li><b>Single countries' income: bands too wide.</b> Countries drift apart 1.3–1.9 times too much since the 1990s.</li>
<li><b>World growth: no stable direction.</b> In 15-year hindcasts it ran 11–18% high from 1970s starts and 3–17% low from 1990s starts, so the headline is not corrected.</li>
</ul>

<p class="muted">Purpose and use of the country numbers: <a href="ethics.html">purpose and ethics</a>. Full version history and every test: <a href="methodology.html">methodology</a>.</p>
"""
    return page("Seldon in two minutes", body, "The short version of the Seldon report: forecasts, explorations by AI timing, track record and known biases.")


if __name__ == "__main__":
    os.makedirs(SITE, exist_ok=True)
    for name, fn in (("data_sources.html", data_sources), ("methodology.html", methodology), ("ethics.html", ethics), ("tldr.html", tldr)):
        with open(os.path.join(SITE, name), "w", encoding="utf-8") as f:
            f.write(fn())
        print(os.path.join(SITE, name))
