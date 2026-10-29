"""Companion pages for the public site: the methodology (docs/methodology.md) and the data source tree
(raw sources from scripts/draw_dependencies.py, the files under data/raw, each folder's README summary).
python -P scripts/build_site_pages.py  -> outputs/netlify_site/{methodology,data_sources}.html
"""
import os
import re
import sys
from html import escape

import markdown

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
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

NAV = ('<nav class="top"><a href="index.html">Report</a><a href="methodology.html">Methodology</a>'
       '<a href="data_sources.html">Data source tree</a></nav>')
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
        "<h2>Files</h2><p class='muted'>The raw data directory as used for the 29 October 2026 forecast. "
        f"Folders with more than {MAX_FILES} files are summarised by file type.</p>" + "".join(tree))
    return page("Data source tree", body, "Every data source behind the Seldon world crisis forecast.")


def methodology():
    md = open(os.path.join(ROOT, "docs", "methodology.md"), encoding="utf-8").read()
    html = markdown.markdown(md, extensions=["tables"])
    return page("Seldon methodology", html, "How the Seldon world crisis forecasting engine works and how it is tested.")


if __name__ == "__main__":
    os.makedirs(SITE, exist_ok=True)
    for name, fn in (("data_sources.html", data_sources), ("methodology.html", methodology)):
        with open(os.path.join(SITE, name), "w", encoding="utf-8") as f:
            f.write(fn())
        print(os.path.join(SITE, name))
