"""Live crowd and market forecasts next to the model, and a blend learned on resolved questions.

Sources (public APIs, no keys): Polymarket (gamma-api public search) and Manifold (search-markets). Metaculus needs a
personal token for its API (used only if METACULUS_TOKEN is set; not required).

Matching: a question is matched when it names one or two countries of the model and an event type the model
forecasts:
  interstate force   strike / attack / airstrike / missile / clash / military action between two named states
                     -> the pair's use-of-force probability (interstate.force_5y)
  interstate war     invade / invasion / war between two named states
                     -> the pair's conflict probability (UCDP >= 25 battle deaths, interstate_5y)
  coup               coup in one named country -> the country's coup_5y
Five-year probabilities become the question's window (now to its close date) with a constant hazard.

Blend: on the resolved crowd questions of scripts/crowd_benchmark.py (use of force, Metaculus / Good Judgment Open /
INFER, 2015-2024), a ridge-penalised logistic regression of the outcome on logit(crowd) and logit(model) (no
intercept; L2 C=1). The crowd carried almost all the weight there (outputs/crowd_live.json reports the weights): the
report shows the crowd where a crowd exists, and the model alone elsewhere.
The ledger's "blend" (2026-10-28) is a fixed logarithmic pool, logit p = 0.9 logit crowd + 0.1 logit model: the fitted
logistic blend was worse than the crowd in leave-one-out (Brier 0.111 vs 0.100), the 0.1 pool slightly better (0.0965;
90% interval of the difference with the crowd -0.011..+0.005, scripts/crowd_pool_test.py), so it is logged to be
tested on new questions, not published.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np

POOL_W = 0.1          # model weight in the logged crowd-model pool (see above)

from .data import RAW, ROOT

UA = {"User-Agent": "psychohistory-research/1.0"}
QUERIES = ["invade", "invasion", "military strike", "airstrike", "war with", "military clash", "attack on", "coup",
           "missile strike", "military action", "ceasefire"]
ALIASES = {"united states": "USA", "u.s.": "USA", "us ": "USA", "usa": "USA", "america": "USA", "russia": "RUS",
           "china": "CHN", "taiwan": "TWN", "iran": "IRN", "israel": "ISR", "ukraine": "UKR", "north korea": "PRK",
           "south korea": "KOR", "india": "IND", "pakistan": "PAK", "venezuela": "VEN", "mexico": "MEX",
           "greenland": "DNK", "denmark": "DNK", "cuba": "CUB", "japan": "JPN", "philippines": "PHL", "poland": "POL",
           "lithuania": "LTU", "latvia": "LVA", "estonia": "EST", "finland": "FIN", "turkey": "TUR", "syria": "SYR",
           "lebanon": "LBN", "iraq": "IRQ", "saudi": "SAU", "yemen": "YEM", "egypt": "EGY", "ethiopia": "ETH",
           "eritrea": "ERI", "sudan": "SDN", "afghanistan": "AFG", "armenia": "ARM", "azerbaijan": "AZE",
           "serbia": "SRB", "kosovo": "XKX", "guyana": "GUY", "colombia": "COL", "panama": "PAN", "canada": "CAN",
           "thailand": "THA", "cambodia": "KHM", "vietnam": "VNM", "nigeria": "NGA", "niger": "NER", "mali": "MLI",
           "burkina": "BFA", "chad": "TCD", "madagascar": "MDG", "guinea": "GIN", "guinea-bissau": "GNB", "equatorial guinea": "GNQ",
           "papua new guinea": "PNG", "south sudan": "SSD", "united kingdom": "GBR", "britain": "GBR", "france": "FRA",
           "germany": "DEU", "gabon": "GAB", "belarus": "BLR",
           "moldova": "MDA", "georgia": "GEO", "rwanda": "RWA", "congo": "COD", "somalia": "SOM", "libya": "LBY",
           "bangladesh": "BGD", "myanmar": "MMR", "nepal": "NPL", "bolivia": "BOL", "peru": "PER", "ecuador": "ECU"}
FORCE = re.compile(r"\b(strike|strikes|attack|attacks|airstrike|air strike|missile|clash|military action|bomb|"
                   r"military engagement|shoot down|military conflict)\b", re.I)
WAR = re.compile(r"\b(invade|invades|invasion|war (with|between|against)|declare war|go to war|ground troops)\b", re.I)
COUP = re.compile(r"\bcoup\b", re.I)


def _get(url, headers=None):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def fetch():
    """open binary questions: [{source, id, question, p, close, volume, url}]"""
    out, seen = [], set()
    for q in QUERIES:
        try:
            d = _get("https://gamma-api.polymarket.com/public-search?" + urllib.parse.urlencode({"q": q, "limit_per_type": 50}))
            for e in d.get("events", []):
                for m in e.get("markets", []):
                    if m.get("closed") or not m.get("outcomePrices"):
                        continue
                    pr = json.loads(m["outcomePrices"]) if isinstance(m["outcomePrices"], str) else m["outcomePrices"]
                    outs = json.loads(m.get("outcomes", '["Yes","No"]')) if isinstance(m.get("outcomes"), str) else m.get("outcomes")
                    if not outs or outs[0].lower() != "yes":
                        continue
                    key = ("polymarket", m.get("id"))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"source": "polymarket", "id": m.get("id"), "question": m.get("question"), "p": float(pr[0]),
                                "close": m.get("endDate"), "volume": float(m.get("volume") or 0),
                                "url": f"https://polymarket.com/event/{e.get('slug')}"})
        except Exception as ex:                       # a failing source is reported, not fatal
            print(f"  polymarket {q!r}: {ex!r}")
        try:
            d = _get("https://api.manifold.markets/v0/search-markets?" + urllib.parse.urlencode({"term": q, "limit": 50, "filter": "open"}))
            for m in d:
                if m.get("outcomeType") != "BINARY" or m.get("isResolved"):
                    continue
                key = ("manifold", m.get("id"))
                if key in seen:
                    continue
                seen.add(key)
                out.append({"source": "manifold", "id": m.get("id"), "question": m.get("question"), "p": float(m.get("probability", np.nan)),
                            "close": dt.datetime.utcfromtimestamp(m["closeTime"] / 1000).isoformat() + "Z" if m.get("closeTime") else None,
                            "volume": float(m.get("volume") or 0), "url": m.get("url")})
        except Exception as ex:
            print(f"  manifold {q!r}: {ex!r}")
        time.sleep(0.3)
    return out


def countries_in(text):
    t = " " + text.lower().replace("’", "'") + " "
    found = {}
    # longest aliases first (so "south sudan" is not read as "sudan"), then ordered by position in the question
    for k, v in sorted(ALIASES.items(), key=lambda kv: -len(kv[0])):
        pat = r"(?<![a-z])" + re.escape(k.strip()) + r"(?![a-z])"
        m = re.search(pat, t)
        if m:
            if v not in found or m.start() < found[v]:
                found[v] = m.start()
            t = re.sub(pat, lambda mm: " " * len(mm.group(0)), t)    # (blank, keeping positions)
    return sorted(found, key=found.get)


def classify(q):
    cs = countries_in(q)
    if COUP.search(q) and len(cs) >= 1:
        return "coup", cs[:1]
    if len(cs) >= 2 and WAR.search(q):
        return "war", cs[:2]
    if len(cs) >= 2 and FORCE.search(q):
        return "force", cs[:2]
    return None, cs


def window_prob(p5, days):
    """constant-hazard conversion of a five-year probability to `days`"""
    p5 = min(max(p5, 1e-6), 1 - 1e-6)
    return float(1 - (1 - p5) ** (max(days, 1) / (5 * 365.25)))


def blend_weights():
    """ridge logistic of outcome on logit(crowd), logit(model) over the resolved use-of-force questions"""
    from sklearn.linear_model import LogisticRegression
    b = json.load(open(ROOT / "outputs" / "crowd_benchmark.json"))
    qs = [q for q in b["questions"] if q.get("snap") == "open"]
    lg = lambda v: np.log(np.clip(v, 0.005, 0.995) / (1 - np.clip(v, 0.005, 0.995)))
    X = np.c_[lg(np.array([q["crowd"] for q in qs])), lg(np.array([q["model"] for q in qs]))]
    y = np.array([q["y"] for q in qs])
    m = LogisticRegression(C=1.0, fit_intercept=True).fit(X, y)
    # leave-one-out Brier of crowd, model, blend
    pb = np.zeros(len(y))
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        pb[i] = LogisticRegression(C=1.0).fit(X[k], y[k]).predict_proba(X[i:i + 1])[0, 1]
    sig = lambda z: 1 / (1 + np.exp(-z))
    return {"w_crowd": float(m.coef_[0][0]), "w_model": float(m.coef_[0][1]), "b": float(m.intercept_[0]), "n": int(len(y)),
            "events": int(y.sum()), "brier_crowd": float(np.mean((sig(X[:, 0]) - y) ** 2)),
            "brier_model": float(np.mean((sig(X[:, 1]) - y) ** 2)), "brier_blend_loo": float(np.mean((pb - y) ** 2))}


def match(questions, forecast):
    """attach model probabilities and the blend"""
    I = forecast.get("interstate", {})
    pairs = {(r["a"], r["b"]): r for r in I.get("top", []) + I.get("watch", [])}
    cty = {c["code"]: c for c in forecast["countries"]}
    W = blend_weights()
    now = dt.datetime.now(dt.timezone.utc)
    lg = lambda v: np.log(np.clip(v, 0.005, 0.995) / (1 - np.clip(v, 0.005, 0.995)))
    out = []
    for q in questions:
        kind, cs = classify(q["question"] or "")
        if kind is None or not q.get("close") or not np.isfinite(q["p"]):
            continue
        try:
            close = dt.datetime.fromisoformat(q["close"].replace("Z", "+00:00"))
        except ValueError:
            continue
        days = (close - now).days
        if days < 7 or days > 5 * 366:
            continue
        p5 = None
        if kind == "coup" and cs[0] in cty:
            p5 = cty[cs[0]].get("coup_5y")
        elif kind in ("force", "war"):
            a, b = sorted(cs[:2])
            r = pairs.get((a, b))
            if r is not None:
                p5 = r["p_force_5y"] if kind == "force" else r["p_conflict_5y"]
                if r.get("observed") == 1.0 or r.get("ongoing") == 1.0:
                    p5 = None                 # already fighting: the model's probability is not a forecast
        if p5 is None:
            continue
        pm = window_prob(p5, days)
        pbl = float(1 / (1 + np.exp(-((1 - POOL_W) * lg(q["p"]) + POOL_W * lg(pm)))))
        out.append({**q, "kind": kind, "countries": cs, "days": days, "model": round(pm, 4), "blend": round(pbl, 4)})
    out.sort(key=lambda r: -r["volume"])
    return out, W
