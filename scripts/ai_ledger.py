"""AI ledger (docs/ai_ledger.md, pre-registered 6 November 2026): rebuild the arrival assumption mechanically from
the latest value of each of its three sources, and score the AI-macro indicators.

Each source's (year, cumulative probability) points are fitted with a log-normal in years after 2026 (least squares on
the probit scale), conditioned on no arrival before the update date; the three are averaged with equal weight and
read off at the knot years of world.ARRIVAL_CDF. The result is a proposal for the next quarterly model release; the
code is not changed between releases.
python -P scripts/ai_ledger.py [--asof 2026.85] -> outputs/ai_ledger.json"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.stats import norm

ROOT = os.path.join(os.path.dirname(__file__), "..")
T0 = 2026.0


def fit(points):
    """log-normal (mu, sigma) in log(year - T0) through [(year, p)] by least squares on probit(p)"""
    x = np.log(np.array([y for y, _ in points], float) - T0)
    z = norm.ppf(np.clip(np.array([p for _, p in points], float), 1e-4, 1 - 1e-4))
    if len(points) == 1:
        raise ValueError("a source needs at least two points")
    b, a = np.polyfit(x, z, 1)          # z = a + b x  ->  sigma = 1 / b, mu = -a / b
    return -a / b, 1 / b


def cdf(mu, sig, years):
    return norm.cdf((np.log(np.maximum(np.asarray(years, float) - T0, 1e-6)) - mu) / sig)


def main():
    from psychohistory.world import ARRIVAL_CDF
    asof = float(sys.argv[sys.argv.index("--asof") + 1]) if "--asof" in sys.argv else 2026.85
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "ai_ledger", "sources.csv"))))
    latest = max(r["quarter"] for r in rows)
    src = {}
    for r in rows:
        if r["quarter"] == latest or r["source"] not in {q["source"] for q in rows if q["quarter"] == latest}:
            src.setdefault(r["source"], []).append((float(r["year"]), float(r["cum_prob"])))
    knots = [y for y, _ in ARRIVAL_CDF if y > asof]
    per = {}
    for s, pts in src.items():
        mu, sig = fit(pts)
        f0 = cdf(mu, sig, [asof])[0]
        per[s] = {"mu": round(float(mu), 4), "sigma": round(float(sig), 4),
                  "cdf": [round(float((v - f0) / (1 - f0)), 4) for v in cdf(mu, sig, knots)]}
    mix = np.mean([per[s]["cdf"] for s in per], 0)
    cur = dict(ARRIVAL_CDF)
    out = {"asof": asof, "sources_quarter": latest, "per_source": per,
           "proposal": [[y, round(float(v), 3)] for y, v in zip(knots, mix)],
           "current": [[y, cur[y]] for y in knots],
           "note": "a proposal for the next quarterly release; ARRIVAL_CDF changes only at a release (docs/ai_ledger.md)"}
    ind = os.path.join(ROOT, "data", "ai_ledger", "indicators.csv")
    out["indicators"] = list(csv.DictReader(open(ind))) if os.path.exists(ind) else []
    json.dump(out, open(os.path.join(ROOT, "outputs", "ai_ledger.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("proposal", "current")}))


if __name__ == "__main__":
    main()
