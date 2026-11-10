"""Direct 12-month conflict model with leaves sized in country-years instead of overlapping country-months (third
debate, 9 November 2026): monthly.fit_annual min_child_samples 200 (about 17 country-years) against 2400 (200), and
600 (50) reported. Same protocol as scripts/monthly_annual_test.py: issued each December 2017-2023 (final GED), scored
on the next calendar year.
Pre-registered (before the run): adopt 2400 for the February 2027 release if, for countries without a 25-death month
in the past year, Brier AND log loss both improve, and the all-country Brier is not worse by more than 0.5%. Also
written: a reliability table for those countries (forecast bins against observed rates).
python -P scripts/monthly_cluster_test.py -> outputs/monthly_cluster_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import monthly as MO
from psychohistory.data import get_panel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monthly_annual_test import scores

VARIANTS = {"m200": 200, "m600": 600, "m2400": 2400}
BINS = [0, .01, .02, .05, .1, .2, .35, 1.0]


def reliability(p, y):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            out.append({"bin": f"{lo}-{hi}", "n": int(m.sum()), "forecast": round(float(p[m].mean()), 4), "observed": round(float(y[m].mean()), 4)})
    return out


def main():
    gw2iso, codes = MO.gw_map()
    D = MO.deaths(codes, gw2iso, 2025, 12)
    F, hit = MO.features(D, codes, get_panel())
    P = {k: [] for k in VARIANTS}
    Y, NEW, out = [], [], {}
    for yr in range(2017, 2024):
        t = MO.midx(yr, 12)
        y = hit[:, t + 1:t + 13].max(1)
        new = hit[:, t - 11:t + 1].max(1) == 0
        row = {}
        for k, mc in VARIANTS.items():
            p = MO.predict_annual(MO.fit_annual(F, hit, t, min_child=mc), F, t)
            P[k].append(p)
            row[k] = {"all": scores(p, y), "new_only": scores(p[new], y[new])}
        Y.append(y), NEW.append(new)
        out[yr] = row
        print(yr, json.dumps({k: (v["new_only"]["brier"], v["all"]["brier"]) for k, v in row.items()}), flush=True)
    y, new = np.concatenate(Y), np.concatenate(NEW)
    pooled = {k: {"all": scores(np.concatenate(v), y), "new_only": scores(np.concatenate(v)[new], y[new]),
                  "reliability_new_only": reliability(np.concatenate(v)[new], y[new])} for k, v in P.items()}
    out["pooled"] = pooled
    a, b = pooled["m200"], pooled["m2400"]
    out["adopt_m2400"] = bool(b["new_only"]["brier"] < a["new_only"]["brier"] and b["new_only"]["logloss"] < a["new_only"]["logloss"]
                              and b["all"]["brier"] <= 1.005 * a["all"]["brier"])
    print(json.dumps({k: {"new": v["new_only"], "all": v["all"]} for k, v in pooled.items()}, indent=1))
    print("adopt m2400:", out["adopt_m2400"])
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "monthly_cluster_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
