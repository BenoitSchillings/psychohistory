"""12-month conflict probability: the product of 12 monthly forecasts treated as independent (monthly.annual_probability)
against a model fitted directly on "any 25-death month in the next 12" (monthly.fit_annual). Issued each December
2017-2023 with data to that month (final GED), scored on the next calendar year: Brier, log loss, calibration by
bins (expected calibration error), for all countries and for countries without a 25-death month in the past year.
python -P scripts/monthly_annual_test.py -> outputs/monthly_annual_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import monthly as MO
from psychohistory.data import get_panel


def scores(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    return {"brier": round(float(np.mean((p - y) ** 2)), 5), "logloss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5),
            "ece": round(float(ece), 4), "mean_p": round(float(p.mean()), 4), "rate": round(float(y.mean()), 4)}


def main():
    gw2iso, codes = MO.gw_map()
    D = MO.deaths(codes, gw2iso, 2025, 12)
    F, hit = MO.features(D, codes, get_panel())
    out, P = {}, {"product": [], "direct": [], "y": [], "new": []}
    for yr in range(2017, 2024):
        t = MO.midx(yr, 12)
        prod = MO.annual_probability(MO.predict(MO.fit(F, hit, t), F, t))
        direct = MO.predict_annual(MO.fit_annual(F, hit, t), F, t)
        y = hit[:, t + 1:t + 13].max(1)
        new = hit[:, t - 11:t + 1].max(1) == 0
        out[yr] = {k: scores(v, y) for k, v in (("product", prod), ("direct", direct))}
        out[yr]["new_only"] = {k: scores(v[new], y[new]) for k, v in (("product", prod), ("direct", direct))}
        for k, v in (("product", prod), ("direct", direct), ("y", y), ("new", new)):
            P[k].append(v)
        print(yr, json.dumps(out[yr]), flush=True)
    y, new = np.concatenate(P["y"]), np.concatenate(P["new"])
    out["pooled"] = {k: scores(np.concatenate(P[k]), y) for k in ("product", "direct")}
    out["pooled_new_only"] = {k: scores(np.concatenate(P[k])[new], y[new]) for k in ("product", "direct")}
    print(json.dumps({k: out[k] for k in ("pooled", "pooled_new_only")}, indent=1))
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "monthly_annual_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
