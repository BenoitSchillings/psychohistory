"""Violence beyond battles (second debate, 6 November 2026): the monthly layer forecast only UCDP state-based violence
(type 1, the ViEWS target). Non-state fighting between militias or communities (type 2) and one-sided violence against
civilians (type 3) drive much of the death toll in Darfur, eastern DRC or the Sahel. Same features and direct 12-month
model (monthly.fit_annual) on "any month with 25 or more deaths from types 2 and 3 in the country", issued each
December 2017-2023 with data to that month and scored on the next year, against persistence with its historical rates
(P(a month in the next 12 | one in the past 12) and P(... | none), estimated on the 10 years before each issue).
python -P scripts/monthly_civilian_test.py -> outputs/monthly_civilian_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import monthly as MO
from psychohistory.data import get_panel


def sc(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"brier": round(float(np.mean((p - y) ** 2)), 5),
            "logloss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5), "mean_p": round(float(p.mean()), 4),
            "rate": round(float(y.mean()), 4)}


def main():
    gw2iso, codes = MO.gw_map()
    D = MO.deaths(codes, gw2iso, 2025, 12, types=(2, 3))
    F, hit = MO.features(D, codes, get_panel())
    out, P = {}, {"m": [], "r": [], "y": [], "new": []}
    for yr in range(2017, 2024):
        t = MO.midx(yr, 12)
        pm = MO.predict_annual(MO.fit_annual(F, hit, t), F, t)
        past = hit[:, t - 11:t + 1].max(1)
        a_ = [(hit[:, s - 11:s + 1].max(1), hit[:, s + 1:s + 13].max(1)) for s in range(t - 12 - 120, t - 11, 12)]
        was = np.concatenate([a for a, _ in a_]); nxt = np.concatenate([b for _, b in a_])
        r_in, r_out = nxt[was == 1].mean(), nxt[was == 0].mean()
        pr = np.where(past == 1, r_in, r_out)
        y = hit[:, t + 1:t + 13].max(1)
        out[yr] = {"model": sc(pm, y), "persistence": sc(pr, y)}
        for k, v in (("m", pm), ("r", pr), ("y", y), ("new", past == 0)):
            P[k].append(v)
        print(yr, json.dumps(out[yr]), flush=True)
    m, r, y, new = (np.concatenate(P[k]) for k in ("m", "r", "y", "new"))
    out["pooled"] = {"model": sc(m, y), "persistence": sc(r, y),
                     "brier_skill": round(float(1 - np.mean((m - y) ** 2) / np.mean((r - y) ** 2)), 4)}
    out["pooled_new_only"] = {"model": sc(m[new], y[new]), "persistence": sc(r[new], y[new]),
                              "brier_skill": round(float(1 - np.mean((m[new] - y[new]) ** 2) / np.mean((r[new] - y[new]) ** 2)), 4)}
    print(json.dumps({k: out[k] for k in ("pooled", "pooled_new_only")}, indent=1))
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "monthly_civilian_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
