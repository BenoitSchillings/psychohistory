"""Age structure and the workforce: does growth in the working-age share (15-64 / total, UN WPP via OWID) predict growth
of GDP per person beyond the simulator's growth equation inputs?

Accounting: GDP per person = output per worker x workers per person, so a falling working-age share drags growth one
for one unless participation or productivity offset it. Panel 1960-2023, next-year and next-5-year growth per person
regressed on the controls (log GDP gap to the frontier, past growth, conflict, democracy, population growth) plus the
change in log working-age share over the same period (known in advance from demographic projections), and the old-age
share (65+ / 15-64). Year effects, clustered by country; out-of-sample: fit to 2000, score 2001-2023.
python -P scripts/workforce_test.py  -> outputs/workforce_test.json
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import statsmodels.api as sm

from psychohistory.data import RAW, ROOT, get_panel
from psychohistory.features import base_arrays

a = pd.read_csv(RAW / "population-young-working-elderly-with-projections.csv")
cols = {k: [c for c in a.columns if k in c] for k in ("age_0_14", "age_15_64", "age_65plus")}
for k, cs in cols.items():
    a[k] = a[cs].bfill(axis=1).iloc[:, 0]                                  # estimates, else the medium projection
a = a.dropna(subset=["code"]).set_index(["code", "year"])
wshare = np.log(a.age_15_64 / (a.age_0_14 + a.age_15_64 + a.age_65plus))
old = a.age_65plus / a.age_15_64

p = get_panel()
b = base_arrays(p)
Y = list(p.years)
lg = b["lg"]
front = np.nanmax(np.where(np.isfinite(lg), lg, -np.inf), 0)
rows = []
for i, c in enumerate(p.codes):
    for t in range(1960, 2024):
        ti = Y.index(t)
        if not all(np.isfinite(lg[i, ti - k]) for k in range(0, 6)) or not np.isfinite(lg[i, ti + 1]):
            continue
        w = [wshare.get((c, t + k)) for k in (0, 1, 5)]
        if any(v is None or not np.isfinite(v) for v in w):
            continue
        rows.append({"c": c, "year": t, "g1": lg[i, ti + 1] - lg[i, ti],
                     "g5": (lg[i, ti + 5] - lg[i, ti]) / 5 if t + 5 <= 2024 and np.isfinite(lg[i, ti + 5]) else np.nan,
                     "dw1": w[1] - w[0], "dw5": (w[2] - w[0]) / 5, "old": old.get((c, t), np.nan),
                     "gap": front[ti] - lg[i, ti], "g_prev": lg[i, ti] - lg[i, ti - 1], "g_prev5": (lg[i, ti] - lg[i, ti - 5]) / 5,
                     "conflict": float(b["conflict"][i, ti] > 0), "dem": b["dem"][i, ti],
                     "popg": (b["lpop"][i, ti] - b["lpop"][i, ti - 5]) / 5})
d = pd.DataFrame(rows)
for k in ("g1", "g_prev"):
    d[k] = d[k].clip(-0.3, 0.3)
out = {"n": len(d)}
CTRL = ["gap", "g_prev", "g_prev5", "conflict", "dem", "popg"]
for target, dw in (("g1", "dw1"), ("g5", "dw5")):
    res = {}
    for name, extra in (("controls", []), ("+working-age", [dw]), ("+working-age+old", [dw, "old"])):
        sub = d.dropna(subset=[target] + CTRL + extra + ["dw1", "old"])
        fe = pd.get_dummies(sub.year, prefix="y", drop_first=True).astype(float)
        X = pd.concat([sub[CTRL + extra], fe], axis=1).assign(const=1.0)
        m = sm.OLS(sub[target], X).fit(cov_type="cluster", cov_kwds={"groups": sub.c.astype("category").cat.codes})
        res[name] = {k: [round(float(m.params[k]), 4), round(float(m.bse[k]), 4)] for k in extra}
        tr, te = sub.year <= 2000, sub.year >= 2001
        if target == "g5":
            tr = sub.year <= 1995                                       # no 5-year window straddles the split
        Z = sm.add_constant(sub[CTRL + extra])
        mo = sm.OLS(sub[target][tr], Z[tr]).fit()
        res[name]["oos_rmse"] = round(float(np.sqrt(np.mean((mo.predict(Z[te]) - sub[target][te]) ** 2))), 5)
        res[name]["n"] = int(len(sub))
    out[target] = res
    print(target, json.dumps(res, indent=1), flush=True)
# what the projections imply for 2025-2050: mean yearly change in log working-age share
proj = {}
for c in ("CHN", "KOR", "JPN", "DEU", "ITA", "ESP", "USA", "IND", "NGA", "ETH", "PAK", "BRA", "RUS", "IRN", "TUR", "EGY"):
    try:
        proj[c] = round(float((wshare[(c, 2050)] - wshare[(c, 2025)]) / 25 * 100), 2)
    except KeyError:
        pass
out["proj_dw_2025_2050_pct_per_year"] = proj
print("projected working-age share change 2025-2050, % per year:", proj)
json.dump(out, open(ROOT / "outputs" / "workforce_test.json", "w"), indent=1)
