"""Ageing leaders and succession, tested on history.

1. Death in office by age: leader-years 1900-2015 (Archigos 4.1), P(the leader dies or leaves for ill health that year)
   as a logit in age (Gompertz-like) with a linear era trend; the simulator uses it on top of its regime-based turnover.
2. Does a death in office set off instability? A natural death is close to a random shock to who rules (Jones & Olken
   2005 use it the same way). Country-years 1900-2015: outcomes over the next 2 years (coup attempt, civil-conflict
   onset, democracy up or down by more than 0.1, an irregular exit of the successor), regressed on a death-in-office
   indicator, separately for autocracies (democracy < 0.5) and democracies, with controls (democracy, income, conflict,
   recent coups, leader tenure) and year effects, clustered by country.
python -P scripts/succession_test.py  -> outputs/succession_test.json
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

a = pd.read_stata(RAW / "politics" / "archigos" / "Archigos_4.1_stata14.dta")
iso = pd.read_csv(RAW / "politics" / "cow_to_iso3.csv").set_index("ccode").iso3.to_dict()
a["iso3"] = a.ccode.map(iso)
a["start"], a["end"] = pd.to_datetime(a.eindate), pd.to_datetime(a.eoutdate)
a["died"] = a.exit.isin(["Natural Death", "Retired Due to Ill Health"])
a["irregular"] = a.exit.isin(["Irregular", "Foreign"])
a = a.dropna(subset=["iso3", "start", "end"])

# 1. hazard of death / ill-health exit by age
rows = []
for r in a.itertuples():
    if not np.isfinite(r.yrborn):
        continue
    for y in range(max(r.start.year, 1900), min(r.end.year, 2015) + 1):
        rows.append({"age": y - r.yrborn, "year": y, "d": float(r.died and y == r.end.year)})
h = pd.DataFrame(rows)
h = h[(h.age >= 25) & (h.age <= 100)]
X = sm.add_constant(h[["age"]].assign(era=(h.year - 2000) / 100))
m = sm.Logit(h.d, X).fit(disp=0)
out = {"hazard": {"coef": {k: round(float(v), 4) for k, v in m.params.items()},
                  "se": {k: round(float(v), 4) for k, v in m.bse.items()}, "n": int(len(h)), "deaths": int(h.d.sum()),
                  "by_age": {f"{lo}-{lo + 9}": round(float(h[(h.age >= lo) & (h.age < lo + 10)].d.mean()), 4)
                             for lo in range(40, 100, 10)},
                  "fitted_2025": {age: round(float(m.predict(pd.DataFrame({"const": [1.0], "age": [age], "era": [0.25]}))[0]), 4)
                                  for age in (60, 70, 75, 80, 85, 90)}}}
print("death / ill-health exit hazard:", json.dumps(out["hazard"], indent=1), flush=True)

# 2. what follows a death in office
p = get_panel()
b = base_arrays(p)
Y = list(p.years)
ci = {c: i for i, c in enumerate(p.codes)}
death = {}
irreg = {}
for r in a.itertuples():
    if r.iso3 in ci:
        if r.died:
            death[(r.iso3, r.end.year)] = 1.0
        if r.irregular:
            irreg[(r.iso3, r.end.year)] = 1.0
rows = []
for c, i in ci.items():
    for t in range(1900, 2014):
        ti = Y.index(t)
        dem = b["dem"][i, ti]
        if not np.isfinite(dem) or not np.isfinite(b["dem"][i, ti + 2]):
            continue
        conf = b["conflict"][i, ti]
        rows.append({
            "c": c, "year": t, "death": death.get((c, t), 0.0), "autoc": float(dem < 0.5), "dem": dem,
            "lg": b["lg"][i, ti] if np.isfinite(b["lg"][i, ti]) else np.nan, "conflict": float(conf > 0),
            "coup_recent": float(np.nan_to_num(b["coup_recent"][i, ti]) > 0),
            "coup2": float(np.nansum(b["coup_attempts"][i, ti + 1:ti + 3]) > 0),
            "onset2": float(conf == 0 and np.nansum(b["conflict"][i, ti + 1:ti + 3]) > 0) if conf == 0 else np.nan,
            "dem_up": float(b["dem"][i, ti + 2] - dem > 0.1), "dem_down": float(b["dem"][i, ti + 2] - dem < -0.1),
            "irreg2": float(irreg.get((c, t + 1), 0) + irreg.get((c, t + 2), 0) > 0)})
d = pd.DataFrame(rows).dropna(subset=["lg"])
res = {}
for regime, sub0 in (("autocracy", d[d.autoc == 1]), ("democracy", d[d.autoc == 0])):
    rr = {"deaths": int(sub0.death.sum()), "n": int(len(sub0))}
    for y in ("coup2", "onset2", "dem_up", "dem_down", "irreg2"):
        sub = sub0.dropna(subset=[y])
        sub = sub[sub.groupby("year")[y].transform("max") > 0]
        fe = pd.get_dummies(sub.year, prefix="y", drop_first=True).astype(float)
        X = pd.concat([sub[["death", "dem", "lg", "conflict", "coup_recent"]], fe], axis=1).assign(const=1.0)
        X = X.loc[:, X.std() > 0].assign(const=1.0)
        try:
            mm = sm.Logit(sub[y], X).fit(disp=0, maxiter=300, cov_type="cluster",
                                         cov_kwds={"groups": sub.c.astype("category").cat.codes})
            rr[y] = {"coef": round(float(mm.params["death"]), 3), "se": round(float(mm.bse["death"]), 3),
                     "rate_after_death": round(float(sub[sub.death == 1][y].mean()), 3),
                     "rate_otherwise": round(float(sub[sub.death == 0][y].mean()), 3)}
        except Exception as e:
            rr[y] = repr(e)[:100]
    res[regime] = rr
out["after_death"] = res
print("after a death in office:", json.dumps(res, indent=1))
json.dump(out, open(ROOT / "outputs" / "succession_test.json", "w"), indent=1)
