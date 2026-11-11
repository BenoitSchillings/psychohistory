"""Deaths in civil conflicts, for the simulator's population accounting (simulate.CIVIL_DEATHS; 10 November 2026).
Found by the code audit of 10 November 2026: an ongoing conflict kept its origin-year battle-death rate for as long as
it lasted (Israel: 2,781 per million a year, the Gaza war's deaths, which UCDP places in Israel) and every conflict
that started in the simulation had 2 per million a year.
Estimate (UCDP battle deaths per million, country-years 1990-2024 in civil conflict):
  continuing conflict:  log d_t = a + b log d_{t-1} + e,  e ~ N(0, sd)
  first year:           lognormal(median m0, log sd sd0) from the quartiles
The risk models' battle-death feature is unchanged, so hazards and hindcast scores do not move; only the population.
python -P scripts/civil_deaths_fit.py  -> outputs/civil_deaths_fit.json"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from psychohistory.data import get_panel          # noqa: E402
from psychohistory.features import base_arrays    # noqa: E402

p = get_panel()
b = base_arrays(p)
yrs = np.array(p.years)
bd, c = b["battle_deaths_pc"], np.nan_to_num(b["conflict"])
X, Y, first = [], [], []
for t in np.where((yrs >= 1990) & (yrs <= 2024))[0]:
    ok = (c[:, t] > 0) & (c[:, t - 1] > 0) & (bd[:, t] > 0) & (bd[:, t - 1] > 0)
    X += list(np.log(bd[ok, t - 1])); Y += list(np.log(bd[ok, t]))
    first += list(bd[(c[:, t] > 0) & (c[:, t - 1] == 0) & (bd[:, t] > 0), t])
X, Y = np.array(X), np.array(Y)
A = np.c_[np.ones(len(X)), X]
co = np.linalg.lstsq(A, Y, rcond=None)[0]
sd = float(np.std(Y - A @ co))
q25, q50, q75 = np.percentile(first, [25, 50, 75])
sd0 = float((np.log(q75) - np.log(q25)) / 1.349)
m = (c > 0) & (bd > 0) & ((yrs >= 1990) & (yrs <= 2024))[None]
res = {"continuing": {"n": int(len(X)), "a": round(float(co[0]), 3), "b": round(float(co[1]), 3), "resid_sd": round(sd, 2),
                      "stationary_mean": round(float(np.exp(co[0] / (1 - co[1]) + 0.5 * sd ** 2 / (1 - co[1] ** 2))), 1)},
       "first_year": {"n": int(len(first)), "median": round(float(q50), 1), "log_sd": round(sd0, 2),
                      "implied_mean": round(float(q50 * np.exp(0.5 * sd0 ** 2)), 1), "observed_mean": round(float(np.mean(first)), 1)},
       "observed_mean_all_conflict_years": round(float(np.mean(bd[m])), 1),
       "CIVIL_DEATHS": [round(float(co[0]), 3), round(float(co[1]), 3), round(sd, 2), round(float(q50), 1), round(sd0, 2)]}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(ROOT, "outputs", "civil_deaths_fit.json"), "w"), indent=1)
