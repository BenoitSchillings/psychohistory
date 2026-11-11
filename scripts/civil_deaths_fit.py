"""Deaths in civil conflicts, for the simulator's population accounting (simulate.CIVIL_DEATHS; 10 November 2026).
Found by the code audit of 10 November 2026: an ongoing conflict kept its origin-year battle-death rate for as long as
it lasted (Israel: 2,781 per million a year, the Gaza war's deaths, which UCDP places in Israel) and every conflict
that started in the simulation had 2 per million a year.
Estimate (UCDP battle deaths per million, country-years 1990-2024 in civil conflict):
  continuing conflict:  log d_t = a + b log d_{t-1} + c log population + e
  first year:           log d = a0 + c0 log population + e0
  e and e0 are drawn from the equations' own residuals (all of them, with replacement): they are skewed and
  fat-tailed, and normal draws put the world's deaths at 34,000 a year against 61,000 recorded.
Population is in both because the rate per person falls almost one-for-one with it (a rebellion is fought in a region,
not in a whole country): without it every country at war had the average country's rate, and the world's deaths came
out about 4.5 times the record (found the same day, on the first version 57 build, before it was published).
The check at the end replays 1990-2024 with the fitted equations on the observed conflict years.
The risk models' battle-death feature is unchanged, so hazards and hindcast scores do not move; only the population.
python -P scripts/civil_deaths_fit.py  -> psychohistory/civil_deaths.json (read by the simulator)"""
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
bd, c, lpop = b["battle_deaths_pc"], np.nan_to_num(b["conflict"]), b["lpop"]
sel = (yrs >= 1990) & (yrs <= 2024)
X, Y, P, F, FP = [], [], [], [], []
for t in np.where(sel)[0]:
    ok = (c[:, t] > 0) & (c[:, t - 1] > 0) & (bd[:, t] > 0) & (bd[:, t - 1] > 0) & np.isfinite(lpop[:, t])
    X += list(np.log(bd[ok, t - 1])); Y += list(np.log(bd[ok, t])); P += list(lpop[ok, t])
    ok = (c[:, t] > 0) & (c[:, t - 1] == 0) & (bd[:, t] > 0) & np.isfinite(lpop[:, t])
    F += list(np.log(bd[ok, t])); FP += list(lpop[ok, t])
X, Y, P, F, FP = map(np.array, (X, Y, P, F, FP))
A = np.c_[np.ones(len(X)), X, P]
co = np.linalg.lstsq(A, Y, rcond=None)[0]
rq = np.sort(Y - A @ co)
A0 = np.c_[np.ones(len(F)), FP]
co0 = np.linalg.lstsq(A0, F, rcond=None)[0]
rq0 = np.sort(F - A0 @ co0)
co, co0 = np.round(co, 3), np.round(co0, 3)

# replay: the observed conflict years, the fitted rates, 400 draws; world deaths a year against the record
m = (c > 0) & sel[None] & np.isfinite(lpop)
obs = np.nansum(np.where(m & (bd > 0), bd * np.exp(lpop) / 1e6, 0), 0)[sel]
rng = np.random.default_rng(0)
R = 400
tot = np.zeros((R, int(sel.sum())))
prev = np.zeros((R, c.shape[0]))
for j, t in enumerate(np.where(sel)[0]):
    u = rng.random((R, c.shape[0]))
    lp = np.nan_to_num(lpop[:, t])[None]
    cont = np.exp(co[0] + co[1] * np.log(np.clip(prev, 0.1, None)) + co[2] * lp + rq[(u * len(rq)).astype(int)])
    new = np.exp(co0[0] + co0[1] * lp + rq0[(u * len(rq0)).astype(int)])
    prev = np.where(m[:, t][None], np.where(prev > 0, cont, new), 0.0).clip(max=1e4)
    tot[:, j] = (prev * np.exp(lp) / 1e6).sum(1)
res = {"fitted_on": "UCDP battle deaths per million, country-years in civil conflict 1990-2024",
       "continuing": {"n": int(len(X)), "a": float(co[0]), "b": float(co[1]), "c_log_pop": float(co[2]),
                      "resid": [round(float(v), 3) for v in rq]},
       "first_year": {"n": int(len(F)), "a0": float(co0[0]), "c0_log_pop": float(co0[1]),
                      "resid": [round(float(v), 3) for v in rq0]},
       "replay_1990_2024": {"observed_deaths_per_year": round(float(obs.mean())),
                            "simulated_mean": round(float(tot.mean())),
                            "simulated_q05_q95_of_period_means": [round(float(v)) for v in np.percentile(tot.mean(1), [5, 95])],
                            "observed_median_year": round(float(np.median(obs))),
                            "simulated_median_year": round(float(np.median(tot)))}}
print(json.dumps({k: v for k, v in res.items() if k == "replay_1990_2024"}, indent=1), co, co0)
json.dump(res, open(os.path.join(ROOT, "psychohistory", "civil_deaths.json"), "w"), indent=1)
