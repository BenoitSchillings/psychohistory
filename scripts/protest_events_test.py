"""Do protest and unrest indices from machine-coded events (ICEWS 1995-2022, GDELT 2023-2025; data/raw/events/
protest_build.py) predict conflict, coups or economic collapse beyond what the model already knows, including the
Mass Mobilization protest counts (protest_recent, to 2020) and NAVCO campaigns (to 2019)?

New columns (country-year; NaN before 1995):
  protest_ev_z   within-year z-score of the protest share of the country's domestic events
  unrest_ev_z    the same for material conflict (protest, coercion, assault)
  protest_ev_3y  mean of protest_ev_z over the past 3 years
  protest_surge  protest_ev_z minus its mean over the previous 3 years
1. logit for each task with a small control set (log GDP, log population, conflict history, protest_recent), errors
   clustered by country, years 1996 on: does the new column carry a coefficient?
2. rolling-origin backtest of the tuned GBM with and without the columns: development origins 1975-2005, then the
   2010 / 2015 lockbox. A data family is kept only if it helps on development and does not hurt on the lockbox.
python -P scripts/protest_events_test.py -> outputs/protest_events_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

from psychohistory.pipeline import prepare
from psychohistory.features import FEATURES, TASKS
from psychohistory.data import RAW, ROOT
from psychohistory.backtest import flatten
from psychohistory import models as M
from psychohistory.evaluate import evaluate, DEV_ORIGINS, LOCK_ORIGINS
sys.path.insert(0, os.path.dirname(__file__))
from exp import Wrap, TUNED_GBM

NEW = ["protest_ev_z", "unrest_ev_z", "protest_ev_3y", "protest_surge"]


def event_arrays(p):
    d = pd.read_csv(RAW / "events" / "protest_cy_spliced.csv")
    piv = lambda c: d.pivot_table(index="iso3", columns="year", values=c).reindex(index=list(p.codes), columns=list(p.years)).to_numpy(float)
    pz, uz = piv("protest_ev_z"), piv("unrest_ev_z")
    lag3 = np.full(pz.shape, np.nan)
    for t in range(3, pz.shape[1]):
        lag3[:, t] = np.nanmean(pz[:, t - 3:t], 1)
    m3 = np.full(pz.shape, np.nan)
    for t in range(2, pz.shape[1]):
        m3[:, t] = np.nanmean(pz[:, t - 2:t + 1], 1)
    return np.stack([pz, uz, m3, pz - lag3], -1).astype(np.float32)


def logit_test(p, X, Yl):
    import statsmodels.api as sm
    Xf, Yf, ci, year = flatten(p, X, Yl)
    names = FEATURES + NEW
    ctrl = ["log_gdp", "log_pop", "conflict_past_10y", "log_years_since_conflict", "protest_recent"]
    out = {}
    for task in ("onset_1y", "civil_conflict_5y", "coup_1y", "econ_collapse_5y"):
        k = TASKS.index(task)
        for new in NEW:
            cols = ctrl + [new]
            Z = Xf[:, [names.index(c) for c in cols]].astype(float)
            y = Yf[:, k]
            ok = np.isfinite(Z).all(1) & np.isfinite(y) & (year >= 1996)
            Zs = (Z[ok] - Z[ok].mean(0)) / (Z[ok].std(0) + 1e-9)
            m = sm.Logit(y[ok], sm.add_constant(Zs)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": ci[ok]})
            out[f"{task}:{new}"] = {"coef_sd": round(float(m.params[-1]), 3), "se": round(float(m.bse[-1]), 3),
                                    "p": round(float(m.pvalues[-1]), 3), "n": int(ok.sum()), "events": int(y[ok].sum())}
    return out


def main():
    p, base, X, Yl = prepare()
    Xa = np.concatenate([X, event_arrays(p)], -1)
    res = {"logit": logit_test(p, Xa, Yl)}
    print(json.dumps(res["logit"], indent=1), flush=True)
    fac = lambda: Wrap(M.GBMModel(TUNED_GBM))
    bt = {}
    for name, XX in (("base", X), ("events", Xa)):
        for split, origins in (("dev", DEV_ORIGINS), ("lock", LOCK_ORIGINS)):
            s, out, _ = evaluate(p, XX, Yl, fac, origins)
            bt[f"{name}:{split}"] = {t: {m: round(v, 4) for m, v in out[t].items() if m in ("auc", "bss")}
                                     for t in ("civil_conflict_5y", "onset_1y", "coup_1y", "econ_collapse_5y") if t in out} | {"score": round(s, 4)}
            print(name, split, bt[f"{name}:{split}"], flush=True)
    res["gbm_backtest"] = bt
    json.dump(res, open(ROOT / "outputs" / "protest_events_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
