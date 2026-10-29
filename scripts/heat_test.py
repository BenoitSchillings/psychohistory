"""Does humid heat (ERA5 wet-bulb, scripts/heat_build.py) predict conflict, coups or economic collapse beyond what the
model already knows (including dry heat: heat_shock, from the hottest-day temperature)?

New columns (country-year, data to 2025):
  tw_level   hottest-month wet-bulb, area-mean (C)
  tw_shock   this year's tw_level minus the country's mean of the previous 10 years, over their s.d.
  tw_hot24   share of the country's area whose hottest-month wet-bulb reaches 24 C
  tw_ta_gap  tw_shock minus the dry-temperature shock built the same way (humidity beyond heat)
1. logit for each task with a small control set (log GDP, log population, conflict history, dry heat_shock), errors
   clustered by country: does the new column carry a coefficient?
2. rolling-origin backtest of the tuned GBM with and without the columns: development origins 1975-2005, then the
   2010 / 2015 lockbox. A data family is kept only if it helps on development and does not hurt on the lockbox.
python -P scripts/heat_test.py -> outputs/heat_test.json"""
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

NEW = ["tw_level", "tw_shock", "tw_hot24", "tw_ta_gap"]


def shock(a):
    out = np.full(a.shape, np.nan)
    for t in range(10, a.shape[1]):
        past = a[:, t - 10:t]
        out[:, t] = (a[:, t] - np.nanmean(past, 1)) / (np.nanstd(past, 1) + 1e-6)
    return out


def heat_arrays(p):
    h = pd.read_csv(RAW / "heat" / "wetbulb_country_year.csv")
    piv = lambda c: h.pivot_table(index="code", columns="year", values=c).reindex(index=list(p.codes), columns=list(p.years)).to_numpy(float)
    tw, hot, ta = piv("tw_mean"), piv("tw_hot24"), piv("ta_mean")
    s_tw = shock(tw)
    return np.stack([tw, s_tw, hot, s_tw - shock(ta)], -1).astype(np.float32)


def logit_test(p, X, Yl):
    import statsmodels.api as sm
    Xf, Yf, ci, year = flatten(p, X, Yl)
    names = FEATURES + NEW
    ctrl = ["log_gdp", "log_pop", "conflict_past_10y", "log_years_since_conflict", "heat_shock"]
    out = {}
    for task in ("onset_1y", "civil_conflict_5y", "coup_1y", "econ_collapse_5y"):
        k = TASKS.index(task)
        for new in NEW:
            cols = ctrl + [new]
            Z = Xf[:, [names.index(c) for c in cols]].astype(float)
            y = Yf[:, k]
            ok = np.isfinite(Z).all(1) & np.isfinite(y) & (year >= 1970)
            Zs = (Z[ok] - Z[ok].mean(0)) / (Z[ok].std(0) + 1e-9)
            m = sm.Logit(y[ok], sm.add_constant(Zs)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": ci[ok]})
            out[f"{task}:{new}"] = {"coef_sd": round(float(m.params[-1]), 3), "se": round(float(m.bse[-1]), 3),
                                    "p": round(float(m.pvalues[-1]), 3), "n": int(ok.sum()), "events": int(y[ok].sum())}
    return out


def main():
    p, base, X, Yl = prepare()
    Xa = np.concatenate([X, heat_arrays(p)], -1)
    res = {"logit": logit_test(p, Xa, Yl)}
    print(json.dumps(res["logit"], indent=1), flush=True)
    fac = lambda: Wrap(M.GBMModel(TUNED_GBM))
    bt = {}
    for name, XX in (("base", X), ("heat", Xa)):
        for split, origins in (("dev", DEV_ORIGINS), ("lock", LOCK_ORIGINS)):
            s, out, _ = evaluate(p, XX, Yl, fac, origins)
            bt[f"{name}:{split}"] = {t: {m: round(v, 4) for m, v in out[t].items() if m in ("auc", "bss")}
                                     for t in ("civil_conflict_5y", "onset_1y", "coup_1y", "econ_collapse_5y") if t in out} | {"score": round(s, 4)}
            print(name, split, bt[f"{name}:{split}"], flush=True)
    res["gbm_backtest"] = bt
    json.dump(res, open(ROOT / "outputs" / "heat_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
