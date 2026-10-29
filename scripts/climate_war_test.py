"""Does sustained warming, rather than a year's unusual weather, predict civil conflict where people farm, are poor
or are politically excluded?

python scripts/climate_war_test.py
The yearly weather shocks in features.py (heat, rain, drought against the past decade) add nothing, nor does drought
for farming, excluded populations (drought_civil_test.py) or humid heat (heat_test.py). Those are short-run shocks;
the risk that worries forecasters is the slow climb. This tests the sustained version, at the country level:
  warm10        10-year mean temperature minus the country's 1961-1990 mean (ERA5), degC
  heat10        10-year mean hottest-day temperature minus its 1961-1990 mean, degC
  warm10_agri   warm10 x agriculture share of GDP
  warm10_poor   warm10 x poverty (minus log GDP per person, centred)
  warm10_agex   warm10 x agriculture share x excluded population share (EPR)
  heat10_agri   heat10 x agriculture share
  hot_agri      warm10 x agriculture share x baseline heat (1961-1990 mean temperature above 20 degC)
1. logit for civil-conflict onset (1 year, countries at peace) and 5-year civil conflict, with controls including a
   linear year trend (warming rises with time everywhere), errors clustered by country;
2. rolling-origin backtest of the tuned GBM with and without the columns, development origins (1975-2005) and the
   lockbox (2010, 2015).
Writes outputs/climate_war_test.json.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory.pipeline import prepare
from psychohistory.features import FEATURES, TASKS
from psychohistory.data import ROOT
from psychohistory.backtest import flatten
from psychohistory import models as M
from psychohistory.evaluate import evaluate, DEV_ORIGINS, LOCK_ORIGINS
sys.path.insert(0, os.path.dirname(__file__))
from exp import Wrap, TUNED_GBM
from drought_civil_test import agri_array

NEW = ["warm10", "heat10", "warm10_agri", "warm10_poor", "warm10_agex", "heat10_agri", "hot_agri"]


def sustained(p, v):
    """10-year trailing mean minus the 1961-1990 mean, per country-year."""
    x = p.x[f"c_{v}"].astype(float)
    yrs = np.asarray(p.years)
    ref = np.nanmean(x[:, (yrs >= 1961) & (yrs <= 1990)], 1, keepdims=True)
    out = np.full_like(x, np.nan)
    for i in range(9, x.shape[1]):
        w = x[:, i - 9:i + 1]
        ok = np.isfinite(w).sum(1) >= 7
        out[ok, i] = np.nanmean(w[ok], 1)
    return out - ref, ref[:, 0]


def augment(p, X):
    ag = agri_array(p)
    w10, t_ref = sustained(p, "tas")
    h10, _ = sustained(p, "txx")
    f = lambda n: X[..., FEATURES.index(n)].astype(float)
    lg, ex = f("log_gdp"), f("ethnic_excluded")
    poor = -(lg - np.nanmean(lg))
    hot = (t_ref > 20).astype(float)[:, None]
    new = np.stack([w10, h10, w10 * ag, w10 * poor, w10 * ag * ex, h10 * ag, w10 * ag * hot], -1).astype(np.float32)
    return np.concatenate([X, new], -1)


def logit_test(p, X, Yl):
    import statsmodels.api as sm
    Xf, Yf, ci, year = flatten(p, X, Yl)
    names = FEATURES + NEW
    ctrl = ["log_gdp", "log_pop", "conflict_past_10y", "log_years_since_conflict", "ethnic_excluded", "temp_shock"]
    ctrl = [c for c in ctrl if c in names]
    ag = agri_array(p)
    out = {}
    for task in ("onset_1y", "civil_conflict_5y"):
        k = TASKS.index(task)
        for inter in NEW:
            extra = ["warm10"] if inter.startswith("warm10_") or inter == "hot_agri" else []
            extra += ["heat10"] if inter == "heat10_agri" else []
            cols = ctrl + extra + [inter]
            Z = Xf[:, [names.index(c) for c in cols]].astype(float)
            Z = np.c_[Z, year.astype(float)]                               # common warming trend
            y = Yf[:, k]
            ok = np.isfinite(Z).all(1) & np.isfinite(y) & (year >= 1960)
            Zs = (Z[ok] - Z[ok].mean(0)) / (Z[ok].std(0) + 1e-9)
            j = cols.index(inter)
            m = sm.Logit(y[ok], sm.add_constant(Zs)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": ci[ok]})
            out[f"{task}:{inter}"] = {"coef_sd": round(float(m.params[1 + j]), 3), "se": round(float(m.bse[1 + j]), 3),
                                      "p": round(float(m.pvalues[1 + j]), 3), "n": int(ok.sum()),
                                      "events": int(y[ok].sum())}
    return out


def main():
    p, base, X, Yl = prepare()
    Xa = augment(p, X)
    res = {"logit": logit_test(p, Xa, Yl)}
    print(json.dumps(res["logit"], indent=1), flush=True)
    fac = lambda: Wrap(M.GBMModel(TUNED_GBM))
    bt = {}
    for name, XX in (("base", X), ("climate", Xa)):
        for split, origins in (("dev", DEV_ORIGINS), ("lock", LOCK_ORIGINS)):
            s, out, _ = evaluate(p, XX, Yl, fac, origins)
            bt[f"{name}:{split}"] = {t: {m: round(v, 4) for m, v in out[t].items() if m in ("auc", "bss")}
                                     for t in ("civil_conflict_5y", "onset_1y")} | {"score": round(s, 4)}
            print(name, split, bt[f"{name}:{split}"], flush=True)
    res["gbm_backtest"] = bt
    json.dump(res, open(ROOT / "outputs" / "climate_war_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
