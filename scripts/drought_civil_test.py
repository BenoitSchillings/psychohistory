"""Does drought predict civil conflict where people farm and groups are politically excluded?

python scripts/drought_civil_test.py
Earlier tests (features.py, "Climate shocks") found that a country's unusual heat, rain or drought in a year adds
nothing to the conflict, coup and collapse forecasts. Research at the group level (von Uexkull et al. 2016, PNAS)
finds drought matters for agriculturally dependent, politically excluded groups. This tests that conditional
version at the country level:
  agri          agriculture, forestry and fishing, % of GDP (World Bank via OWID; carried 10 years either way)
  drought_agri  drought_shock x agri share
  drought_excl  drought_shock x ethnic_excluded (share of population in politically excluded groups, EPR)
  drought_agex  drought_shock x agri share x ethnic_excluded
  dry_agex      (-rain_shock) x agri share x ethnic_excluded
1. logit for civil-conflict onset (1 year, countries at peace) and 5-year civil conflict with a small control set,
   errors clustered by country: does the interaction carry a coefficient?
2. rolling-origin backtest (development origins 1975-2005) of the tuned GBM with and without the new columns.
Writes outputs/drought_civil_test.json.
"""
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

NEW = ["agri", "drought_agri", "drought_excl", "drought_agex", "dry_agex"]


def agri_array(p):
    a = pd.read_csv(RAW / "agriculture-share-gdp.csv")
    col = a.columns[-1]
    piv = a.pivot_table(index="code", columns="year", values=col).reindex(index=list(p.codes), columns=list(p.years))
    piv = piv.T.ffill(limit=10).bfill(limit=10).T
    return piv.to_numpy(float) / 100


def augment(p, X):
    ag = agri_array(p)
    f = lambda n: X[..., FEATURES.index(n)].astype(float)
    dr, rn, ex = f("drought_shock"), f("rain_shock"), f("ethnic_excluded")
    new = np.stack([ag, dr * ag, dr * ex, dr * ag * ex, -rn * ag * ex], -1).astype(np.float32)
    return np.concatenate([X, new], -1)


def logit_test(p, X, Yl):
    import statsmodels.api as sm
    Xf, Yf, ci, year = flatten(p, X, Yl)
    names = FEATURES + NEW
    ctrl = ["log_gdp", "log_pop", "conflict_past_10y", "log_years_since_conflict", "ethnic_excluded", "drought_shock",
            "rain_shock", "agri"]
    ctrl = [c for c in ctrl if c in names]
    out = {}
    for task in ("onset_1y", "civil_conflict_5y"):
        k = TASKS.index(task)
        for inter in ("drought_agri", "drought_excl", "drought_agex", "dry_agex"):
            cols = ctrl + [inter]
            Z = Xf[:, [names.index(c) for c in cols]].astype(float)
            y = Yf[:, k]
            ok = np.isfinite(Z).all(1) & np.isfinite(y) & (year >= 1960)
            Zs = (Z[ok] - Z[ok].mean(0)) / (Z[ok].std(0) + 1e-9)
            m = sm.Logit(y[ok], sm.add_constant(Zs)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": ci[ok]})
            out[f"{task}:{inter}"] = {"coef_sd": round(float(m.params[-1]), 3), "se": round(float(m.bse[-1]), 3),
                                      "p": round(float(m.pvalues[-1]), 3), "n": int(ok.sum()), "events": int(y[ok].sum())}
    return out


def main():
    p, base, X, Yl = prepare()
    missing = [n for n in ("log_gdp", "log_pop") if n not in FEATURES]
    print("features not found (skipped):", missing)
    Xa = augment(p, X)
    res = {"logit": logit_test(p, Xa, Yl)}
    print(json.dumps(res["logit"], indent=1), flush=True)
    fac = lambda: Wrap(M.GBMModel(TUNED_GBM))
    bt = {}
    for name, XX in (("base", X), ("drought", Xa)):
        for split, origins in (("dev", DEV_ORIGINS), ("lock", LOCK_ORIGINS)):
            s, out, _ = evaluate(p, XX, Yl, fac, origins)
            bt[f"{name}:{split}"] = {t: {m: round(v, 4) for m, v in out[t].items() if m in ("auc", "bss")}
                                     for t in ("civil_conflict_5y", "onset_1y")} | {"score": round(s, 4)}
            print(name, split, bt[f"{name}:{split}"], flush=True)
    res["gbm_backtest"] = bt
    json.dump(res, open(ROOT / "outputs" / "drought_civil_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
