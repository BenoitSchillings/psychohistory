"""Turchin's popular immiseration on 19th- and 20th-century data (Clio-Infra, CC BY; the series end 2000-2013, so they
cannot feed a forecast from 2025): do falling real wages, shrinking heights and strike waves come before civil war,
coups and autocratisation, beyond income, population, democracy and conflict history? Logit, errors clustered by
country, label years 1850-2005 and the pre-1946 / post-1946 halves separately.
python -P scripts/clio_history_test.py -> outputs/new_data/clio_history.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import statsmodels.api as sm

from psychohistory.backtest import flatten
from psychohistory.data import RAW, ROOT
from psychohistory.features import FEATURES, TASKS
from psychohistory.pipeline import prepare

CL = RAW / "newdata" / "clio"


def series(p, name):
    d = pd.read_excel(CL / f"{name}.xlsx", sheet_name=0)
    iso = pd.read_csv(RAW / "iso3166.csv", dtype={"country-code": str})
    num2a3 = dict(zip(iso["country-code"].astype(int), iso["alpha-3"]))
    d["iso3"] = d.ccode.map(lambda c: num2a3.get(int(c)) if pd.notna(c) else None)
    yrs = [c for c in d.columns if str(c).isdigit()]
    l = d.melt(id_vars=["iso3"], value_vars=yrs, var_name="year", value_name="v").dropna()
    l["year"] = l.year.astype(int)
    a = l.pivot_table(index="iso3", columns="year", values="v").reindex(index=list(p.codes), columns=list(p.years)).to_numpy(float)
    return a


def main():
    p, base, X, Yl = prepare()
    X = X.cpu().numpy() if hasattr(X, "cpu") else X
    lag = lambda a, k: np.concatenate([np.full((a.shape[0], k), np.nan), a[:, :-k]], 1)
    ff = lambda a, n: pd.DataFrame(a).T.interpolate(limit=n, limit_area="inside").ffill(limit=n).T.to_numpy(float)
    wage = np.log(np.where(series(p, "LabourersRealWage") > 0, series(p, "LabourersRealWage"), np.nan))
    wage = ff(wage, 5)
    height = ff(series(p, "Height"), 10)                 # birth-decade cohorts
    pop = np.exp(base["lpop"])
    strikes = np.log1p(ff(series(p, "NumberofWorkersInvolvedinLabourDisputes"), 2) / pop * 1e3)   # workers per 1,000 people
    new = {"real_wage_change_10y": wage - lag(wage, 10), "real_wage_drawdown": wage - pd.DataFrame(wage).T.rolling(20, min_periods=5).max().T.to_numpy(),
           "height_change": height - lag(height, 20), "strikers_per_1000": strikes, "strikers_change_5y": strikes - lag(strikes, 5)}
    Xa = np.concatenate([X, np.stack(list(new.values()), -1)], -1)
    names = FEATURES + list(new)
    Xf, Yf, ci, year = flatten(p, Xa, Yl)
    ctrl = ["log_gdp", "growth_5y", "log_pop", "conflict_past_10y", "log_years_since_conflict", "democracy"]
    res = {}
    for task in ("civil_conflict_5y", "onset_1y", "coup_5y", "autocratization_5y", "econ_collapse_5y"):
        k = TASKS.index(task)
        for col in new:
            for era, lo, hi in (("1850-2005", 1850, 2005), ("1850-1945", 1850, 1945), ("1946-2005", 1946, 2005)):
                Z = Xf[:, [names.index(c) for c in ctrl + [col]]].astype(float); y = Yf[:, k]
                ok = np.isfinite(Z).all(1) & np.isfinite(y) & (year >= lo) & (year <= hi)
                if ok.sum() < 300 or y[ok].sum() < 15:
                    continue
                Zs = (Z[ok] - Z[ok].mean(0)) / (Z[ok].std(0) + 1e-9)
                try:
                    m = sm.Logit(y[ok], sm.add_constant(Zs)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": ci[ok]})
                except Exception:
                    continue
                res[f"{task}:{col}:{era}"] = {"coef_sd": round(float(m.params[-1]), 3), "se": round(float(m.bse[-1]), 3),
                                               "p": round(float(m.pvalues[-1]), 4), "n": int(ok.sum()), "events": int(y[ok].sum())}
    for k, v in res.items():
        flag = " *" if v["p"] < 0.01 else ""
        print(f"{k:58s} {v['coef_sd']:+.2f} ({v['se']:.2f}) n={v['n']} ev={v['events']}{flag}")
    json.dump(res, open(os.path.join(ROOT, "outputs", "new_data", "clio_history.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
