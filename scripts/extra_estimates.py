"""Estimates behind the 2026-10-26 additions (independent review of missing factors). python scripts/extra_estimates.py

C1 financial-crisis hazard on growth, public debt, its 5-year change and inflation (Laeven-Valencia crises in the
   panel), 1955-2024, country-clustered; out-of-sample check (fit < 2000, score 2000-2024) against the simulator's
   current two-bin rule (growth below -5% or not)
C3 mineral rents (World Bank NY.GDP.MINR.RT.ZS) as a civil-conflict predictor: logit with the naive controls and
   tuned-GBM backtest with and without
D1 fertility and income: 10-year change in log fertility on 10-year change in log GDP per person, with year effects
D2 cereal yields and weather shocks (ERA5 tas, txx, cdd, pr anomalies against the country's previous 10 years),
   country and year effects; and the cross-country correlation of those anomalies (how much a global harvest
   shock is shared), 1980-2022
Writes outputs/extra_estimates.json.
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
from sklearn.metrics import roc_auc_score

from psychohistory.data import RAW, ROOT, get_panel
from psychohistory.features import base_arrays

p = get_panel()
b = base_arrays(p)
Y = p.years
C = len(p.codes)
out = {}


def long(**arrs):
    """country-year frame from [C, T] arrays"""
    ci, ti = np.meshgrid(np.arange(C), np.arange(len(Y)), indexing="ij")
    d = {"c": ci.ravel(), "year": Y[ti].ravel()}
    d.update({k: np.asarray(v, float).ravel() for k, v in arrs.items()})
    return pd.DataFrame(d)


lag = lambda a, k=1: np.concatenate([np.full((C, k), np.nan), a[:, :-k]], 1)

# ---------------------------------------------------------------- C1
g = b["lg"] - lag(b["lg"])
debt = b["public_debt"]
d = long(crisis=b["fin_crisis"], g=np.clip(g, -0.3, 0.3), ldebt=np.log1p(np.clip(lag(debt), 0, None)),
         ddebt=lag(debt) - lag(debt, 6), linfl=np.log1p(np.clip(lag(b["inflation"]), 0, 1000) / 100))
d = d[(d.year >= 1955) & (d.year <= 2024)].dropna()
d["low"] = (d.g < -0.05).astype(float)
cols = ["g", "low", "ldebt", "ddebt", "linfl"]
m = sm.Logit(d.crisis, sm.add_constant(d[cols])).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.c})
out["C1_crisis_logit"] = {"coef": dict(zip(["const"] + cols, m.params.round(4).tolist())),
                          "se": dict(zip(["const"] + cols, m.bse.round(4).tolist())), "n": int(len(d)), "rate": float(d.crisis.mean())}
tr, te = d.year < 2000, d.year >= 2000
m2 = sm.Logit(d.crisis[tr], sm.add_constant(d.loc[tr, cols])).fit(disp=0)
pr_new = m2.predict(sm.add_constant(d.loc[te, cols]))
two = d[tr].groupby("low").crisis.mean()
pr_old = d.loc[te, "low"].map(two)
y = d.crisis[te]
bs = lambda q: float(np.mean((q - y) ** 2))
out["C1_oos_2000_2024"] = {"auc_new": float(roc_auc_score(y, pr_new)), "auc_two_bin": float(roc_auc_score(y, pr_old)),
                           "brier_new": bs(pr_new), "brier_two_bin": bs(pr_old), "brier_clim": bs(np.full(len(y), d.crisis[tr].mean())),
                           "n": int(te.sum()), "events": int(y.sum())}
print("C1", json.dumps(out["C1_crisis_logit"]), json.dumps(out["C1_oos_2000_2024"]), flush=True)


# ---------------------------------------------------------------- World Bank series as [C, T]
def wb(ind):
    j = json.load(open(RAW / "wb" / f"{ind}.json"))[1]
    w = pd.DataFrame([(r["countryiso3code"], int(r["date"]), r["value"]) for r in j if r["value"] is not None],
                     columns=["code", "year", "v"])
    piv = w.pivot_table(index="code", columns="year", values="v").reindex(index=list(p.codes), columns=list(Y))
    return piv.to_numpy(float)


# ---------------------------------------------------------------- C3
rents = wb("NY.GDP.MINR.RT.ZS")
rents_f = pd.DataFrame(rents).T.ffill(limit=5).T.to_numpy()
from psychohistory.features import FEATURES, TASKS
from psychohistory.pipeline import prepare
from psychohistory.backtest import flatten
_, _, X, Yl = prepare()
Xf, Yf, ci, yr = flatten(p, X, Yl)
rf = rents_f.ravel()
names = ["log_gdp", "log_pop", "conflict_past_10y", "log_years_since_conflict"]
res = {}
for task in ("onset_1y", "civil_conflict_5y"):
    k = TASKS.index(task)
    Z = np.c_[Xf[:, [FEATURES.index(n) for n in names]], np.log1p(rf)]
    ok = np.isfinite(Z).all(1) & np.isfinite(Yf[:, k]) & (yr >= 1970)
    Zs = (Z[ok] - Z[ok].mean(0)) / Z[ok].std(0)
    mm = sm.Logit(Yf[ok, k], sm.add_constant(Zs)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": ci[ok]})
    res[task] = {"coef_sd_log_rents": round(float(mm.params[-1]), 3), "se": round(float(mm.bse[-1]), 3), "n": int(ok.sum())}
out["C3_rents_logit"] = res
sys.path.insert(0, os.path.dirname(__file__))
from exp import Wrap, TUNED_GBM
from psychohistory import models as M
from psychohistory.evaluate import evaluate, DEV_ORIGINS
Xa = np.concatenate([X, np.log1p(rents_f)[..., None].astype(np.float32)], -1)
bt = {}
for nm, XX in (("base", X), ("rents", Xa)):
    s, o, _ = evaluate(p, XX, Yl, lambda: Wrap(M.GBMModel(TUNED_GBM)), DEV_ORIGINS)
    bt[nm] = {"score": round(s, 4), **{t: {kk: round(float(v), 4) for kk, v in o[t].items() if kk in ("auc", "bss")} for t in ("civil_conflict_5y", "onset_1y")}}
out["C3_rents_gbm_dev"] = bt
print("C3", json.dumps(res), json.dumps(bt), flush=True)

# ---------------------------------------------------------------- D1
lf = np.log(b["fertility"])
d = long(dlf=lf - lag(lf, 10), dlg=b["lg"] - lag(b["lg"], 10), lf0=lag(lf, 10))
d = d[(d.year >= 1970) & (d.year <= 2023) & (d.year % 5 == 0)].dropna()
Xd = pd.get_dummies(d.year, prefix="y", drop_first=True).astype(float)
Xd["dlg"], Xd["lf0"] = d.dlg, d.lf0
m = sm.OLS(d.dlf, sm.add_constant(Xd)).fit(cov_type="cluster", cov_kwds={"groups": d.c})
out["D1_fertility_income"] = {"elasticity_10y": round(float(m.params["dlg"]), 4), "se": round(float(m.bse["dlg"]), 4),
                              "n": int(len(d))}
print("D1", json.dumps(out["D1_fertility_income"]), flush=True)

# ---------------------------------------------------------------- D2
an = {}
for v in ("tas", "txx", "cdd", "pr"):
    a = p.x[f"c_{v}"].astype(float)
    if v == "pr":
        a = np.log(np.clip(a, 1, None))
    ref = pd.DataFrame(a).T.rolling(10, min_periods=5).mean().shift(1).T.to_numpy()
    an[v] = a - ref
ly = np.log(np.clip(b["cereal_yield"], 1e-3, None))
d = long(dly=ly - lag(ly), **{f"a_{v}": an[v] for v in an})
d = d[(d.year >= 1962) & (d.year <= 2022)].dropna()
d = d[d.dly.abs() < 0.7]
Xd = pd.get_dummies(d.year, prefix="y", drop_first=True).astype(float)
for v in an:
    Xd[f"a_{v}"] = d[f"a_{v}"]
Xd = Xd.join(pd.get_dummies(d.c, prefix="c", drop_first=True).astype(float))
m = sm.OLS(d.dly, sm.add_constant(Xd)).fit(cov_type="cluster", cov_kwds={"groups": d.c})
out["D2_yield_weather"] = {v: {"coef": round(float(m.params[f"a_{v}"]), 4), "se": round(float(m.bse[f"a_{v}"]), 4)} for v in an}
out["D2_yield_weather"]["n"] = int(len(d))
# cross-country correlation of anomalies 1980-2022
t = (Y >= 1980) & (Y <= 2022)
corr = {}
for v in an:
    A = an[v][:, t]
    ok = np.isfinite(A).all(1)
    R = np.corrcoef(A[ok])
    corr[v] = {"mean_offdiag": float((R.sum() - ok.sum()) / (ok.sum() * (ok.sum() - 1))), "n_countries": int(ok.sum())}
out["D2_anomaly_correlation"] = corr
print("D2", json.dumps(out["D2_yield_weather"]), json.dumps(corr), flush=True)
json.dump(out, open(ROOT / "outputs" / "extra_estimates.json", "w"), indent=1)
