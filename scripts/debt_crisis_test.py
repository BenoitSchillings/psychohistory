"""Sovereign debt crises and currency crises, tested on history (Global Macro Database crisis dates, 1960-2024).
1. Hazard: what predicts a crisis starting (logit; fit 1960-1999, scored 2000-2024 against a base model with income and
   the year's world rate): public debt / GDP, its 5-year change, last year's growth, inflation (log), the real interest
   rate minus growth, a banking crisis last year, the long-term rate.
2. Cost: growth in the crisis year and the next two (local projection, country and year effects removed by demeaning),
   fit on 1960-1999 and on 2000-2024 separately.
3. Debt dynamics: does the interest-growth gap times last year's debt improve the yearly debt change out of sample
   (fit 1960-1999, scored 2000-2024)?
python -P scripts/debt_crisis_test.py -> outputs/debt_crisis_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import statsmodels.api as sm

from psychohistory.data import get_panel, RAW, ROOT
from psychohistory.pipeline import base_arrays

p = get_panel(False)
b = base_arrays(p)
yrs = np.asarray(p.years)
C, T = len(p.codes), len(yrs)
g = pd.read_csv(RAW / "GMD.csv", usecols=["ISO3", "year", "SovDebtCrisis", "CurrencyCrisis"])
ci, yi = {c: i for i, c in enumerate(p.codes)}, {y: j for j, y in enumerate(yrs)}
crisis = {k: np.full((C, T), np.nan) for k in ("SovDebtCrisis", "CurrencyCrisis")}
for r in g.itertuples():
    if r.ISO3 in ci and r.year in yi:
        for k in crisis:
            v = getattr(r, k)
            if pd.notna(v):
                crisis[k][ci[r.ISO3], yi[r.year]] = float(v)
lag = lambda a, k=1: np.concatenate([np.full((C, k), np.nan), a[:, :-k]], 1)
lg = b["lg"]
growth = lg - lag(lg)
debt = b["public_debt"]
infl = np.sign(b["inflation"]) * np.log1p(np.abs(np.clip(b["inflation"], -99, 1e4)))
# (features.real_rate clips inflation at 100% but not the nominal rate, so hyperinflations give huge positive real
#  rates exactly when inflation is eroding debt; here the real rate is kept only where inflation is under 50% and the
#  rate within +-30 points - review fix, 2026-11-02. The deficit is the overall one, interest included: no primary
#  balance in GMD for most countries)
rr = np.where((np.abs(b["inflation"]) < 50) & (np.abs(b["real_rate"]) < 30), b["real_rate"], np.nan)
r_g = rr / 100 - growth
X = {"debt": lag(debt) / 100, "debt_chg5": lag(b["debt_change_5y"]) / 100, "growth": lag(growth), "infl": lag(infl),
     "r_minus_g": lag(r_g), "bank_crisis": lag(b["fin_crisis"]), "long_rate": lag(b["long_rate"]) / 100}
Y_ = np.broadcast_to(yrs[None], (C, T))
out = {}


def logit_test(name, onset):
    res = {}
    base_keys = ["lg"]
    allX = dict(X, lg=lag(lg))
    lean = ["debt", "debt_chg5", "growth", "infl", "bank_crisis"]
    # (every spec is scored on the same rows: those where the lean predictors exist; the rate spec on its own rows)
    rows = np.isfinite(onset) & np.isfinite(np.stack([allX[k] for k in base_keys + lean], -1)).all(-1)
    for spec, ks in (("base", base_keys), ("lean", base_keys + lean), ("lean+rates", base_keys + lean + ["r_minus_g", "long_rate"])):
        D = np.stack([allX[k] for k in ks], -1)
        ok = rows & np.isfinite(D).all(-1) if spec != "lean+rates" else np.isfinite(onset) & np.isfinite(D).all(-1)
        dev, test = ok & (Y_ >= 1960) & (Y_ < 2000), ok & (Y_ >= 2000)
        Xd = sm.add_constant(D[dev]); yd = onset[dev]
        m = sm.Logit(yd, Xd).fit(disp=0)
        pt = m.predict(sm.add_constant(D[test], has_constant="add"))
        yt = onset[test]
        ll = -np.mean(yt * np.log(np.clip(pt, 1e-6, 1)) + (1 - yt) * np.log(np.clip(1 - pt, 1e-6, 1)))
        from sklearn.metrics import roc_auc_score
        res[spec] = {"n_dev": int(dev.sum()), "events_dev": int(yd.sum()), "n_test": int(test.sum()), "events_test": int(yt.sum()),
                     "test_logloss": round(float(ll), 5), "test_auc": round(float(roc_auc_score(yt, pt)), 4) if 0 < yt.sum() < len(yt) else None,
                     "coef": {k: (round(float(c), 3), round(float(t), 1)) for k, c, t in zip(["const"] + ks, m.params, m.tvalues)}}
    out[f"hazard_{name}"] = res
    print(name, "hazard:", {s: (r["test_logloss"], r["test_auc"], r["events_test"]) for s, r in res.items()})
    print("   lean coefficients (coef, t):", res["lean"]["coef"])


def cost_test(name, onset):
    res = {}
    for h in (0, 1, 2):
        yh = np.concatenate([growth[:, h:], np.full((C, h), np.nan)], 1) if h else growth
        for split, (a, b_) in (("dev", (1960, 2000)), ("test", (2000, 2025))):
            ok = np.isfinite(yh) & np.isfinite(onset) & (Y_ >= a) & (Y_ < b_)
            yv = np.where(ok, yh, np.nan)
            # remove year effects and country means (two-way demeaning)
            yv = yv - np.nanmean(yv, 0, keepdims=True)
            yv = yv - np.nanmean(yv, 1, keepdims=True)
            xv = np.where(ok, onset, np.nan)
            xv = xv - np.nanmean(xv, 0, keepdims=True)
            xv = xv - np.nanmean(xv, 1, keepdims=True)
            m_ = ok & np.isfinite(yv) & np.isfinite(xv)
            fit = sm.OLS(yv[m_], sm.add_constant(xv[m_])).fit(cov_type="cluster", cov_kwds={"groups": np.broadcast_to(np.arange(C)[:, None], (C, T))[m_]})
            res[f"h{h}_{split}"] = (round(float(fit.params[1]), 4), round(float(fit.tvalues[1]), 1))
    out[f"cost_{name}"] = res
    print(name, "growth cost (coef, t) by year after onset:", res)


for k, nm in (("SovDebtCrisis", "sovereign"), ("CurrencyCrisis", "currency")):
    c = crisis[k]
    onset = np.where(np.isfinite(c) & np.isfinite(lag(c)), ((c == 1) & (lag(c) == 0)).astype(float), np.nan)
    print(nm, "onsets 1960-1999:", int(np.nansum(onset[:, (yrs >= 1960) & (yrs < 2000)])), "2000-2024:", int(np.nansum(onset[:, yrs >= 2000])))
    logit_test(nm, onset)
    cost_test(nm, onset)

# debt dynamics: yearly change in debt / GDP against the interest-growth gap x last year's debt
dd = (debt - lag(debt)) / 100
for spec, ks in (("base", ["growth_now", "def"]), ("with_r_g", ["growth_now", "def", "rg_debt"])):
    Z = {"growth_now": growth, "def": -b["gov_deficit"] / 100 if "gov_deficit" in b else np.zeros_like(growth),
         "rg_debt": (rr / 100 - growth) * lag(debt) / 100}
    D = np.stack([Z[k] for k in ks], -1)
    # (both specs on the same rows: those where the clean rate exists)
    ok = np.isfinite(dd) & np.isfinite(D).all(-1) & np.isfinite(rr) & (np.abs(dd) < 0.5)
    dev, test = ok & (Y_ >= 1960) & (Y_ < 2000), ok & (Y_ >= 2000)
    m = sm.OLS(dd[dev], sm.add_constant(D[dev])).fit()
    pt = m.predict(sm.add_constant(D[test], has_constant="add"))
    r2 = 1 - np.mean((dd[test] - pt) ** 2) / np.mean((dd[test] - dd[test].mean()) ** 2)
    out[f"debt_dyn_{spec}"] = {"test_r2": round(float(r2), 4), "n_test": int(test.sum()),
                               "coef": {k: (round(float(c), 3), round(float(t), 1)) for k, c, t in zip(["const"] + ks, m.params, m.tvalues)}}
    print("debt dynamics", spec, out[f"debt_dyn_{spec}"])
json.dump(out, open(os.path.join(ROOT, "outputs", "debt_crisis_test.json"), "w"), indent=1)
