"""Financial contagion through bank lending, tested on history before it goes into the simulator.

Exposure of borrower i in year t: the share of i's foreign bank claims (BIS consolidated banking statistics, immediate
counterparty, end of year t-1) held by banks of countries in a systemic banking crisis in year t (Laeven-Valencia,
as in the panel), plus the same with trade shares (CEPII BACI, as risks.trade_weights) as a control. Outcomes:
a banking-crisis start in i, and a growth collapse (GDP per person falling more than 5%). Logit with year effects
(so a global wave such as 2008 is not counted as contagion: identification is from which countries borrowed from the
banks in crisis that year), clustered by country, 1985-2024; and an out-of-sample check, fit to 2004, scored
2005-2024, against the same model without lender exposure.
python scripts/contagion_test.py  -> outputs/contagion_test.json
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
from psychohistory import risks

p = get_panel()
b = base_arrays(p)
Y = list(p.years)
codes = list(p.codes)
ci = {c: i for i, c in enumerate(codes)}
C = len(codes)
crisis = np.nan_to_num(b["fin_crisis"])
lg = b["lg"]

i23 = risks._iso2to3()
cb = pd.read_csv(RAW / "bis" / "cbs_bilateral.csv")
cb = cb[cb.rep != "5A"]
cb["rep3"], cb["cp3"] = cb.rep.map(i23), cb.cp.map(i23)
cb = cb.dropna(subset=["rep3", "cp3"])
cb = cb[cb.rep3.isin(ci) & cb.cp3.isin(ci) & (cb.rep3 != cb.cp3)]

Tw = np.nan_to_num(risks.trade_weights(codes, np.nan_to_num(np.exp(lg[:, Y.index(2015)] + b["lpop"][:, Y.index(2015)]))))                 # [C, C] trade shares (row: partner shares of i)
np.fill_diagonal(Tw, 0)

rows = []
for t in range(1985, 2025):
    if t not in Y:
        continue
    ti = Y.index(t)
    prev = cb[cb.year == t - 1]
    if prev.empty:
        continue
    W = np.zeros((C, C))
    for r in prev.itertuples():
        W[ci[r.cp3], ci[r.rep3]] += max(r.claims, 0)
    tot = W.sum(1)
    sh = np.divide(W, tot[:, None], out=np.zeros_like(W), where=tot[:, None] > 0)
    cr = crisis[:, ti]
    lend = sh @ cr
    trade = Tw @ cr
    for i in range(C):
        if tot[i] <= 0 or not np.isfinite(lg[i, ti]) or not np.isfinite(lg[i, ti - 1]):
            continue
        rows.append({"c": i, "year": t, "start": float(cr[i] > 0 and crisis[i, ti - 1] == 0),
                     "in_prev": float(crisis[i, ti - 1] > 0), "collapse": float(lg[i, ti] - lg[i, ti - 1] < -0.05),
                     "lend": lend[i], "trade": trade[i], "lclaims": np.log1p(tot[i]),
                     "g_prev": float(np.clip(lg[i, ti - 1] - lg[i, ti - 2], -0.3, 0.3)) if np.isfinite(lg[i, ti - 2]) else np.nan})
d = pd.DataFrame(rows).dropna()
fe = pd.get_dummies(d.year, prefix="y", drop_first=True).astype(float)
out = {"n": int(len(d)), "years": [int(d.year.min()), int(d.year.max())]}
for outcome, sub in (("crisis_start", d[d.in_prev == 0]), ("growth_collapse", d)):
    y = sub["start" if outcome == "crisis_start" else "collapse"]
    res = {}
    for name, cols in (("lend+trade", ["lend", "trade", "lclaims", "g_prev"]), ("trade only", ["trade", "lclaims", "g_prev"])):
        X = sm.add_constant(pd.concat([sub[cols], fe.loc[sub.index]], axis=1))
        X = X.loc[:, X.std() > 0].assign(const=1.0)
        try:
            m = sm.Logit(y, X).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": sub.c})
            res[name] = {k: [round(float(m.params[k]), 3), round(float(m.bse[k]), 3)] for k in cols}
        except Exception as e:                                     # perfect separation in a year cell
            res[name] = repr(e)[:100]
        # out of sample: fit to 2004, score 2005-2024 (year effects replaced by the world crisis share that year)
        tr, te = sub.year <= 2004, sub.year >= 2005
        wshare = sub.year.map(pd.Series(crisis.mean(0), index=Y))
        Z = sm.add_constant(sub[cols].assign(world=wshare))
        mo = sm.Logit(y[tr], Z[tr]).fit(disp=0, maxiter=200)
        pr = mo.predict(Z[te])
        res[name + " oos_brier"] = round(float(np.mean((pr - y[te]) ** 2)), 5)
    res["events"] = int(y.sum())
    res["n"] = int(len(sub))
    out[outcome] = res
    print(outcome, json.dumps(res), flush=True)
json.dump(out, open(os.path.join(ROOT, "outputs", "contagion_test.json"), "w"), indent=1)
# the simulator's contagion keeps the historical crisis rate: first-round crises are drawn at p / mean(exp(beta e))
beta = out["crisis_start"]["lend+trade"]["trade"][0] if isinstance(out["crisis_start"]["lend+trade"], dict) else 4.3
out["trade_exposure"] = {"mean": float(d.trade.mean()), "p90": float(d.trade.quantile(0.9)), "max": float(d.trade.max()),
                         "mean_exp_beta_e": float(np.exp(out["crisis_start"]["trade only"]["trade"][0] * d.trade).mean())}
print(out["trade_exposure"])
json.dump(out, open(os.path.join(ROOT, "outputs", "contagion_test.json"), "w"), indent=1)
