"""Dated coups in the 12-month coup forecast (data debate, 10 November 2026). The Cline Center dated coups could not be
downloaded (the repository refuses automated and browser downloads alike); REIGN (One Earth Future, to August 2021)
carries the Powell-Thyne attempts and successes by country-month and is used instead. The panel already has REIGN's
leader variables at each year's end (tenure, military career, months since an irregular leader change); what dating
adds is coup recency: months since the last attempt (failed ones included), attempts in the last 12 and 60 months, a
success in the last 12 months, all measured at the end of the issue year.

The model under test stacks the existing 12-month coup forecast (backtest task coup_1y, the Seldon ensemble, from
outputs/backtest_predictions.parquet) with those recency features in a logistic regression, fitted rolling: the row for
year t (forecasting a coup attempt in t+1) is predicted by a fit on rows t' <= t-2 only, whose outcomes were known.
Control: the same stack with the ensemble's log-odds alone (recalibration without dating).

Pre-registered (10 November 2026, before the first run). Scored rows t = 2000-2019 (attempts in 2001-2020), outcome the
panel's coup_1y label (Powell-Thyne via V-Dem, the same events as REIGN's). Adopt for the February 2027 release if:
1. pooled Brier AND log loss improve on the ensemble;
2. Brier improves in at least 3 of the 4 five-year blocks (2001-05, 2006-10, 2011-15, 2016-20);
3. pooled Brier is also better than the control (the gain comes from dating, not from recalibration).
Reported, not part of the decision: REIGN's own CoupCast risk at December of t, made 12-monthly as 1-(1-p)^12 (its
history may be fitted in-sample, so it is not a fair benchmark).
Live use would need dated attempts from September 2021 on, which REIGN does not have.
python -P scripts/coup_dated_test.py -> outputs/coup_dated_test.json"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def recency():
    """REIGN monthly attempts -> features at December of each year, by ISO3 code"""
    from psychohistory.data import REIGN_NAME_FIX, RAW, SERIES
    names = pd.read_csv(RAW / f"{SERIES['pop'][0]}.csv")[["entity", "code"]].drop_duplicates()
    n2c = dict(zip(names.entity, names.code))
    r = pd.read_csv(RAW / "REIGN_2021_8.csv", low_memory=False,
                    usecols=["country", "year", "month", "pt_attempt", "pt_suc", "couprisk"])
    r["code"] = r.country.map(lambda n: n2c.get(REIGN_NAME_FIX.get(n, n)))
    r = r.dropna(subset=["code", "year", "month"])
    r["m"] = (r.year.astype(int) * 12 + r.month.astype(int) - 1)
    # one row per country-month (leader changes inside a month give duplicates)
    r = r.groupby(["code", "m"]).agg(att=("pt_attempt", "max"), suc=("pt_suc", "max"), risk=("couprisk", "last")).reset_index()
    out = []
    for code, g in r.groupby("code"):
        g = g.set_index("m").reindex(range(g.m.min(), g.m.max() + 1))
        att, suc = g.att.fillna(0).values, g.suc.fillna(0).values
        ms = np.full(len(g), 600.0)
        last = None
        for i in range(len(g)):
            if att[i] > 0:
                last = i
            if last is not None:
                ms[i] = min(i - last, 600)
        c12 = pd.Series(att).rolling(12, min_periods=1).sum().values
        c60 = pd.Series(att).rolling(60, min_periods=1).sum().values
        s12 = pd.Series(suc).rolling(12, min_periods=1).sum().values
        mm = np.array(g.index)
        dec = (mm % 12) == 11
        out.append(pd.DataFrame({"code": code, "year": mm[dec] // 12, "lms": np.log1p(ms[dec]), "a12": (c12[dec] > 0) * 1.0,
                                 "a60": np.log1p(c60[dec]), "s12": (s12[dec] > 0) * 1.0, "reign": g.risk.values[dec]}))
    return pd.concat(out)


def scores(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(np.mean((p - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    from sklearn.linear_model import LogisticRegression
    b = pd.read_parquet(os.path.join(ROOT, "outputs", "backtest_predictions.parquet"))
    b = b[(b.task == "coup_1y") & (b.model == "Seldon ensemble")][["code", "year", "p", "y"]].dropna()
    d = b.merge(recency(), on=["code", "year"], how="left")
    print("rows", len(d), "with REIGN recency", int(d.lms.notna().sum()), flush=True)
    # countries outside REIGN: no attempt recorded = the far end of recency
    d = d.fillna({"lms": np.log1p(600), "a12": 0.0, "a60": 0.0, "s12": 0.0})
    d["z"] = lg(d.p.values)
    F1, F0 = ["z", "lms", "a12", "a60", "s12"], ["z"]
    d["p_dated"] = np.nan
    d["p_ctrl"] = np.nan
    for t in range(2000, 2020):
        tr, te = d.year <= t - 2, d.year == t
        for col, F in (("p_dated", F1), ("p_ctrl", F0)):
            m = LogisticRegression(C=1.0, max_iter=2000).fit(d.loc[tr, F], d.loc[tr, "y"])
            d.loc[te, col] = m.predict_proba(d.loc[te, F])[:, 1]
    s = d[(d.year >= 2000) & (d.year <= 2019)].copy()
    res = {"rows": int(len(s)), "events": int(s.y.sum()), "pooled": {}, "blocks": {}}
    for col in ("p", "p_ctrl", "p_dated"):
        res["pooled"][col] = dict(zip(("brier", "logloss"), [round(v, 6) for v in scores(s[col].values, s.y.values)]))
    for a in (2000, 2005, 2010, 2015):
        q = s[(s.year >= a) & (s.year <= a + 4)]
        res["blocks"][f"{a + 1}-{a + 5}"] = {col: round(scores(q[col].values, q.y.values)[0], 6) for col in ("p", "p_ctrl", "p_dated")}
    r = s.dropna(subset=["reign"])
    r12 = 1 - (1 - r.reign.clip(0, 1)) ** 12
    res["reign_rows"] = {"rows": int(len(r)), "reign_coupcast": round(scores(r12.values, r.y.values)[0], 6),
                         "ensemble": round(scores(r.p.values, r.y.values)[0], 6), "dated": round(scores(r.p_dated.values, r.y.values)[0], 6)}
    m = LogisticRegression(C=1.0, max_iter=2000).fit(d.loc[d.year <= 2017, F1], d.loc[d.year <= 2017, "y"])
    res["coefficients_through_2017"] = dict(zip(F1, [round(float(c), 3) for c in m.coef_[0]]))
    P = res["pooled"]
    c1 = P["p_dated"]["brier"] < P["p"]["brier"] and P["p_dated"]["logloss"] < P["p"]["logloss"]
    c2 = sum(v["p_dated"] < v["p"] for v in res["blocks"].values()) >= 3
    c3 = P["p_dated"]["brier"] < P["p_ctrl"]["brier"]
    res["criteria"] = {"1_pooled_brier_and_logloss": c1, "2_three_of_four_blocks": c2, "3_beats_control": c3}
    res["adopt"] = bool(c1 and c2 and c3)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT, "outputs", "coup_dated_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
