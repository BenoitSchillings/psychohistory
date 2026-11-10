"""Monthly conflict model: P(at least 25 state-based battle deaths located in a country in a month).

Features at month t: the country's fighting history (log deaths this month, 3- and 12-month means, the 12-month
maximum, months since the last 25-death month, share of such months in the past two years), structural variables
from the previous calendar year (income, population, democracy, civil conflict, regional conflict) and the
horizon h = 1..12. One LightGBM model is trained on all (t, h) pairs whose target month is observed.

Uses: the 2025 head-to-head with ViEWS (scripts/monthly_nowcast.py), the dated prospective ledger
(scripts/monthly_ledger.py), and the first simulated year (annual probabilities from the last observed month).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import RAW
from .features import base_arrays

O = RAW / "outcomes"
M0 = 1989                                   # month index 0 = 1989-01


def midx(year, month):
    return (year - M0) * 12 + month - 1


def gw_map():
    vw = pd.read_csv(O / "views_2025_cm.csv")
    return dict(zip(vw.gwcode, vw.iso3)), sorted(vw.iso3.unique())


def _events(path, gw2iso, years=None, types=(1,)):
    """types: UCDP type_of_violence (1 state-based, 2 non-state, 3 one-sided violence against civilians)"""
    e = pd.read_csv(path, usecols=["type_of_violence", "country_id", "date_start", "best"])
    e = e[e.type_of_violence.isin(types)]
    e["iso3"] = e.country_id.map(gw2iso)
    e = e.dropna(subset=["iso3"])
    d = pd.to_datetime(e.date_start)
    if years is not None:
        keep = d.dt.year.isin(years)
        e, d = e[keep], d[keep]
    e["m"] = (d.dt.year - M0) * 12 + d.dt.month - 1
    return e.groupby(["iso3", "m"]).best.sum()


FINAL_THROUGH = 2025                        # last year covered by the final GED release in use (26.1)


def candidate_files():
    """newest cumulative UCDP Candidate file per year after FINAL_THROUGH: {year: (path, last full month)}.
    Files are GEDEvent_vYY_01_YY_MM.csv (January to month MM), in data/raw/outcomes/ucdp or .../ucdp/candidate_live
    (where scripts/prospective.py downloads new releases)."""
    import re
    out = {}
    for d in (O / "ucdp", O / "ucdp" / "candidate_live"):
        for f in d.glob("GEDEvent_v*_01_*_*.csv") if d.exists() else []:
            m = re.match(r"GEDEvent_v(\d\d)_01_(\d\d)_(\d\d)\.csv$", f.name)
            if m and m.group(1) == m.group(2):
                y, mo = 2000 + int(m.group(2)), int(m.group(3))
                if y > FINAL_THROUGH and mo > out.get(y, (None, 0))[1]:
                    out[y] = (f, mo)
    return out


def last_observed():
    """(year, month) of the last month with data: the newest candidate file, else December of FINAL_THROUGH"""
    cf = candidate_files()
    if not cf:
        return FINAL_THROUGH, 12
    y = max(cf)
    return y, cf[y][1]


def deaths(codes, gw2iso, end_year, end_month, vintage_2024=False, types=(1,)):
    """country x month state-based deaths, 1989-01 .. end month: final GED 26.1 to 2025-12, then for each later year
    its newest cumulative candidate file; with vintage_2024, 2024 comes from the candidate file released 2025-01-24."""
    M = midx(end_year, end_month) + 1
    D = np.zeros((len(codes), M))
    ci = {c: i for i, c in enumerate(codes)}

    def put(g, lo=None, hi=None):
        for (c, m), v in g.items():
            if c in ci and m < M and (lo is None or lo <= m < hi):
                D[ci[c], m] = v
    put(_events(O / "ucdp" / "ged261-csv" / "GEDEvent_v26_1.csv", gw2iso, types=types))
    if vintage_2024:
        lo, hi = midx(2024, 1), midx(2025, 1)
        D[:, lo:hi] = 0
        put(_events(O / "ucdp" / "candidate2024" / "GEDEvent_v24_01_24_12.csv", gw2iso, [2024], types=types), lo, hi)
    for y, (f, _) in sorted(candidate_files().items()):
        if y <= end_year:
            put(_events(f, gw2iso, [y], types=types))
    return D


def features(D, codes, p):
    ld = np.log1p(D)
    hit = (D >= 25).astype(float)
    M = D.shape[1]

    def roll(a, n, f=np.mean):
        out = np.full(a.shape, np.nan)
        for t in range(n - 1, a.shape[1]):
            out[:, t] = f(a[:, t - n + 1:t + 1], axis=1)
        return out
    F = {"ld": ld, "ld3": roll(ld, 3), "ld12": roll(ld, 12), "max12": roll(ld, 12, np.max), "hit24": roll(hit, 24)}
    since = np.full(D.shape, 240.0)
    run = np.full(len(codes), 240.0)
    for t in range(M):
        run = np.where(hit[:, t] > 0, 0, np.minimum(run + 1, 240))
        since[:, t] = run
    F["since"] = np.log1p(since)
    b = base_arrays(p)
    reg = np.array([p.regions[p.codes.index(c)] if c in p.codes else -1 for c in codes])
    last = int(p.years[np.where(np.isfinite(b["dem"]).any(0))[0].max()])
    for name, arr in [("lg", b["lg"]), ("lpop", b["lpop"]), ("dem", b["dem"]), ("conf", b["conflict"])]:
        a = np.full(D.shape, np.nan)
        for i, c in enumerate(codes):
            if c in p.codes:
                row = arr[p.codes.index(c)]
                for t in range(M):
                    y = min(M0 + t // 12 - 1, last)        # previous calendar year (latest available)
                    if p.years[0] <= y:
                        a[i, t] = row[p.yi(y)]
        F[name] = a
    rc = np.full(D.shape, np.nan)
    for r_ in set(reg):
        if r_ >= 0:
            rc[reg == r_] = np.nanmean(F["conf"][reg == r_], 0)
    F["reg_conf"] = rc
    if NEWS_TOPICS:
        F.update(news_topics(codes, M))
    return F, hit


NEWS_TOPICS = False   # conflictforecast.org news-topic stocks as features (tested 2026-11-08; scripts/monthly_topics_test.py)


def news_topics(codes, M):
    """{name: [C, M]} news-topic stocks (15 topics' shares, discounted over past months) and log token stock per
    country-month from conflictforecast.org (Mueller and Rauh 2022; free for non-commercial use with citation),
    January 2010 onward; NaN before and where the country is not covered"""
    f = RAW / "conflictforecast" / "conflictforecast_ons_armedconf_12.csv"
    cols = [f"stock_topic_{k}" for k in range(15)] + ["stock_tokens"]
    d = pd.read_csv(f, usecols=["isocode", "period"] + cols)
    d["m"] = (d.period // 100 - M0) * 12 + d.period % 100 - 1
    ci = {c: i for i, c in enumerate(codes)}
    out = {}
    for c in cols:
        a = np.full((len(codes), M), np.nan)
        for iso, m, v in zip(d.isocode, d.m, d[c]):
            if iso in ci and 0 <= m < M:
                a[ci[iso], m] = np.log1p(v) if c == "stock_tokens" else v
        out["nt_" + c.replace("stock_", "")] = a
    return out


def fit(F, hit, last_obs):
    """train on every (t, h) with target month t + h <= last_obs"""
    names = list(F)
    Xs, ys = [], []
    for h in range(1, 13):
        for t in range(12, last_obs - h + 1):
            Xs.append(np.stack([F[f][:, t] for f in names] + [np.full(hit.shape[0], h)], 1))
            ys.append(hit[:, t + h])
    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, verbose=-1)
    model.fit(np.concatenate(Xs), np.concatenate(ys))
    model.names_ = names
    return model


def predict(model, F, t, horizons=range(1, 13)):
    """{h: [C]} probabilities for months t + h, issued at month t"""
    n = next(iter(F.values())).shape[0]
    return {h: model.predict_proba(np.stack([F[f][:, t] for f in model.names_] + [np.full(n, h)], 1))[:, 1]
            for h in horizons}


def annual_probability(pred):
    """P(at least one 25-death month in the next 12 months), treating months as independent given the forecast.
    An upper bound: a country's months are positively correlated, so this overstates the 12-month risk (Benin 0.76 in
    the October 2026 ledger). Use fit_annual / predict_annual (debate 2026-11-06; scripts/monthly_annual_test.py)."""
    return 1 - np.prod([1 - pred[h] for h in sorted(pred)], 0)


# leaf size of the direct 12-month model, in country-months. Each country contributes 12 overlapping rows per year
# (their 12-month targets share 11 months), so 200 rows is about 17 country-years (third debate, 9 November 2026);
# 2400 = 200 country-years. scripts/monthly_cluster_test.py decides.
ANNUAL_MIN_CHILD = 2400        # (version 55, 9 November 2026: scripts/monthly_cluster_test.py)


def fit_annual(F, hit, last_obs, min_child=None):
    """direct 12-month model: P(any 25-death month in t+1..t+12), trained on every t whose 12 months are observed"""
    names = list(F)
    Xs, ys = [], []
    for t in range(12, last_obs - 12 + 1):
        Xs.append(np.stack([F[f][:, t] for f in names], 1))
        ys.append(hit[:, t + 1:t + 13].max(1))
    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=min_child or ANNUAL_MIN_CHILD,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, verbose=-1)
    model.fit(np.concatenate(Xs), np.concatenate(ys))
    model.names_ = names
    return model


def predict_annual(model, F, t):
    """[C] P(any 25-death month in the 12 months after month t)"""
    return model.predict_proba(np.stack([F[f][:, t] for f in model.names_], 1))[:, 1]
