"""Monthly conflict forecasts, head to head with ViEWS on 2025.

Target (ViEWS's own): at least 25 state-based battle deaths located in the country in a month (UCDP GED 26.1).
Features at month t: the country's fighting history (log deaths this month, 3- and 12-month means, the 12-month
maximum, months since the last 25-death month, share of such months in the past two years) plus structural
variables from the previous calendar year (income, population, democracy, regional conflict) and the horizon.
Training: months 1990-01 .. 2024-12 with targets up to 2024-12 only. Forecasts for 2025-01 .. 2025-12 are all
issued from 2024-12, like ViEWS run fatalities002_2024_12_t01. Writes outputs/monthly_2025.json.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from psychohistory.data import get_panel
from psychohistory.features import base_arrays

O = os.path.join(ROOT, "data", "raw", "outcomes")


def main():
    vw = pd.read_csv(os.path.join(O, "views_2025_cm.csv"))
    gw2iso = dict(zip(vw.gwcode, vw.iso3))
    ged = pd.read_csv(os.path.join(O, "ucdp", "ged261-csv", "GEDEvent_v26_1.csv"),
                      usecols=["type_of_violence", "country_id", "date_start", "best"])
    ged = ged[ged.type_of_violence == 1]
    ged["iso3"] = ged.country_id.map(gw2iso)
    ged = ged.dropna(subset=["iso3"])
    d = pd.to_datetime(ged.date_start)
    ged["m"] = (d.dt.year - 1989) * 12 + d.dt.month - 1
    codes = sorted(vw.iso3.unique())
    M = (2025 - 1989 + 1) * 12
    D = np.zeros((len(codes), M))
    ci = {c: i for i, c in enumerate(codes)}
    g = ged.groupby(["iso3", "m"]).best.sum()
    for (c, m), v in g.items():
        if c in ci and m < M:
            D[ci[c], m] = v
    hit_final = (D >= 25).astype(float)
    if "--vintage" in sys.argv:
        # what was known in January 2025: preliminary (candidate) events for 2024, released 2025-01-24
        cand = pd.read_csv(os.path.join(O, "ucdp", "candidate2024", "GEDEvent_v24_01_24_12.csv"),
                           usecols=["type_of_violence", "country_id", "date_start", "best"])
        cand = cand[cand.type_of_violence == 1]
        cand["iso3"] = cand.country_id.map(gw2iso)
        dc = pd.to_datetime(cand.date_start)
        cand = cand[dc.dt.year == 2024]
        cand["m"] = (dc[dc.dt.year == 2024].dt.year - 1989) * 12 + dc[dc.dt.year == 2024].dt.month - 1
        m24 = slice((2024 - 1989) * 12, (2025 - 1989) * 12)
        D[:, m24] = 0
        for (c, m), v in cand.groupby(["iso3", "m"]).best.sum().items():
            if c in ci:
                D[ci[c], m] = v
        vintage = "candidate 2024 (released 2025-01-24)"
    else:
        vintage = "final GED 26.1"
    ld = np.log1p(D)
    hit = (D >= 25).astype(float)

    def roll(a, n, f=np.mean):
        out = np.full(a.shape, np.nan)
        for t in range(n - 1, a.shape[1]):
            out[:, t] = f(a[:, t - n + 1:t + 1], axis=1)
        return out
    feats = {"ld": ld, "ld3": roll(ld, 3), "ld12": roll(ld, 12), "max12": roll(ld, 12, np.max), "hit24": roll(hit, 24)}
    since = np.full(D.shape, 240.0)
    run = np.full(len(codes), 240.0)
    for t in range(M):
        run = np.where(hit[:, t] > 0, 0, np.minimum(run + 1, 240))
        since[:, t] = run
    feats["since"] = np.log1p(since)
    # structural, previous calendar year
    p = get_panel()
    b = base_arrays(p)
    reg = np.array([p.regions[p.codes.index(c)] if c in p.codes else -1 for c in codes])
    for name, arr in [("lg", b["lg"]), ("lpop", b["lpop"]), ("dem", b["dem"]), ("conf", b["conflict"])]:
        a = np.full(D.shape, np.nan)
        for c, i in ci.items():
            if c in p.codes:
                for t in range(M):
                    y = 1989 + t // 12 - 1
                    if p.years[0] <= y <= 2024:
                        a[i, t] = arr[p.codes.index(c), p.yi(y)]
        feats[name] = a
    rc = np.full(D.shape, np.nan)
    for r_ in set(reg):
        if r_ < 0:
            continue
        m_ = reg == r_
        rc[m_] = np.nanmean(feats["conf"][m_], 0)
    feats["reg_conf"] = rc
    F = list(feats)
    last_train = (2024 - 1989) * 12 + 11            # 2024-12
    rows_X, rows_y, rows_w = [], [], []
    for h in range(1, 13):
        for t in range(12, last_train - h + 1):
            X = np.stack([feats[f][:, t] for f in F] + [np.full(len(codes), h)], 1)
            rows_X.append(X); rows_y.append(hit[:, t + h])
    X = np.concatenate(rows_X); y = np.concatenate(rows_y)
    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, verbose=-1)
    model.fit(X, y)
    pred = {}
    for h in range(1, 13):
        Xh = np.stack([feats[f][:, last_train] for f in F] + [np.full(len(codes), h)], 1)
        pred[h] = model.predict_proba(Xh)[:, 1]
    # score against ViEWS on the same country-months
    rows = []
    for _, r in vw.iterrows():
        i, h = ci[r.iso3], int(r.month)
        rows.append({"iso3": r.iso3, "month": h, "y": hit_final[i, last_train + h], "seldon": pred[h][i], "views": r.prob_ge25,
                     "active_2024": hit[i, last_train - 11:last_train + 1].max()})
    S = pd.DataFrame(rows)
    S["average"] = (S.seldon + S.views) / 2
    base = S.y.mean()

    def sc(df):
        out = {"n": len(df), "events": int(df.y.sum())}
        for k in ("seldon", "views", "average"):
            out[k] = {"brier": float(np.mean((df[k] - df.y) ** 2)),
                      "auc": float(roc_auc_score(df.y, df[k])) if 0 < df.y.sum() < len(df) else None,
                      "logloss": float(-np.mean(df.y * np.log(df[k].clip(1e-4, 1 - 1e-4)) + (1 - df.y) * np.log(1 - df[k].clip(1e-4, 1 - 1e-4))))}
        out["brier_base"] = float(np.mean((base - df.y) ** 2))
        return out
    res = {"vintage": vintage, "all": sc(S), "new_cases": sc(S[S.active_2024 == 0]), "by_horizon": {}}
    for hz in (1, 3, 6, 12):
        res["by_horizon"][hz] = sc(S[S.month == hz])
    name = "monthly_2025_vintage.json" if "--vintage" in sys.argv else "monthly_2025.json"
    json.dump(res, open(os.path.join(ROOT, "outputs", name), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("all", "new_cases")}, indent=1))
    print({hz: {k: round(v[k]["brier"], 4) for k in ("seldon", "views", "average")} for hz, v in res["by_horizon"].items()})


if __name__ == "__main__":
    main()
