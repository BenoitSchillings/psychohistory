"""Leaders leaving office: a simple yearly hazard model against the crowd (Good Judgment Open, Metaculus, INFER).

The world model does not forecast leader exits, but V-Dem records every turnover (year-end leader names, 1789-2025)
and the crowd set has ~80 resolved "will X cease to be leader before <date>" questions with forecast histories.
Model: LightGBM on country-years, target = the leader at the end of year t is gone by the end of year t+1, features =
the leader's tenure, age, military background, how they came to power (REIGN), election timing, regime, income,
conflict and coups. For each question it is fitted only on years whose outcome was known before the question opened
(t + 1 <= open year - 1) and evaluated on the year before the question opened; the yearly hazard h becomes
1 - (1 - h) ^ (days / 365) over the question's window (to the deadline stated in the question).
Output: outputs/leader_benchmark.json
"""
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
import lightgbm as lgb
import numpy as np
import pandas as pd

from crowd_benchmark import stated_deadline
from psychohistory.data import get_panel
from psychohistory.features import base_arrays

FEATS = ["leader_log_tenure", "leader_age", "leader_military", "leader_elected", "election_anticipated",
         "elections_delayed", "months_since_irregular", "months_since_loss", "election_year", "years_since_election",
         "dem", "lg", "conflict", "coup_recent"]
B = os.path.join(ROOT, "data", "raw", "benchmarks")


def main():
    p = get_panel()
    b = base_arrays(p)
    yrs = np.asarray(p.years)
    lc = b["leader_change"]
    C, T = lc.shape
    F = np.stack([b[k] if k in b else np.full((C, T), np.nan) for k in FEATS], -1)        # [C, T, F]
    Yn = np.full((C, T), np.nan)
    Yn[:, :-1] = lc[:, 1:]                                                                 # change in year t+1
    q = pd.read_csv(os.path.join(B, "crowd_questions.csv"), parse_dates=["open_date", "close_date"])
    q = q[(q.category == "leader_exit") & (q.subtype == "leader_exit") & (q.polarity == 1) & q.resolution.notna()
          & q.iso3_a.isin(p.codes)]
    models, rows, skipped = {}, [], 0
    for _, r in q.iterrows():
        y0 = int(r.open_date.year) - 1
        if y0 not in set(yrs) or y0 < 1990:
            continue
        if y0 not in models:
            tr = (yrs[None, :] + 1 <= y0) & (yrs[None, :] >= 1950) & np.isfinite(Yn)
            X, y = F[tr], Yn[tr]
            models[y0] = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=50,
                                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(X, y)
        h = float(models[y0].predict_proba(F[p.codes.index(r.iso3_a), p.yi(y0)][None])[0, 1])
        end = stated_deadline(r.title)
        if end is None:
            if r.resolution == 1:
                skipped += 1
                continue
            end = r.close_date
        end = max(end, r.close_date)
        for snap, col in (("open", "p_open_7d"), ("mid", "p_midpoint")):
            if np.isfinite(r[col]):
                s_ = r.open_date + pd.Timedelta(days=7) if snap == "open" else r.open_date + (r.close_date - r.open_date) / 2
                days = max((end - s_).days, 1)
                rows.append({"snap": snap, "source": r.source, "qid": r.qid, "title": r.title, "country": r.iso3_a,
                             "y": float(r.resolution), "crowd": float(r[col]), "model": 1 - (1 - h) ** (days / 365.25),
                             "hazard_1y": h, "days": days, "year": int(r.open_date.year)})
    R = pd.DataFrame(rows)
    out = {"n_questions": int(q.shape[0]), "n_skipped_no_deadline": skipped, "note": __doc__.strip().split("\n\n")[0]}
    rng = np.random.default_rng(0)
    for snap, g in R.groupby("snap"):
        y, c, m = g.y.to_numpy(), g.crowd.to_numpy(), g.model.to_numpy()
        br = lambda f: float(np.mean((f - y) ** 2))
        ctry = g.country.to_numpy()
        uc = np.unique(ctry)
        diffs = []
        for _ in range(2000):                     # country-clustered bootstrap of the Brier difference
            pick = rng.choice(uc, len(uc))
            ix = np.concatenate([np.where(ctry == u)[0] for u in pick])
            diffs.append(np.mean((m[ix] - y[ix]) ** 2) - np.mean((c[ix] - y[ix]) ** 2))
        out[snap] = {"n": int(len(g)), "countries": int(len(uc)), "events": int(y.sum()), "base_rate": float(y.mean()),
                     "brier_crowd": br(c), "brier_model": br(m), "brier_avg": br((c + m) / 2),
                     "brier_base_rate": br(np.full(len(y), y.mean())),
                     "brier_diff_model_minus_crowd_90ci": np.round(np.quantile(diffs, [0.05, 0.95]), 4).tolist(),
                     "mean_crowd": float(c.mean()), "mean_model": float(m.mean())}
    out["questions"] = R.round(4).to_dict("records")
    json.dump(out, open(os.path.join(ROOT, "outputs", "leader_benchmark.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "questions"}, indent=1))


if __name__ == "__main__":
    main()
