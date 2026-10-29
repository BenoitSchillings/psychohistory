"""Do Data Commons state series (scripts/dc_states_fetch.py: FBI crime 2008-2023, ACS education, poverty, age,
foreign-born and uninsured shares ~2010-2024) improve the US state module's five-year crisis forecasts?

New features (carried forward up to 2 years: FBI 2021 is missing, ACS ends 2024):
  log_violent_crime, log_property_crime, log_murder, violent_crime_change_3y, bachelors_share, poverty_rate,
  poverty_change_3y, median_age, foreign_born_share, uninsured_share
Two checks, both with the module's own model (logit + GBM average), base features vs base + new:
1. time split: origins 2016-2020 (each trained on outcomes known by its origin; the new series start in 2008-2012,
   so only these late origins can learn from them), test years origin..min(origin + 4, 2020);
2. state-grouped 5-fold cross-validation over label years 2010-2020 (whole states held out; common yearly shocks
   are shared between folds, so this measures cross-state signal, not forecasting skill).
Scores: AUC and Brier skill per task, pooled; differences are noise unless they hold in both checks.
python -P scripts/dc_states_test.py -> outputs/dc_states_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from psychohistory import us_states as U

ROOT = os.path.join(os.path.dirname(__file__), "..")
NEW = ["log_violent_crime", "log_property_crime", "log_murder", "violent_crime_change_3y", "bachelors_share",
       "poverty_rate", "poverty_change_3y", "median_age", "foreign_born_share", "uninsured_share"]


def dc_features():
    d = pd.read_csv(os.path.join(U.US, "datacommons_states.csv"))
    piv = lambda c: U._ffill(U._put(U._blank(), d.rename(columns={c: "v"}), "v"), 2)
    vc, pc, mu, pov = piv("violent_crime"), piv("property_crime"), piv("murder"), piv("poverty_rate")
    f = {"log_violent_crime": np.log1p(vc), "log_property_crime": np.log1p(pc), "log_murder": np.log1p(mu),
         "violent_crime_change_3y": np.log1p(vc) - U._lag(np.log1p(vc), 3), "bachelors_share": piv("bachelors_share"),
         "poverty_rate": pov, "poverty_change_3y": pov - U._lag(pov, 3), "median_age": piv("median_age"),
         "foreign_born_share": piv("foreign_born_share"), "uninsured_share": piv("uninsured_share")}
    return np.stack([f[k] for k in NEW], -1)


def score(rows):
    out = {}
    for task, g in rows.groupby("task"):
        g = g.dropna()
        if g.y.nunique() < 2:
            continue
        b = ((g.base - g.y) ** 2).mean()
        out[task] = {"n": len(g), "events": int(g.y.sum()), "auc": round(roc_auc_score(g.y, g.p), 4),
                     "bss": round(1 - ((g.p - g.y) ** 2).mean() / b, 4)}
    return out


def time_split(X, Y):
    Xf, Yf, si, year = U._flat(X, Y)
    rows = []
    for T in range(2016, 2021):
        Ytr, tr = U._mask(Yf, year, T)
        m = U.StateModel().fit(Xf[tr], Ytr[tr])
        te = (year >= T) & (year <= min(T + 4, 2020))
        lr, gb = m.predict_parts(Xf[te])
        base = np.nanmean(Ytr[tr], 0)
        for k, task in enumerate(U.TASKS):
            ok = np.isfinite(Yf[te, k])
            rows.append(pd.DataFrame({"task": task, "p": ((lr + gb) / 2)[ok, k], "y": Yf[te, k][ok], "base": base[k]}))
    return score(pd.concat(rows))


def state_cv(X, Y):
    Xf, Yf, si, year = U._flat(X, Y)
    sel = (year >= 2010) & (year <= 2020)
    rng = np.random.default_rng(0)
    fold = rng.permutation(len(U.CODES)) % 5
    rows = []
    for k_ in range(5):
        test_states = np.flatnonzero(fold == k_)
        te = sel & np.isin(si, test_states)
        tr = (year >= 1960) & ~np.isin(si, test_states) & np.isfinite(Yf).any(1) & (year <= 2020)
        m = U.StateModel().fit(Xf[tr], Yf[tr])
        lr, gb = m.predict_parts(Xf[te])
        base = np.nanmean(Yf[tr], 0)
        for k, task in enumerate(U.TASKS):
            ok = np.isfinite(Yf[te, k])
            rows.append(pd.DataFrame({"task": task, "p": ((lr + gb) / 2)[ok, k], "y": Yf[te, k][ok], "base": base[k]}))
    return score(pd.concat(rows))


def main():
    x = U.load()
    X, Y = U.features(x), U.labels(x)
    Xn = np.concatenate([X, dc_features()], -1)
    feats0 = list(U.FEATURES)
    res = {}
    try:
        for name, XX in (("base", X), ("datacommons", Xn)):
            if name == "datacommons":
                U.FEATURES = feats0 + NEW
            res[name] = {"time_split_2016_2020": time_split(XX, Y), "state_cv_2010_2020": state_cv(XX, Y)}
            print(name, json.dumps(res[name], indent=1), flush=True)
    finally:
        U.FEATURES = feats0
    json.dump(res, open(os.path.join(ROOT, "outputs", "dc_states_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
