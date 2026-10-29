"""The onset side of the 2010s interstate gap (docs/world_system.md, round f): new pairs such as Iran-Israel and the
US/UK strikes on Yemen. Do pair features the simulator can update each year improve the 1-year onset hazard?
The simulator evaluates only the logit part of the dyad model (plus a fading per-pair anchor), so the candidates are
scored with that logit alone (interactions learned by the GBM would vanish in simulation).
Candidates (year t, predicting a new UCDP conflict in t+1 for pairs at peace):
  civ_major    one side in civil conflict and the other a major power (outside intervention: USA/GBR-Yemen)
  busy_max     log1p of the other interstate conflicts either side is fighting this year (max over the two sides)
  bellig10     log1p of either side's interstate conflict-years with anyone in the past 10 years (max over sides)
  sys_active   log1p of active interstate conflicts in the world this year (contagion)
Origins: dev 1970-1998 (step 4), test 2000-2014 (step 2); trained on outcomes known at the origin, scored on onsets in
origin..origin+9 (windows overlap, so pooled pair-years repeat). Log-loss and AUC, pooled per split.
python -P scripts/interstate_onset_test.py -> outputs/interstate_onset_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

from psychohistory import interstate as I
from psychohistory.data import get_panel
from psychohistory.features import base_arrays

ROOT = os.path.join(os.path.dirname(__file__), "..")
CANDS = ["civ_major", "busy_max", "bellig10", "sys_active"]


def add_candidates(df):
    p = get_panel()
    base = base_arrays(p)
    codes = list(p.codes)
    ci = {c: i for i, c in enumerate(codes)}
    y0 = int(p.years[0])
    conf = np.nan_to_num(base["conflict"]) > 0
    majors = I._majors(I._ccode_to_iso(set(codes)))
    is_major = lambda c, y: any(s <= y <= e for s, e in majors.get(c, []))
    t = (df.year - y0).to_numpy()
    ok = (t >= 0) & (t < conf.shape[1])
    ia = df.a.map(ci).to_numpy()
    ib = df.b.map(ci).to_numpy()
    ca = np.where(ok, conf[ia, np.clip(t, 0, conf.shape[1] - 1)], False)
    cb = np.where(ok, conf[ib, np.clip(t, 0, conf.shape[1] - 1)], False)
    ma = np.array([is_major(a, y) for a, y in zip(df.a, df.year)])
    mb = np.array([is_major(b, y) for b, y in zip(df.b, df.year)])
    df["civ_major"] = ((ca & mb & ~ma) | (cb & ma & ~mb)).astype(float)
    # conflicts per country-year (active pairs), and the world total
    act = df[df.ongoing == 1][["a", "b", "year"]]
    cy = pd.concat([act.rename(columns={"a": "c"})[["c", "year"]], act.rename(columns={"b": "c"})[["c", "year"]]])
    n_cy = cy.groupby(["c", "year"]).size()
    get = lambda c, y: n_cy.get((c, y), 0)
    na = np.array([get(a, y) for a, y in zip(df.a, df.year)], float)
    nb = np.array([get(b, y) for b, y in zip(df.b, df.year)], float)
    own = df.ongoing.fillna(0).to_numpy()
    df["busy_max"] = np.log1p(np.maximum(na - own, nb - own))
    roll = n_cy.unstack(fill_value=0).reindex(columns=range(1946, int(df.year.max()) + 1), fill_value=0)
    past10 = roll.T.rolling(10, min_periods=1).sum().shift(1).fillna(0).T          # years t-10..t-1
    g10 = lambda c, y: float(past10.loc[c, y]) if c in past10.index and y in past10.columns else 0.0
    df["bellig10"] = np.log1p(np.maximum([g10(a, y) for a, y in zip(df.a, df.year)], [g10(b, y) for b, y in zip(df.b, df.year)]))
    sys_n = act.groupby("year").size()
    df["sys_active"] = np.log1p(df.year.map(sys_n).fillna(0).to_numpy())
    return df


def fit_logit(X, y):
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
    z = lambda A: np.nan_to_num(np.clip((A - mu) / sd, -6, 6))
    m = LogisticRegression(C=0.3, max_iter=4000).fit(z(X), y)
    return lambda A: m.predict_proba(z(A))[:, 1], dict(zip(range(X.shape[1]), m.coef_[0]))


def main():
    df = add_candidates(I.build(False))
    at_peace = df.interstate_1y.notna() & (df.ongoing == 0)
    variants = {"base": I.FEATS_ON, **{c: I.FEATS_ON + [c] for c in CANDS}, "all": I.FEATS_ON + CANDS}
    splits = {"dev": range(1970, 1999, 4), "test": range(2000, 2015, 2)}
    res = {}
    for split, origins in splits.items():
        P = {v: [] for v in variants}
        Y, coefs = [], {v: [] for v in variants}
        for O in origins:
            tr = at_peace & (df.year + 1 <= O) & (df.year >= 1946)
            te = at_peace & (df.year >= O) & (df.year < O + 10)
            Y.append(df.loc[te, "interstate_1y"].to_numpy())
            for v, f in variants.items():
                pred, co = fit_logit(df.loc[tr, f].to_numpy(float), df.loc[tr, "interstate_1y"].to_numpy())
                P[v].append(pred(df.loc[te, f].to_numpy(float)))
                coefs[v].append([float(co[i]) for i in range(len(I.FEATS_ON), len(f))])
        y = np.concatenate(Y)
        out = {}
        for v in variants:
            pr = np.concatenate(P[v])
            out[v] = {"logloss": round(float(log_loss(y, np.clip(pr, 1e-6, 1 - 1e-6))), 5),
                      "auc": round(float(roc_auc_score(y, pr)), 4), "pred_sum": round(float(pr.sum()), 1),
                      "events": int(y.sum()), "n": len(y), "coef_mean": np.round(np.mean(coefs[v], 0), 3).tolist()}
        res[split] = out
        print(split, json.dumps(out, indent=1), flush=True)
    json.dump(res, open(os.path.join(ROOT, "outputs", "interstate_onset_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
