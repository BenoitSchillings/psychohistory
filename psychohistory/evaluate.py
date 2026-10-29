"""Experiment harness for model improvement.

Development origins (1975-2005) are used for every comparison; the lockbox origins (2010, 2015)
are only scored at milestones so that hours of tuning cannot quietly overfit them.
Score = mean over the four headline crises of (AUC + Brier skill score).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .backtest import flatten, train_mask
from .data import ROOT
from .features import TASKS

DEV_ORIGINS = tuple(range(1975, 2006, 5))
LOCK_ORIGINS = (2010, 2015)
HEADLINE = ["civil_conflict_5y", "econ_collapse_5y", "autocratization_5y", "coup_5y"]
LOG = ROOT / "outputs" / "experiments.csv"


def metrics(y, p, base):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p, base = y[ok], p[ok], base[ok]
    if len(y) < 10 or y.min() == y.max():
        return {}
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    return {"auc": roc_auc_score(y, p), "bss": 1 - np.mean((p - y) ** 2) / np.mean((base - y) ** 2),
            "logloss": -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)), "n": int(len(y)), "events": int(y.sum())}


def evaluate(p, X, Yl, factory, origins=DEV_ORIGINS, extra_train=None, verbose=False, first_year=None):
    """factory() -> model with fit(X, Y, groups) and predict(X) -> [N, len(TASKS)].
    extra_train: optional (X, Y, groups, years) of additional (e.g. pre-1946) training rows."""
    Xf, Yf, ci, year = flatten(p, X, Yl)
    preds = []
    for T in origins:
        Ytr, tr = train_mask(Yf, year, T, first_year)
        Xt, Yt, gt = Xf[tr], Ytr[tr], ci[tr]
        if extra_train is not None:
            ex, ey, eg, eyr = extra_train
            ey_m, etr = train_mask(ey, eyr, T)
            Xt, Yt, gt = np.concatenate([Xt, ex[etr]]), np.concatenate([Yt, ey_m[etr]]), np.concatenate([gt, eg[etr]])
        t0 = time.time()
        m = factory()
        yrs_t = np.concatenate([year[tr]] + ([extra_train[3][etr]] if extra_train is not None else []))
        m = m.fit(Xt, Yt, gt, years=yrs_t) if getattr(m, "uses_years", False) else m.fit(Xt, Yt, gt)
        te = (year >= T) & (year < T + 5)
        pr = m.predict(Xf[te])
        # climatology reference for the Brier skill score: post-1955 training rows only, so runs that
        # train on longer history are scored against the same baseline
        r55 = tr & (year >= 1955)
        base = np.broadcast_to(np.nanmean(Ytr[r55], 0), pr.shape)
        preds.append((T, Yf[te], pr, base, ci[te]))
        if verbose:
            print(f"    origin {T}: {time.time() - t0:.1f}s", flush=True)
    out = {}
    for k, t in enumerate(TASKS):
        y = np.concatenate([a[1][:, k] for a in preds])
        pr = np.concatenate([a[2][:, k] for a in preds])
        b = np.concatenate([a[3][:, k] for a in preds])
        out[t] = metrics(y, pr, b)
    score = float(np.mean([out[t]["auc"] + out[t]["bss"] for t in HEADLINE]))
    return score, out, preds


def log(name, score, out, notes="", split="dev"):
    row = {"time": time.strftime("%Y-%m-%d %H:%M"), "name": name, "split": split, "score": round(score, 4)}
    for t in TASKS:
        if out.get(t):
            row[f"{t}_auc"] = round(out[t]["auc"], 4)
            row[f"{t}_bss"] = round(out[t]["bss"], 4)
    row["notes"] = notes
    df = pd.DataFrame([row])
    df.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    hl = " ".join(f"{t.split('_')[0][:6]} {out[t]['auc']:.3f}/{out[t]['bss']:+.3f}" for t in HEADLINE)
    print(f"[{split}] {name:40s} score {score:.4f} | {hl}", flush=True)
    return row


class RecentCalibrated:
    """Wraps a model: fits it on all but the most recent `window` years of training rows, learns a
    per-task Platt map (logit p -> outcome) on those recent rows, then refits on everything and
    applies the map. Corrects base-rate drift between eras using only information available at
    the forecast origin."""
    uses_years = True

    def __init__(self, factory, window=8, min_events=15):
        self.factory, self.window, self.min_events = factory, window, min_events

    def fit(self, X, Y, g, years):
        from sklearn.linear_model import LogisticRegression
        cut = np.nanmax(years[np.isfinite(Y).any(1)]) - self.window
        old = years < cut
        m0 = self.factory().fit(X[old], Y[old], g[old])
        pr = m0.predict(X[~old])
        self.maps = {}
        for k in range(Y.shape[1]):
            y = Y[~old, k]
            ok = np.isfinite(y) & np.isfinite(pr[:, k])
            if ok.sum() > 50 and y[ok].sum() >= self.min_events and y[ok].sum() < ok.sum():
                z = _lg(pr[ok, k])[:, None]
                self.maps[k] = LogisticRegression(C=1.0).fit(z, y[ok])
        self.m = self.factory().fit(X, Y, g)
        return self

    def predict(self, X):
        p = self.m.predict(X)
        for k, mp in self.maps.items():
            p[:, k] = mp.predict_proba(_lg(p[:, k])[:, None])[:, 1]
        return p


def _lg(p):
    p = np.clip(np.nan_to_num(p, nan=0.5), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))
