"""Rolling-origin backtest: stand in year T, train only on outcomes observed by T,
forecast country-years T..T+4, then score against what actually happened."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .features import TASKS
from .models import NaiveModel, SeldonEnsemble, load_combine

DISPLAY = {"G": "GBM", "G60": "GBM (since 1860)", "N": "PsychoNet", "Nq": "PsychoNet (quantile)", "F": "Forest",
           "L": "Logit"}
ALL_FAMILIES = list(DISPLAY)

HORIZON = {"civil_conflict_5y": 5, "econ_collapse_5y": 5, "autocratization_5y": 5, "coup_5y": 5,
           "onset_1y": 1, "termination_1y": 1, "coup_1y": 1}
FIRST_TRAIN_YEAR = 1955


def flatten(p, X, Yl):
    C, Yn, _ = X.shape
    ci, ti = np.meshgrid(np.arange(C), np.arange(Yn), indexing="ij")
    year = p.years[ti].ravel()
    return X.reshape(C * Yn, -1), Yl.reshape(C * Yn, -1), ci.ravel(), year


def train_mask(Yf, year, origin, first_year=None):
    """hide labels that would not have been observed by `origin`."""
    Y = Yf.copy()
    for k, t in enumerate(TASKS):
        Y[year + HORIZON[t] > origin, k] = np.nan
    rows = (year >= (FIRST_TRAIN_YEAR if first_year is None else first_year)) & np.isfinite(Y).any(1)
    return Y, rows


def score(y, p, base, groups=None):
    """`groups` (e.g. backtest origin): the top-decile threshold is set within each group, as a
    forecaster at that origin could have done."""
    ok = np.isfinite(y) & np.isfinite(p)
    y, p, base = y[ok], p[ok], np.broadcast_to(base, ok.shape)[ok]
    groups = np.zeros(len(y)) if groups is None else np.asarray(groups)[ok]
    if len(y) == 0 or y.min() == y.max():
        return {}
    brier = np.mean((p - y) ** 2)
    brier_clim = np.mean((base - y) ** 2)
    top = np.zeros(len(p), bool)
    for g in np.unique(groups):
        i = groups == g
        top[i] = p[i] >= np.quantile(p[i], 0.9)
    return {"n": len(y), "events": int(y.sum()), "auc": roc_auc_score(y, p), "brier": brier,
            "bss": 1 - brier / brier_clim,
            "top10_capture": y[top].sum() / y.sum(),   # share of events flagged in riskiest decile
            "top10_precision": y[top].mean()}


def run_backtest(p, X, Yl, origins=range(1980, 2016, 5), members=8, verbose=True):
    Xf, Yf, ci, year = flatten(p, X, Yl)
    rows_out = []
    for T in origins:
        Ytr, tr = train_mask(Yf, year, T, first_year=1860)
        ens = SeldonEnsemble(load_combine(), members=members, families=ALL_FAMILIES).fit(Xf[tr], Ytr[tr], ci[tr], year[tr])
        r55 = tr & (year >= FIRST_TRAIN_YEAR)
        naive = NaiveModel().fit(Xf[r55], Ytr[r55])
        te = (year >= T) & (year < T + 5)
        fam = ens.predict_families(Xf[te])
        preds = {DISPLAY[f]: v for f, v in fam.items()}
        preds["Seldon ensemble"] = ens.predict(Xf[te])
        preds[naive.name] = naive.predict(Xf[te])
        base = np.nanmean(Ytr[r55], 0)
        for k, task in enumerate(TASKS):
            y = Yf[te, k]
            ok = np.isfinite(y)
            for mname, pr in preds.items():
                rows_out.append(pd.DataFrame({
                    "origin": T, "task": task, "model": mname, "code": np.array(p.codes)[ci[te][ok]],
                    "country": np.array(p.names)[ci[te][ok]], "year": year[te][ok],
                    "p": pr[ok, k], "y": y[ok], "base": base[k]}))
        if verbose:
            print(f"  origin {T}: trained on {tr.sum()} rows, tested {te.sum()}")
    return pd.concat(rows_out, ignore_index=True)


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (task, model), g in bt.groupby(["task", "model"]):
        s = score(g.y.to_numpy(), g.p.to_numpy(), g.base.to_numpy(), g.origin.to_numpy())
        if s:
            out.append({"task": task, "model": model, **s})
    return pd.DataFrame(out)


def summarize_split(bt: pd.DataFrame, dev_last=2005, held_first=2015, models=("Seldon ensemble", "Naive (Fearon-Laitin)"),
                    n_boot=500, seed=0) -> pd.DataFrame:
    """the backtest split into development origins (used to choose features, hyper-parameters and ensemble
    weights: 1975-2005, whose 5-year outcomes reach 2014) and the origins whose outcomes were never seen in tuning
    (2015 on; the 2010 origin's outcomes overlap 2011-2014), with 90% intervals from a country-clustered bootstrap
    (the 5 overlapping yearly rows per origin and country are not independent)"""
    rng = np.random.default_rng(seed)
    out = []
    for split, sel in (("development", bt.origin <= dev_last), ("held out", bt.origin >= held_first)):
        for (task, model), g in bt[sel].groupby(["task", "model"]):
            if model not in models:
                continue
            s = score(g.y.to_numpy(), g.p.to_numpy(), g.base.to_numpy(), g.origin.to_numpy())
            if not s:
                continue
            codes = g.code.to_numpy()
            uc = np.unique(codes)
            idx = {c: np.where(codes == c)[0] for c in uc}
            aucs, bsss = [], []
            for _ in range(n_boot):
                ix = np.concatenate([idx[c] for c in rng.choice(uc, len(uc))])
                b = score(g.y.to_numpy()[ix], g.p.to_numpy()[ix], g.base.to_numpy()[ix], g.origin.to_numpy()[ix])
                if b:
                    aucs.append(b["auc"])
                    bsss.append(b["bss"])
            out.append({"split": split, "task": task, "model": model, "n": s["n"], "events": s["events"],
                        "countries": len(uc), "auc": s["auc"], "auc_lo": float(np.quantile(aucs, .05)),
                        "auc_hi": float(np.quantile(aucs, .95)), "bss": s["bss"], "bss_lo": float(np.quantile(bsss, .05)),
                        "bss_hi": float(np.quantile(bsss, .95))})
    return pd.DataFrame(out)


def calibration(bt: pd.DataFrame, task: str, model="Seldon ensemble", bins=(0, .02, .05, .1, .2, .35, .5, 1)):
    g = bt[(bt.task == task) & (bt.model == model)].copy()
    g["bin"] = pd.cut(g.p, bins, include_lowest=True)
    return g.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")).reset_index()
