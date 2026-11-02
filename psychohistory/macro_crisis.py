"""Sovereign debt crises and currency crises (Global Macro Database dates, 2026-11-02; scripts/debt_crisis_test.py).

Hazard of a crisis starting, logit on last year's state: log GDP per person, public debt / GDP, its 5-year change,
growth and log inflation (rate data are too sparse before 2000 to use). Fit on 1960..origin-1, then the intercept is
shifted so the average predicted chance over the 20 years before the origin matches the observed onset rate there
(debt crises were rarer after 2000 than in the 1980s). Output costs, estimated with country and year effects removed and
stable across 1960-1999 and 2000-2024: sovereign -2.0% in the crisis year and -1.2% the next, currency -2.3% and -1.0%.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from .data import RAW

COST = {"sovereign": (0.020, 0.012), "currency": (0.023, 0.010)}
COLS = {"sovereign": "SovDebtCrisis", "currency": "CurrencyCrisis"}


@functools.lru_cache(maxsize=1)
def _crises(codes, years):
    g = pd.read_csv(RAW / "GMD.csv", usecols=["ISO3", "year", "SovDebtCrisis", "CurrencyCrisis"])
    ci, yi = {c: i for i, c in enumerate(codes)}, {y: j for j, y in enumerate(years)}
    out = {k: np.full((len(codes), len(years)), np.nan) for k in COLS}
    for r in g.itertuples():
        if r.ISO3 in ci and r.year in yi:
            for k, col in COLS.items():
                v = getattr(r, col)
                if pd.notna(v):
                    out[k][ci[r.ISO3], yi[r.year]] = float(v)
    return out


def _design(lg, debt, debt_chg5, growth, inflation):
    """state -> predictors (numpy or torch, broadcast)"""
    xp = np if isinstance(lg, np.ndarray) else __import__("torch")
    infl = xp.sign(inflation) * xp.log1p(xp.abs(xp.clip(inflation, -99, 1e4)))
    return [lg, debt / 100, debt_chg5 / 100, growth, infl]


@functools.lru_cache(maxsize=8)
def fit(origin, codes, years):
    """{kind: (coefficients [6] incl. intercept, crisis means for imputation)} fitted on data before the origin"""
    import statsmodels.api as sm
    from .data import get_panel
    from .pipeline import base_arrays
    p = get_panel(False)
    b = base_arrays(p)
    yrs = np.asarray(p.years)
    C, T = len(p.codes), len(yrs)
    lag = lambda a, k=1: np.concatenate([np.full((C, k), np.nan), a[:, :-k]], 1)
    lg = b["lg"]
    D = np.stack(_design(lag(lg), lag(b["public_debt"]), lag(b["debt_change_5y"]), lag(lg - lag(lg)), lag(b["inflation"])), -1)
    Y_ = np.broadcast_to(yrs[None], (C, T))
    cr = _crises(tuple(p.codes), tuple(yrs))
    out = {}
    for k, c in cr.items():
        onset = np.where(np.isfinite(c) & np.isfinite(lag(c)), ((c == 1) & (lag(c) == 0)).astype(float), np.nan)
        ok = np.isfinite(onset) & np.isfinite(D).all(-1) & (Y_ >= 1960) & (Y_ < origin)
        m = sm.Logit(onset[ok], sm.add_constant(D[ok])).fit(disp=0)
        beta = np.asarray(m.params, float)
        # align the level to the 20 years before the origin
        rec = ok & (Y_ >= origin - 20)
        if rec.sum() > 50:
            lin = sm.add_constant(D[rec], has_constant="add") @ beta
            target = onset[rec].mean()
            lo_, hi_ = -10.0, 10.0
            for _ in range(60):
                mid = (lo_ + hi_) / 2
                lo_, hi_ = (mid, hi_) if (1 / (1 + np.exp(-(lin + mid)))).mean() < target else (lo_, mid)
            beta[0] += (lo_ + hi_) / 2
        out[k] = (beta.astype(np.float32), np.nanmean(D[ok], 0).astype(np.float32))
    return out


def hazard(model, lg, debt, debt_chg5, growth, inflation):
    """yearly chance of a crisis starting, torch [S, C]"""
    import torch
    beta, mean_ = model
    X = _design(lg, debt, debt_chg5, growth, inflation)
    z = float(beta[0])
    for j, x in enumerate(X):
        z = z + float(beta[j + 1]) * torch.nan_to_num(x, nan=float(mean_[j]))
    return torch.sigmoid(z)
