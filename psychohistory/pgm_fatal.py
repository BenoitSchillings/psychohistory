"""PRIO-GRID-month (0.5 x 0.5 degree cells, about 55 km) forecasts of state-based conflict fatalities, as full
predictive distributions: the subnational layer, on the VIEWS grid (Africa and the Middle East, 13,110 cells).

Target: UCDP GED state-based deaths whose event location falls in the cell, per month (data vintages as in cm_fatal).
Features at origin t: the cell's log deaths (t, 3, 12, 36-month means, 12-month max), months since its last death,
deaths in the ring of 8 neighbouring cells and within about 5 cells (distance-weighted, 12-month), the country's
deaths (cm, 12-month), the cell's latitude/longitude (absorbs fixed geography) and the horizon.
Training sample: every (cell, origin) where the cell or its 5-cell neighbourhood saw a death in the previous 24 months,
plus a 5% random sample of quiet cell-months weighted x20 (quiet cells almost never see a death next year; sampling
keeps 44 million rows down to a few million). Model: LightGBM quantile regressions as in cm_fatal, horizons sampled.
"""
from __future__ import annotations

import functools
import glob

import numpy as np
import pandas as pd

from . import cm_fatal as CF
from .data import RAW

PC = RAW / "predcomp" / "pgm"
NCOL = 720
TAIL_K, TAIL_C0 = 2.0, 5.0      # draws <= max(c0, k x the largest month in the cell and its ring over 36 months);
                                # chosen on 2014-2017 (scripts/pgm_tune.py): dev CRPS 0.737 -> 0.366
QUANTS = np.array([0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999])


@functools.lru_cache(maxsize=1)
def cells():
    a = pd.read_parquet(glob.glob(str(PC / "Actuals" / "pgm" / "window=Y2018" / "*.parquet"))[0]).reset_index()
    g = np.sort(a.priogrid_gid.unique())
    row, col = (g - 1) // NCOL, (g - 1) % NCOL
    return g, -90 + (row + 0.5) * 0.5, -180 + (col + 0.5) * 0.5


def _agg_cells(path, year_month=None):
    with open(path) as fh:
        head = fh.readline()
    cols = ["type_of_violence", "priogrid_gid", "date_start", "best", "country_id"]
    if "type_of_violence" in head:
        e = pd.read_csv(path, usecols=cols, low_memory=False)
    else:
        e = pd.read_csv(path, header=None, names=CF._ged_columns(), low_memory=False)[cols]
    e = e[e.type_of_violence == 1]
    d = pd.to_datetime(e.date_start, errors="coerce")
    e = e.assign(m=(d.dt.year - 1980) * 12 + d.dt.month)
    if year_month is not None:
        e = e[e.m == CF.mid(*year_month)]
    return e.groupby(["priogrid_gid", "m"]).best.sum()


@functools.lru_cache(maxsize=1)
def _final():
    return _agg_cells(CF.O / "ucdp" / "ged261-csv" / "GEDEvent_v26_1.csv")


def deaths(origin, final_through=None):
    g, _, _ = cells()
    gi = {x: i for i, x in enumerate(g)}
    D = np.zeros((len(g), origin + 1), np.float32)
    last_final = origin if final_through is None else final_through
    for (c, m), v in _final().items():
        if c in gi and m <= last_final:
            D[gi[c], m] = v
    if final_through is not None:
        for m in range(final_through + 1, origin + 1):
            y, mo = 1980 + (m - 1) // 12, (m - 1) % 12 + 1
            for (c, mm), v in _agg_cells(CF.O / "ucdp" / "candidate_monthly" / f"GEDEvent_v{y % 100}_0_{mo}.csv", (y, mo)).items():
                if c in gi:
                    D[gi[c], mm] = v
    return D


@functools.lru_cache(maxsize=1)
def neighbours():
    """sparse neighbour structure: ring of 8, and distance weights within 5 cells (exp(-d/2 cells))"""
    g, lat, lon = cells()
    gi = {x: i for i, x in enumerate(g)}
    ring, wide = [], []
    for i, x in enumerate(g):
        r, c = (x - 1) // NCOL, (x - 1) % NCOL
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                if dr == 0 and dc == 0:
                    continue
                j = gi.get((r + dr) * NCOL + (c + dc) + 1)
                if j is None:
                    continue
                dist = np.hypot(dr, dc)
                wide.append((i, j, np.exp(-dist / 2)))
                if max(abs(dr), abs(dc)) == 1:
                    ring.append((i, j, 1.0))
    import scipy.sparse as sp
    n = len(g)
    mk = lambda L: sp.csr_matrix(([w for _, _, w in L], ([i for i, _, _ in L], [j for _, j, _ in L])), shape=(n, n))
    return mk(ring), mk(wide)


@functools.lru_cache(maxsize=1)
def cell_country():
    """each cell's ViEWS country (most deaths-weighted GED country id seen there; else nearest country centroid)"""
    e = pd.read_csv(CF.O / "ucdp" / "ged261-csv" / "GEDEvent_v26_1.csv", usecols=["priogrid_gid", "country_id"])
    m = e.groupby("priogrid_gid").country_id.agg(lambda s: s.value_counts().index[0])
    g, lat, lon = cells()
    cs = CF.countries()
    gw2i = {x: i for i, x in enumerate(cs.gwcode)}
    out = np.full(len(g), -1)
    for i, x in enumerate(g):
        if x in m.index and m[x] in gw2i:
            out[i] = gw2i[m[x]]
    return out


def features(D, Dc):
    """D [N, T] cell deaths, Dc [C, T] country deaths (same vintage)"""
    ring, wide = neighbours()
    g, lat, lon = cells()
    N, T = D.shape
    ld = np.log1p(D)

    def roll(a, n):
        cs_ = np.cumsum(np.concatenate([np.zeros((a.shape[0], 1), np.float32), a], 1), 1)
        out = (cs_[:, n:] - cs_[:, :-n]) / n
        return np.concatenate([np.full((a.shape[0], n - 1), np.nan, np.float32), out], 1).astype(np.float32)
    F = {"ld": ld.astype(np.float32), "ld3": roll(ld, 3), "ld12": roll(ld, 12), "ld36": roll(ld, 36)}
    mx = np.full(D.shape, np.nan, np.float32)
    for t in range(11, T):
        mx[:, t] = ld[:, t - 11:t + 1].max(1)
    F["max12"] = mx
    since = np.full(D.shape, 400.0, np.float32)
    run = np.full(N, 400.0, np.float32)
    for t in range(T):
        run = np.where(D[:, t] > 0, 0, np.minimum(run + 1, 400))
        since[:, t] = run
    F["since"] = np.log1p(since)
    d12 = roll(D, 12)
    F["ring12"] = np.log1p(np.nan_to_num(ring @ np.nan_to_num(d12)))
    F["wide12"] = np.log1p(np.nan_to_num(wide @ np.nan_to_num(d12)))
    F["ring1"] = np.log1p(ring @ D)
    cc = cell_country()
    Dc12 = np.nan_to_num(pd.DataFrame(Dc).T.rolling(12, min_periods=1).mean().T.to_numpy())
    F["cty12"] = np.where(cc[:, None] >= 0, np.log1p(Dc12[np.clip(cc, 0, None)]), 0).astype(np.float32)
    F["cty1"] = np.where(cc[:, None] >= 0, np.log1p(Dc[np.clip(cc, 0, None)]), 0).astype(np.float32)
    F["lat"] = np.repeat(lat[:, None], T, 1).astype(np.float32)
    F["lon"] = np.repeat(lon[:, None], T, 1).astype(np.float32)
    return F


def local_max(D, origin, months=36):
    """largest monthly death count in each cell and its 8 neighbours over the `months` to the origin"""
    ring, _ = neighbours()
    own = D[:, max(0, origin - months + 1):origin + 1].max(1)
    nb = ring.multiply(own[None, :]).max(1).toarray().ravel()
    return np.maximum(own, nb)


NN_BLEND = 0.0     # weight of the neural quantile model (cm_fatal.NNQuantile), averaged rank by rank before the bound


def train_and_forecast(origin, target_months, final_through=None, start=CF.mid(1995, 1), seed=0, quants=QUANTS):
    D = deaths(origin, final_through)
    Dc = CF.deaths(origin, final_through)
    F = features(D, Dc)
    names = list(F)
    rng = np.random.default_rng(seed)
    hs_all = sorted({m - origin for m in target_months})
    act = (F["ld12"] > 0) | (F["wide12"] > 0)                                 # active in the past year
    act24 = act | np.concatenate([np.zeros((act.shape[0], 12), bool), act[:, :-12]], 1)
    Xs, ys, ws = [], [], []
    for t in range(start, origin - min(hs_all) + 1):
        hs = [h for h in hs_all if t + h <= origin]
        if not hs:
            continue
        a = act24[:, t]
        quiet = ~a & (rng.random(len(a)) < 0.05)
        idx = np.where(a | quiet)[0]
        h = rng.choice(hs, len(idx))                                            # one sampled horizon per row
        X = np.stack([F[f][idx, t] for f in names] + [h.astype(np.float32)], 1)
        Xs.append(X)
        ys.append(np.log1p(D[idx, t + h]))
        ws.append(np.where(a[idx], 1.0, 20.0))
    X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    m = CF.QuantileModel(quants, {"min_child_samples": 200, "n_estimators": 300})
    import lightgbm as lgb
    import warnings
    m.models = []
    m.cap = float(np.log1p(1.5 * D.max()))                                    # largest cell-month seen to the origin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for q in m.q:
            m.models.append(lgb.LGBMRegressor(objective="quantile", alpha=float(q), **m.params).fit(X, y, sample_weight=w))
    mn = None
    if NN_BLEND:
        mn = CF.NNQuantile(quants).fit(X, y, sample_weight=w)
        mn.cap = m.cap
    out = {}
    N = D.shape[0]
    bound = np.maximum(TAIL_C0, TAIL_K * local_max(D, origin))[:, None]
    for tm in target_months:
        Xt = np.stack([F[f][:, origin] for f in names] + [np.full(N, tm - origin, np.float32)], 1)
        dr = m.sample(Xt)
        if mn is not None:
            dr = (1 - NN_BLEND) * np.sort(dr, 1) + NN_BLEND * np.sort(mn.sample(Xt), 1)
        out[tm] = np.minimum(dr, bound).astype(np.float32)
    return out, m, (len(y), names)
