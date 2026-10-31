"""Country-month forecasts of state-based conflict fatalities as full predictive distributions (VIEWS target).

Target: UCDP GED state-based (type 1) best-estimate deaths located in the country (ViEWS country ids / GW codes),
per calendar month. Month index = ViEWS month_id (1 = January 1980).

Data as known at the forecast origin (`vintage`): final GED 26.1 up to the last year whose final release existed at
the origin, then UCDP Candidate monthly releases (data/raw/outcomes/ucdp/candidate_monthly/GEDEvent_vYY_0_M.csv),
as the 2023/24 VIEWS Prediction Challenge participants had for the true-future windows. For the historical windows the
challenge provided final data, so final GED is used throughout.

Features at origin t: log deaths in months t, t-1, t-2; means of log deaths over 3, 6, 12, 24, 60 months; maxima over
12 and 60 months; share of months with any death (12, 60) and with 25+ (24); months since the last death and the
last 25-death month; the 12-month trend (3-month mean minus 12-month mean); deaths within 1,000 km (distance-weighted,
12 months, log); deaths in the same region; and the previous calendar year's income, population and democracy.
Model: LightGBM quantile regressions of log(1 + deaths) at month t + h on the features and h, one per quantile on a
grid, sorted to stay monotone; 1,000 draws by inverse-CDF interpolation on that grid (tails extended linearly on the
log scale), back-transformed.
"""
from __future__ import annotations

import functools
import glob
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import RAW

O = RAW / "outcomes"
QUANTS = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])
GBM = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=100, subsample=0.7, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=5, verbose=-1)


def mid(year, month):
    return (year - 1980) * 12 + month


@functools.lru_cache(maxsize=1)
def countries():
    v = pd.read_csv(O / "views_2025_cm.csv")[["views_country_id", "gwcode", "iso3"]].drop_duplicates("views_country_id")
    return v.sort_values("views_country_id").reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _ged_columns():
    return list(pd.read_csv(O / "ucdp" / "candidate_monthly" / "GEDEvent_v24_0_2.csv", nrows=0).columns)


def _agg(path, year_month=None):
    with open(path) as fh:
        head = fh.readline()
    if "type_of_violence" in head:
        e = pd.read_csv(path, usecols=["type_of_violence", "country_id", "date_start", "best"], low_memory=False)
    else:                                  # (a release without a header row: GEDEvent_v24_0_1)
        e = pd.read_csv(path, header=None, names=_ged_columns(), low_memory=False)[["type_of_violence", "country_id", "date_start", "best"]]
    e = e[e.type_of_violence == 1]
    d = pd.to_datetime(e.date_start, errors="coerce")
    e = e.assign(m=(d.dt.year - 1980) * 12 + d.dt.month)
    if year_month is not None:
        e = e[e.m == mid(*year_month)]
    return e.groupby(["country_id", "m"]).best.sum()


@functools.lru_cache(maxsize=2)
def _final():
    return _agg(O / "ucdp" / "ged261-csv" / "GEDEvent_v26_1.csv")


VINTAGE_W = 0.25    # damping of the vintage correction, chosen on 2022-2023 revisions (scripts/vintage_correction_fit.py)


@functools.lru_cache(maxsize=8)
def revision(year):
    """per-country log revision of the candidate releases of `year`: log((final + 25) / (candidate + 25)) summed over
    the twelve months; {gwcode: r}. Final = GED 26.1 (participants would have had the year's first final release)."""
    fin = _final()
    c_tot, f_tot = {}, {}
    for mo in range(1, 13):
        f = O / "ucdp" / "candidate_monthly" / f"GEDEvent_v{year % 100}_0_{mo}.csv"
        for (g, _), v in _agg(f, (year, mo)).items():
            c_tot[g] = c_tot.get(g, 0) + v
    m0, m1 = mid(year, 1), mid(year, 12)
    for (g, m), v in fin.items():
        if m0 <= m <= m1:
            f_tot[g] = f_tot.get(g, 0) + v
    return {g: float(np.log((f_tot.get(g, 0) + 25) / (c_tot.get(g, 0) + 25))) for g in set(c_tot) | set(f_tot)}


def deaths(origin, final_through=None, correct_w=None):
    """[C, origin + 1] deaths by month_id 0..origin. final_through: last month_id taken from final GED; later months
    up to the origin from the candidate monthly releases (None: final GED for everything). Candidate months are scaled
    per country by exp(w * r), r the revision of the last final year's candidate releases (w = VINTAGE_W; 0 = off)."""
    cs = countries()
    gw2i = {g: i for i, g in enumerate(cs.gwcode)}
    D = np.zeros((len(cs), origin + 1))
    last_final = origin if final_through is None else final_through
    for (g, m), v in _final().items():
        if g in gw2i and m <= last_final:
            D[gw2i[g], m] = v
    if final_through is not None:
        for m in range(final_through + 1, origin + 1):
            y, mo = 1980 + (m - 1) // 12, (m - 1) % 12 + 1
            f = O / "ucdp" / "candidate_monthly" / f"GEDEvent_v{y % 100}_0_{mo}.csv"
            if not f.exists():
                raise FileNotFoundError(f)
            for (g, mm), v in _agg(f, (y, mo)).items():
                if g in gw2i:
                    D[gw2i[g], mm] = v
        w = VINTAGE_W if correct_w is None else correct_w
        fy = 1980 + (final_through - 1) // 12
        if w and (O / "ucdp" / "candidate_monthly" / f"GEDEvent_v{fy % 100}_0_12.csv").exists():
            r = revision(fy)
            mult = np.array([np.exp(w * r.get(g, 0.0)) for g in cs.gwcode])
            D[:, final_through + 1:] *= mult[:, None]
    return D


@functools.lru_cache(maxsize=1)
def _static():
    from .data import get_panel
    from .features import base_arrays
    from .risks import distances
    p = get_panel()
    b = base_arrays(p)
    cs = countries()
    W = np.exp(-np.nan_to_num(distances(list(cs.iso3)), nan=1e9) / 1000.0)
    np.fill_diagonal(W, 0)
    reg = np.array([p.regions[p.codes.index(c)] if c in p.codes else -1 for c in cs.iso3])
    S = {}
    for name in ("lg", "lpop", "dem"):
        a = np.full((len(cs), len(p.years)), np.nan)
        for i, c in enumerate(cs.iso3):
            if c in p.codes:
                a[i] = b[name][p.codes.index(c)]
        S[name] = pd.DataFrame(a).T.ffill().T.to_numpy()      # carry the last value forward
    return W, reg, S, p.years


@functools.lru_cache(maxsize=1)
def _event_shares():
    """[C, T_max] monthly shares of fighting, material conflict, protest and mass violence among all machine-coded
    events (ICEWS to March 2023, POLECAT April 2023 - June 2024, GDELT after when built): shares rather than counts,
    so the source changes matter less"""
    cs = countries()
    ci = {c: i for i, c in enumerate(cs.iso3)}
    T = mid(2027, 1)
    out = {k: np.full((len(cs), T), np.nan) for k in ("fight", "matc", "protest", "massv")}
    src = [(RAW / "events" / "icews_cm.csv", (1995, 1), (2023, 3)), (RAW / "events" / "polecat_cm.csv", (2023, 4), (2024, 6))]
    g = RAW / "events" / "gdelt_cm.csv"
    if g.exists():
        src.append((g, (2024, 7), (2026, 12)))
    for path, lo, hi in src:
        d = pd.read_csv(path)
        d["m"] = (d.year - 1980) * 12 + d.month
        d = d[(d.m >= mid(*lo)) & (d.m <= mid(*hi)) & (d.n_events >= 5)]
        for c, m, n, fi, ma, pr, mv in zip(d.iso3, d.m, d.n_events, d.n_fight, d.n_material_conflict, d.n_protest, d.n_mass_violence):
            if c in ci and m < T:
                i = ci[c]
                out["fight"][i, m], out["matc"][i, m] = fi / n, ma / n
                out["protest"][i, m], out["massv"][i, m] = pr / n, mv / n
    return out


USE_EVENTS = False
USE_SUSTAIN = False   # sustained-intensity features (share of 1,000+ months in 24, run of 500+ months)
# None: the target is log(1 + deaths). A feature name (e.g. "ld_m3"): the target is log(1 + deaths) minus that
# feature at the origin, added back when forecasting. Trees cannot forecast beyond the averages in their leaves, and
# the largest wars are rare, so a level target pulls them toward smaller wars; a change target does not.
RESID = "ld_m3"                 # chosen on 2010-2017 (predcomp_tune.py): dev CRPS 17.69 -> 16.11


def features(D):
    """{name: [C, T]} features at every month of D"""
    W, reg, S, years = _static()
    C, T = D.shape
    ld = np.log1p(D)

    def roll(a, n, f="mean"):
        cs_ = np.cumsum(np.concatenate([np.zeros((C, 1)), a], 1), 1)
        if f == "mean":
            out = (cs_[:, n:] - cs_[:, :-n]) / n
            return np.concatenate([np.full((C, n - 1), np.nan), out], 1)
        out = np.full(a.shape, np.nan)
        for t in range(n - 1, T):
            out[:, t] = a[:, t - n + 1:t + 1].max(1)
        return out

    def lag(a, k):
        return np.concatenate([np.full((C, k), np.nan), a[:, :-k]], 1)

    F = {"ld": ld, "ld_l1": lag(ld, 1), "ld_l2": lag(ld, 2)}
    for n in (3, 6, 12, 24, 60):
        F[f"ld_m{n}"] = roll(ld, n)
    F["ld_max12"], F["ld_max60"] = roll(ld, 12, "max"), roll(ld, 60, "max")
    F["any12"], F["any60"] = roll((D > 0).astype(float), 12), roll((D > 0).astype(float), 60)
    F["hit24"] = roll((D >= 25).astype(float), 24)
    for nm, thr in (("since_any", 0.5), ("since25", 25)):
        since = np.full(D.shape, 600.0)
        run = np.full(C, 600.0)
        for t in range(T):
            run = np.where(D[:, t] >= thr, 0, np.minimum(run + 1, 600))
            since[:, t] = run
        F[nm] = np.log1p(since)
    F["trend"] = F["ld_m3"] - F["ld_m12"]
    if USE_SUSTAIN:
        F["hit1000_24"] = roll((D >= 1000).astype(float), 24)
        runh = np.zeros(C)
        rh = np.zeros(D.shape)
        for t in range(T):
            runh = np.where(D[:, t] >= 500, runh + 1, 0)
            rh[:, t] = runh
        F["run500"] = np.log1p(rh)
    d12 = roll(D, 12)
    F["nb12"] = np.log1p(np.nan_to_num(W @ np.nan_to_num(d12)))
    rg = np.full(D.shape, np.nan)
    for r_ in set(reg):
        if r_ >= 0:
            rg[reg == r_] = np.nanmean(F["ld_m12"][reg == r_], 0)
    F["reg12"] = rg
    if USE_EVENTS:
        E = _event_shares()
        for k, a in E.items():
            a = a[:, :T]
            if a.shape[1] < T:
                a = np.concatenate([a, np.full((C, T - a.shape[1]), np.nan)], 1)
            m3 = pd.DataFrame(a).T.rolling(3, min_periods=1).mean().T.to_numpy()
            m12 = pd.DataFrame(a).T.rolling(12, min_periods=3).mean().T.to_numpy()
            F[f"ev_{k}3"], F[f"ev_{k}_d"] = m3, m3 - m12
    yi = {y: i for i, y in enumerate(years)}
    for name, a in S.items():
        out = np.full(D.shape, np.nan)
        for t in range(1, T):
            y = 1980 + (t - 1) // 12 - 1
            if y in yi:
                out[:, t] = a[:, yi[y]]
        F[name] = out
    return F


def design(F, t_list, horizons):
    names = list(F)
    Xs, meta = [], []
    for h in horizons:
        for t in t_list:
            Xs.append(np.stack([F[f][:, t] for f in names] + [np.full(F[names[0]].shape[0], h)], 1))
            meta.append((t, h))
    return np.concatenate(Xs).astype(np.float32), meta, names + ["h"]


class QuantileModel:
    def __init__(self, quants=QUANTS, params=None):
        self.q = np.asarray(quants)
        self.params = {**GBM, **(params or {})}

    def fit(self, X, y, sample_weight=None):
        self.models = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for q in self.q:
                self.models.append(lgb.LGBMRegressor(objective="quantile", alpha=float(q), **self.params)
                                   .fit(X, y, sample_weight=sample_weight))
        return self

    def predict_q(self, X, base=None):
        Q = np.stack([m.predict(X) for m in self.models], 1)
        if base is not None:                                # residual target: add the recent level back
            Q = Q + base[:, None]
        return np.maximum.accumulate(np.clip(Q, 0, None), 1)

    def sample(self, X, n=1000, seed=0, base=None):
        """[N, n] draws of deaths: inverse CDF through the quantile grid (log scale), tails extended linearly"""
        Q = np.sort(self.predict_q(X, base), 1)                 # boosted quantiles can cross
        if getattr(self, "cap", None) is not None:          # and can sum past anything seen in training
            Q = np.minimum(Q, self.cap)
        q = self.q
        u = (np.arange(n) + 0.5) / n
        # extend the grid to 0 and 1 by the slope of the outermost quantiles (log1p scale), floored at 0
        lo = np.clip(Q[:, 0] - (Q[:, 1] - Q[:, 0]) / (q[1] - q[0]) * q[0], 0, None)
        hi = Q[:, -1] + (Q[:, -1] - Q[:, -2]) / (q[-1] - q[-2]) * (1 - q[-1])
        grid = np.concatenate([[0.0], q, [1.0]])
        QQ = np.concatenate([lo[:, None], Q, hi[:, None]], 1)
        out = np.empty((len(X), n))
        for i in range(len(X)):
            out[i] = np.interp(u, grid, QQ[i])
        return np.expm1(out)


HALFLIFE = None    # recency weighting of training months (years; None = equal weights)
NN_BLEND = 0.5     # weight of the neural quantile model, averaged rank by rank with LightGBM (chosen on 2010-2017:
                   # dev CRPS 16.11 -> 15.80; test 2018-2023 47.35 -> 46.33, live 32.0/31.9 -> 28.9/29.6)


class NNQuantile(QuantileModel):
    """MLP (2 x 256, missing-value indicators, standardised inputs) with pinball loss on the quantile grid, 5 seeds"""

    def __init__(self, quants=QUANTS, seeds=5, epochs=6, hidden=256, lr=2e-3):
        self.q = np.asarray(quants)
        self.seeds, self.epochs, self.hidden, self.lr = seeds, epochs, hidden, lr

    def _z(self, X):
        Z = np.nan_to_num((X - self.mu) / self.sd).clip(-8, 8)
        return np.concatenate([Z, np.isnan(X).astype(np.float32)], 1).astype(np.float32)

    def fit(self, X, y, sample_weight=None):
        import torch
        from torch import nn
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = dev
        self.mu, self.sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
        Z = torch.as_tensor(self._z(X), device=dev)
        Y = torch.as_tensor(y.astype(np.float32), device=dev)
        qt = torch.as_tensor(self.q, dtype=torch.float32, device=dev)
        Wt = torch.as_tensor((sample_weight / sample_weight.mean()).astype(np.float32), device=dev) if sample_weight is not None else None
        self.nets = []
        for s in range(self.seeds):
            torch.manual_seed(s)
            net = nn.Sequential(nn.Linear(Z.shape[1], self.hidden), nn.GELU(), nn.Dropout(0.1),
                                nn.Linear(self.hidden, self.hidden), nn.GELU(), nn.Linear(self.hidden, len(self.q))).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=self.epochs * (len(Z) // 4096 + 1))
            for _ in range(self.epochs):
                perm = torch.randperm(len(Z), device=dev)
                for b in range(0, len(Z), 4096):
                    i = perm[b:b + 4096]
                    e = Y[i, None] - net(Z[i])
                    pl = torch.maximum(qt * e, (qt - 1) * e)
                    loss = (pl * Wt[i, None]).mean() if Wt is not None else pl.mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
            net.eval()
            self.nets.append(net)
        return self

    def predict_q(self, X, base=None):
        import torch
        with torch.no_grad():
            Z = torch.as_tensor(self._z(X), device=self.dev)
            Q = torch.stack([n(Z) for n in self.nets]).mean(0).cpu().numpy()
        if base is not None:
            Q = Q + base[:, None]
        return np.maximum.accumulate(np.clip(np.sort(Q, 1), 0, None), 1)


def train_and_forecast(origin, target_months, final_through=None, start=mid(1995, 1), quants=QUANTS, params=None, correct_w=None):
    """train on every (t, h) with t + h <= origin, forecast the months in target_months from the origin.
    Returns {month_id: [C, 1000] draws}"""
    D = deaths(origin, final_through, correct_w)
    F = features(D)
    hs = sorted({m - origin for m in target_months})
    base = (lambda ts: np.concatenate([np.nan_to_num(F[RESID][:, t]) for t in ts])) if RESID else (lambda ts: 0.0)
    X_parts, y_parts, w_parts = [], [], []
    for h in hs:
        ts = list(range(start, origin - h + 1))
        X, _, _ = design(F, ts, [h])
        y = np.concatenate([np.log1p(D[:, t + h]) for t in ts]) - base(ts)
        X_parts.append(X)
        y_parts.append(y)
        if HALFLIFE:
            w_parts.append(np.concatenate([np.full(D.shape[0], 0.5 ** ((origin - t) / (12 * HALFLIFE))) for t in ts]))
    X, y = np.concatenate(X_parts), np.concatenate(y_parts)
    sw = np.concatenate(w_parts) if HALFLIFE else None
    m = QuantileModel(quants, params).fit(X, y, sample_weight=sw)
    mn = NNQuantile(quants).fit(X, y, sample_weight=sw) if NN_BLEND else None
    out = {}
    for tm in target_months:
        Xt, _, _ = design(F, [origin], [tm - origin])
        out[tm] = m.sample(Xt, base=base([origin]) if RESID else None)
        if mn is not None:
            out[tm] = np.expm1((1 - NN_BLEND) * np.log1p(np.sort(out[tm], 1))
                               + NN_BLEND * np.log1p(np.sort(mn.sample(Xt, base=base([origin]) if RESID else None), 1)))
    return out, m
