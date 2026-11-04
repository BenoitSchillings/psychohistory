"""The Prime Radiant: Monte Carlo simulation of all countries, jointly, on the GPU.

Each scenario is a full alternative history of the world from an origin year.
Per simulated year:
  1. structural features are recomputed from the simulated state (same code as training)
  2. civil conflict starts/ends with the hazards of a PsychoNet ensemble member
     (each scenario uses one member -> model uncertainty is part of the fan)
  3. GDP per capita and democracy move by learned transition equations plus
     residuals bootstrapped from history, including a *global* shock drawn from a
     historical year, so world-wide crises (1974, 1982, 2009, 2020) hit all
     countries of a scenario together
  4. demography follows UN WPP projections; urbanisation, child mortality,
     temperature and technology follow damped trends; slow-moving survey data persist
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import Ridge

from .features import CURRENT_VARS, DECAY, FEATURES, TASKS, W, WINDOW_VARS, compute_features
from .models import DEVICE, PsychoNet

ON, TERM, COUP = TASKS.index("onset_1y"), TASKS.index("termination_1y"), TASKS.index("coup_1y")
EVENTS = ["conflict_next", "onset_next", "coup_next", "interstate_next"]   # same-year events in the transition equations
QS = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])


# fiscal capacity added 2026-10-28 (scripts/growth_screen.py: better next-year and 10-year growth forecasts on dev and
# test; hindcasts sim_eval gr_fc: country GDP CRPS -0.5% dev (12/15), -1.7% test (5/7), world GDP CRPS -1.8% / -3.1%).
# log GDP alongside the frontier gap was rejected: world GDP CRPS +45% dev, +40% test
GROWTH_FEATURES = ["gdp_gap_frontier", "growth_1y", "growth_5y", "gdp_drawdown", "growth_vol_10y",
                   "conflict_now", "democracy", "democracy_sq", "democracy_change_5y",
                   "region_conflict", "inflation_log", "pop_growth_5y", "fiscal_capacity"]
# free-expression erosion (5-year change) and neopatrimonial rule added 2026-10-28 (scripts/dem_screen.py: 25 V-Dem
# variables improved democracy-change forecasts on dev and test, this pair the most; hindcasts sim_eval dm_fn:
# democracy CRPS -3.0% dev (14/15), -6.5% test (7/7), other scores within 1.4%)
# personalist legitimation added 2026-10-30, with free expression given its own equation (dynamics.INST_SLOW): it beat
# baseline in all four dem_screen quadrants; hindcasts sim_eval l2dem3 (2 network seeds): democracy CRPS -1.2% dev
# (12/15), -4.8% test (7/7), coup count CRPS +0.5% / +1.3% (seed 1: -0.2% / +0.3%), other scores within 1%; it moves
# with democracy (COMOVE): frozen at 2025 levels it would act on democracy for forty years unchanged
DEM_FEATURES = ["democracy", "democracy_sq", "democracy_change_5y", "region_dem_gap", "conflict_now",
                "growth_1y", "gdp_drawdown", "gdp_gap_frontier", "freexp_change_5y", "neopatrimonialism",
                "legit_personalist"]


def _pools(values, bins, n_bins):
    """ragged per-bin sample pools (an empty bin gets a single zero)."""
    return [values[bins == b] if (bins == b).any() else np.zeros(1) for b in range(n_bins)]


class _Ragged:
    """per-bin pools on the GPU, sampled uniformly within each bin."""

    def __init__(self, pools, device):
        self.vals = torch.as_tensor(np.concatenate(pools), dtype=torch.float32, device=device)
        lens = np.array([len(p_) for p_ in pools])
        self.len = torch.as_tensor(lens, device=device)
        self.off = torch.as_tensor(np.concatenate([[0], np.cumsum(lens)[:-1]]), device=device)

    def draw(self, bins, gen):
        u = torch.rand(bins.shape, device=bins.device, generator=gen)
        i = self.off[bins] + torch.minimum((u * self.len[bins]).long(), self.len[bins] - 1)
        return self.vals[i]


@dataclass
class Transition:
    """y_{t+1} - y_t = f(selected features_t, conflict_{t+1}, onset_{t+1}, coup_{t+1})
                       + global shock of a historical year
                       + regional era (region-decade) + regime offset (country-decade) + noise

    * Only a small theory-motivated feature set is used: trending variables (technology
      adoption, fertility...) would otherwise act as era dummies and extrapolate absurdly.
    * Global shocks are block-bootstrapped by decade, so a scenario lives through a
      coherent historical era (a 1950s boom, a 1980s debt-crisis decade...). Decades are
      drawn in proportion to the years they contain, so the short 2020s do not dominate.
    * Regimes and noise are drawn from countries in the same bin of `bin_col`
      (income level for growth, regime type for democracy): rich countries are less volatile."""
    cols: list
    squares: list
    mu: np.ndarray
    sd: np.ndarray
    coef: np.ndarray
    intercept: float
    year_shocks: np.ndarray   # pooled residual mean per year of `shock_years` (fixed grid)
    shock_years: np.ndarray
    bin_col: str
    edges: np.ndarray
    region_eras: np.ndarray   # pool of region-decade mean residuals (regional booms / lost decades)
    regimes: list             # per bin: country-decade mean residuals
    resid: list               # per bin: remaining idiosyncratic residuals
    lo: float
    hi: float
    size_b: float = 0.0       # log noise scale per unit log population (SIZE_NOISE; 0 = no size scaling)
    size_c: float = 0.0       # (centre of log population)
    size_regime: bool = False
    z_lo: np.ndarray = None   # training support of each standardised feature (0.5-99.5%): no extrapolation beyond it
    z_hi: np.ndarray = None
    inter: tuple = ()         # interaction columns (feature index pairs), appended after cols
    vol_b: float = 0.0        # slope of log|noise| on standardised growth volatility (VOL_NOISE)
    vol_mu: float = 0.0
    vol_sd: float = 1.0
    pool_state: str = None    # noise pools split by this feature (top third vs rest; POOL_STATE)
    pool_edge: float = 0.0

    @staticmethod
    def _design(z, ev, sq_idx, xp):
        cat = np.concatenate if xp is np else torch.cat
        return cat([z, z[..., sq_idx] ** 2, ev], -1)

    @staticmethod
    def fit(X, events, target, years, codes, regions, cols, lo, hi, bin_col, grid, squares=(), alpha=10.0,
            n_bins=4, halflife=None, size=None, shock_w=None, inter=(), vol=False, pool_state=None):
        ci = [FEATURES.index(c) for c in cols]
        inter = tuple((FEATURES.index(a_), FEATURES.index(b_)) for a_, b_ in inter)
        gv_all = X[:, FEATURES.index("growth_vol_10y")] if vol else None
        ok = np.isfinite(target) & np.isfinite(events).all(1)
        bv = X[ok][:, FEATURES.index(bin_col)]
        lp = X[ok][:, FEATURES.index("log_pop")] if size else None
        Xo = X[ok]
        X, ev, y, yr, cc, rg = (Xo[:, ci], events[ok], np.clip(target[ok], lo, hi), years[ok], codes[ok], regions[ok])
        if inter:
            X = np.c_[X, np.stack([Xo[:, a_] * Xo[:, b_] for a_, b_ in inter], 1)]
        gv = gv_all[ok] if vol else None
        sw = np.ones(ok.sum()) if shock_w is None else np.nan_to_num(shock_w[ok])
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
        sq_idx = [cols.index(c) for c in squares]
        Z = Transition._design(np.nan_to_num(np.clip((X - mu) / sd, -6, 6)), ev, sq_idx, np)
        z_lo, z_hi = ((np.full(X.shape[1], s_) if SUPPORT_Q is None else np.nan_to_num(np.nanquantile((X - mu) / sd, q, 0), nan=s_))
                      for q, s_ in ((SUPPORT_Q, -6.0), (1 - (SUPPORT_Q or 0), 6.0)))
        # optional recency weighting: the structural mean leans on recent decades
        w = None if halflife is None else 0.5 ** ((yr.max() - yr) / halflife)
        m = Ridge(alpha=alpha).fit(Z, y, sample_weight=w)
        r = y - m.predict(Z)
        gi = np.searchsorted(grid, yr)
        n_y = np.bincount(gi, sw, minlength=len(grid))
        shocks = np.where(n_y > 0, np.bincount(gi, r * sw, minlength=len(grid)) / np.where(n_y > 0, n_y, 1), 0.0)
        if shock_w is not None:
            # (weighted shocks carry the timing of world booms and busts; their mean over all country-years is the
            # equation's, not the big economies' average residual, or replaying them would drift every country)
            shocks = np.where(n_y > 0, shocks - shocks[gi].mean(), 0.0)
        r = r - shocks[gi]
        edges = np.nanquantile(bv, np.linspace(0, 1, n_bins + 1)[1:-1])
        bins = np.digitize(np.nan_to_num(bv, nan=np.nanmedian(bv)), edges)
        reg_dec, rk, rn = Transition._shrunk_means(r, rg * 10000 + yr // 10)
        r = r - reg_dec[rk]
        dec, ki, n = Transition._shrunk_means(r, cc * 10000 + yr // 10)
        idio = r - dec[ki]
        size_b = size_c = 0.0
        if size:
            # large countries are less volatile (diversified): pooled residuals are put on a common size scale and
            # rescaled to the drawing country's population (slope of log|residual| on log population within bins)
            lp = np.nan_to_num(lp, nan=np.nanmean(lp))
            size_c = float(lp.mean())
            D_ = np.c_[np.ones(len(lp)), lp - size_c, np.eye(n_bins)[bins][:, 1:]]
            size_b = float(np.linalg.lstsq(D_, np.log(np.abs(idio) + 1e-3), rcond=None)[0][1])
            idio = idio / np.exp(size_b * (lp - size_c))
            if size == "both":
                dec = dec / np.exp(size_b * (np.bincount(ki, lp) / n - size_c))
        vol_b, vol_mu, vol_sd = 0.0, 0.0, 1.0
        if vol:
            # quiet economies draw quieter noise: slope of log|residual| on standardised 10-year growth volatility,
            # within bins; pooled residuals are put on a common scale and rescaled to the drawing country's volatility
            gv = np.nan_to_num(gv, nan=np.nanmedian(gv))
            vol_mu, vol_sd = float(gv.mean()), float(gv.std() + 1e-9)
            zv = np.clip((gv - vol_mu) / vol_sd, -3, 3)
            D_ = np.c_[np.ones(len(zv)), zv, np.eye(n_bins)[bins][:, 1:]]
            vol_b = float(np.linalg.lstsq(D_, np.log(np.abs(idio) + 1e-3), rcond=None)[0][1])
            idio = idio / np.exp(vol_b * zv)
        dec_bin = np.round(np.bincount(ki, bins) / n).astype(int)   # typical bin of the country-decade
        pool_edge, n_ib = 0.0, n_bins
        if pool_state:
            # (state-conditional noise: a high-debt country draws from what high-debt countries went through)
            sv = Xo[:, FEATURES.index(pool_state)]
            pool_edge = float(np.nanquantile(sv, 2 / 3))
            hi_ = np.nan_to_num(sv, nan=-np.inf) > pool_edge
            if POOL_CENTER:
                # keep only the shape of each sub-pool: both halves get their parent bin's mean, so debt does not
                # enter the growth mean through the noise (it was rejected as a growth-equation term)
                idio = idio.copy()
                for b_ in range(n_bins):
                    par = bins == b_
                    for s_ in (False, True):
                        sub = par & (hi_ == s_)
                        if sub.any():
                            idio[sub] += idio[par].mean() - idio[sub].mean()
            bins = bins * 2 + hi_
            n_ib = n_bins * 2
        keep = n >= 5
        if POOL_HALFLIFE:
            # recent history counts more: country volatility has fallen since the 1990s (hindcast bands too wide from
            # 2000 origins, too narrow before); pools are resampled (fixed seed) with weights by year / decade
            rs = np.random.default_rng(int(yr.max()))
            wi = 0.5 ** ((yr.max() - yr) / POOL_HALFLIFE)
            ii = rs.choice(len(idio), len(idio), p=wi / wi.sum())
            idio, bins = idio[ii], bins[ii]
            dyr = np.bincount(ki, yr) / n
            wd = np.where(keep, 0.5 ** ((yr.max() - dyr) / POOL_HALFLIFE), 0)
            jj = rs.choice(len(dec), int(keep.sum()), p=wd / wd.sum())
            dec, dec_bin, keep = dec[jj], dec_bin[jj], np.ones(len(jj), bool)
        return Transition(cols, list(squares), mu, sd, m.coef_, float(m.intercept_), shocks, grid, bin_col, edges,
                          reg_dec[rn >= 20], _pools(dec[keep], dec_bin[keep], n_bins), _pools(idio, bins, n_ib), lo, hi,
                          size_b, size_c, size == "both", z_lo, z_hi, inter, vol_b, vol_mu, vol_sd, pool_state, pool_edge)

    @staticmethod
    def _shrunk_means(r, key):
        """group means shrunk toward 0 (noise in short windows would overstate persistence)."""
        _, ki = np.unique(key, return_inverse=True)
        n = np.bincount(ki)
        m = np.bincount(ki, r) / n
        return m * np.clip(1 - r.var() / np.maximum(n * m ** 2, 1e-12), 0, 1), ki, n

    def to(self, device):
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
        self.ci_t = torch.as_tensor([FEATURES.index(c) for c in self.cols], device=device)
        self.sq_t = [self.cols.index(c) for c in self.squares]
        self.bi = FEATURES.index(self.bin_col)
        self.si = FEATURES.index("log_pop")
        self.mu_t, self.sd_t, self.w, self.edges_t = f(self.mu), f(self.sd), f(self.coef), f(self.edges)
        self.zlo_t = f(self.z_lo if self.z_lo is not None else np.full(len(self.cols), -6.0))
        self.zhi_t = f(self.z_hi if self.z_hi is not None else np.full(len(self.cols), 6.0))
        self.shocks_t, self.eras_t = f(self.year_shocks), f(self.region_eras)
        self.regimes_t, self.resid_t = _Ragged(self.regimes, device), _Ragged(self.resid, device)
        self.decade_t = torch.as_tensor(self.shock_years // 10, device=device)
        dec, cnt = np.unique(self.shock_years // 10, return_counts=True)
        self.decades_t, self.decade_w = torch.as_tensor(dec, device=device), f(cnt)
        return self

    def _bin(self, X):
        v = X[..., self.bi]
        return torch.bucketize(torch.nan_to_num(v, nan=float(self.edges[len(self.edges) // 2])), self.edges_t)

    def _size(self, X):
        lp = torch.nan_to_num(X[..., self.si], nan=self.size_c)
        return torch.exp(self.size_b * (lp - self.size_c)).clamp(0.3, 3.0)

    def draw_regime(self, X, gen):
        r = self.regimes_t.draw(self._bin(X), gen)
        return r * self._size(X) if self.size_b and self.size_regime else r

    def draw_region_era(self, S, R, gen):
        return self.eras_t[torch.randint(len(self.eras_t), (S, R), device=self.w.device, generator=gen)]

    def draw_era(self, S, gen):
        """a historical decade per scenario, weighted by the number of years it contains."""
        return self.decades_t[torch.multinomial(self.decade_w.expand(S, -1), 1, replacement=True, generator=gen)[:, 0]]

    def draw_year(self, decade, gen):
        """random historical year (index into year_shocks) inside each scenario's decade."""
        cand = (self.decade_t[None, :] == decade[:, None]).float()
        return torch.multinomial(cand, 1, generator=gen)[:, 0]

    def step(self, X, events, shock_idx, regime, gen):
        # (held inside the training support: the squared income-gap term would otherwise turn a war-torn economy that
        # falls below any observed level into an accelerating collapse; 30-year hindcasts from 1994, code review 2026-10-29)
        Xs = X[..., self.ci_t]
        if self.inter:
            Xs = torch.cat([Xs] + [(X[..., a_] * X[..., b_])[..., None] for a_, b_ in self.inter], -1)
        z = torch.nan_to_num(torch.minimum(torch.maximum((Xs - self.mu_t) / self.sd_t, self.zlo_t), self.zhi_t))
        mean = self._design(z, events, self.sq_t, torch) @ self.w + self.intercept
        ib = self._bin(X)
        if self.pool_state:
            ib = ib * 2 + (torch.nan_to_num(X[..., FEATURES.index(self.pool_state)], nan=-float("inf")) > self.pool_edge).long()
        eps = self.resid_t.draw(ib, gen)
        if self.size_b:
            eps = eps * self._size(X)
        if self.vol_b:
            gv = torch.nan_to_num(X[..., FEATURES.index("growth_vol_10y")], nan=self.vol_mu)
            eps = eps * torch.exp(self.vol_b * ((gv - self.vol_mu) / self.vol_sd).clamp(-3, 3)).clamp(0.3, 3.0)
        sh = self.shocks_t[shock_idx]
        if ASYM_SHOCK is not None:
            # booms and busts scaled apart, recentred over the replayed years so the average shock is unchanged
            if getattr(self, "_asym_mu", None) is None:
                a_all = torch.where(self.shocks_t > 0, ASYM_SHOCK[0] * self.shocks_t, ASYM_SHOCK[1] * self.shocks_t)
                self._asym_mu = float(a_all.mean() - getattr(self, "shock_scale", 1.0) * self.shocks_t.mean())
            sh = torch.where(sh > 0, ASYM_SHOCK[0] * sh, ASYM_SHOCK[1] * sh) - self._asym_mu
        else:
            sh = getattr(self, "shock_scale", 1.0) * sh
        return (mean + sh[:, None] + regime + eps).clamp(self.lo, self.hi)

    def coefficients(self):
        names = self.cols + [f"{FEATURES[a_]}*{FEATURES[b_]}" for a_, b_ in self.inter]
        return dict(zip(names + [f"{c}^2" for c in self.squares] + EVENTS, self.coef))


# growth noise scaled by country size: None, "idio" (yearly noise) or "both" (+ decade regimes). Adopted ("both")
# 2026-10-28: the pools drew small-country volatility for China and India (annual growth sd 1960-2024 falls with
# population within income bins: top 5% by population 0.03-0.04 vs 0.06-0.09 for the smallest third), so simulated
# world growth was too volatile (2026-2065 worlds: sd 2.6%/yr vs 1.5% in 1960-2025; world GDP per person fell in 22% of
# years vs 3%). Hindcasts, 3 network seeds each (sim_eval szb*/co2,ns1,ns2): world GDP CRPS -6.9% dev (11/15), -2.3%
# test (4/7), country GDP CRPS 0.0% / -0.3%, conflict, coup and democracy scores unchanged, 90% GDP coverage -0.8 pt;
# annual world volatility vs realised, pooled over the 22 origins, 1.25 -> 1.02 (scripts/disaster_rca.py found it)
SIZE_NOISE = "both"


def fit_transitions(p, X, base, origin, halflife=None):
    """growth and democracy-change equations, trained on transitions ending by `origin`."""
    C, Yn, F = X.shape
    lg, dem, conf = base["lg"], base["dem"], base["conflict"]
    t = np.arange(Yn - 1)
    years = np.broadcast_to(p.years[t], (C, len(t)))
    codes = np.broadcast_to(np.arange(C)[:, None], (C, len(t)))
    regions = np.broadcast_to(np.asarray(p.regions)[:, None], (C, len(t)))
    ok_year = (years >= 1955) & (years + 1 <= origin)
    onset_next = np.where(np.isfinite(conf[:, 1:]) & np.isfinite(conf[:, :-1]),
                          (conf[:, 1:] == 1) & (conf[:, :-1] == 0), np.nan)
    ca = base["coup_attempts"][:, 1:]
    coup_next = np.where(np.isfinite(ca), ca > 0, np.nan)
    flat = lambda a: a[ok_year]
    Xf = X[:, :-1][ok_year]
    inter = base["interstate"][:, 1:]
    inter_next = np.where(np.isfinite(inter), np.nan_to_num(inter), np.nan)
    events = np.stack([flat(conf[:, 1:]), flat(onset_next.astype(float)), flat(coup_next.astype(float)),
                       flat(inter_next)], 1)
    grid = np.arange(1955, origin)   # same year grid for both equations -> one shock index serves both
    args = (Xf, events)
    growth = Transition.fit(*args, flat(lg[:, 1:] - lg[:, :-1]), flat(years), flat(codes), flat(regions),
                            GROWTH_FEATURES, -0.4, 0.3, bin_col="gdp_gap_frontier", grid=grid,
                            squares=["gdp_gap_frontier"], halflife=halflife, size=SIZE_NOISE, vol=VOL_NOISE, pool_state=POOL_STATE,
                            shock_w=flat(np.exp(YEAR_SHOCK_POW * (lg[:, :-1] + base["lpop"][:, :-1] - 25)))
                            if YEAR_SHOCK_POW else None)
    ddem = Transition.fit(*args, flat(dem[:, 1:] - dem[:, :-1]), flat(years), flat(codes), flat(regions),
                          DEM_FEATURES, -0.5, 0.5, bin_col="democracy", grid=grid, halflife=halflife, inter=DEM_INTER,
                          size=DEM_SIZE)
    if DEM_DRIFT is not None:
        # the equation's level carries the average democracy change of its training years (1955..origin: the third
        # wave, +0.006 a year), which the simulation would extrapolate for decades. Replace it: "zero" (no trend, as
        # for the slow variables), a number (that share of it), or "longrun" (the average yearly change since 1900,
        # reversals of the 1920s-30s included). Tested 2026-11-02.
        dy = flat(dem[:, 1:] - dem[:, :-1])
        m_tr = float(np.nanmean(dy))
        if DEM_DRIFT == "longrun":
            yy = np.asarray(p.years)
            seg = (yy[1:] >= 1900) & (yy[1:] <= origin)
            target = float(np.nanmean((dem[:, 1:] - dem[:, :-1])[:, seg]))
        elif DEM_DRIFT == "zero":
            target = 0.0
        else:
            target = float(DEM_DRIFT) * m_tr
        ddem.intercept += target - m_tr
    return growth, ddem


# V-Dem indicators that move with the democracy index (shifted in proportion to simulated change)
COMOVE = ["rule_of_law", "physical_integrity", "political_killings", "religious_freedom",
          "religious_repression", "corruption", "neopatrimonialism",
          "jud_constraints", "leg_constraints", "horiz_acc", "diag_acc", "free_expression", "media_censorship",
          "cso_repression", "free_fair_elections", "elected_regime", "liberal_dem", "civil_liberties",
          "cso_participation", "exec_respects_const", "clientelism", "legit_personalist"]
# institution levels whose 5-year change is a feature: the change is recomputed from simulated levels
INST_CHANGE = {"jucon_change_5y": "jud_constraints", "legcon_change_5y": "leg_constraints",
               "freexp_change_5y": "free_expression", "libdem_change_5y": "liberal_dem",
               "csrepress_change_5y": "cso_repression"}


def comove_slopes(base, origin, years):
    """pooled OLS slope of the yearly change of each indicator on the yearly democracy change."""
    dd = np.diff(base["dem"], axis=1)
    ok_t = (years[1:] <= origin)[None, :]
    out = {}
    for k in COMOVE:
        dx = np.diff(base[k], axis=1)
        ok = ok_t & np.isfinite(dd) & np.isfinite(dx)
        out[k] = float((dd[ok] * dx[ok]).sum() / (dd[ok] ** 2).sum())
    return out


_SLOW_CACHE = {}


def _slow_equations(base, years, origin):
    """slow-variable equations fitted on data before the origin (cached per origin)"""
    if origin not in _SLOW_CACHE:
        from .dynamics import fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _SLOW_CACHE[origin] = fit(base, np.asarray(years), origin)
    return _SLOW_CACHE[origin]


def crisis_given_growth(base, origin, years, thresh=-0.05):
    """P(financial crisis in year t | growth_t below / above `thresh`), used to regenerate
    the decaying crisis memory inside the simulation."""
    g = np.diff(base["lg"], axis=1)
    fc = base["fin_crisis"][:, 1:]
    ok = np.isfinite(g) & np.isfinite(fc) & (years[None, 1:] <= origin) & (years[None, 1:] >= 1955)
    low = g < thresh
    return float(fc[ok & low].mean()), float(fc[ok & ~low].mean())


def _trend_rate(a, t0, years=10):
    seg = a[:, max(0, t0 - years):t0 + 1]
    d = np.diff(seg, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(np.where(np.isfinite(d), d, np.nan), 1) if d.shape[1] else np.zeros(len(a))


def _logistic_rate(a, t0, cap=100.0, years=10):
    seg = a[:, max(0, t0 - years):t0 + 1]
    d = np.diff(seg, axis=1)
    x = seg[:, :-1]
    denom = x * (cap - x) / cap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = np.nanmean(np.where(denom > 0.5, d / np.maximum(denom, 1e-6), np.nan), 1)
    return np.nan_to_num(np.clip(r, 0, 0.3))


def _nanmedian_recent(a, t0, years=20):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(a[:, max(0, t0 - years + 1):t0 + 1], 1)


# leaders: yearly chance of dying or leaving for ill health, logit in age with an era trend (Archigos 1900-2015,
# scripts/succession_test.py); a death in office in an autocracy raises the coup log-odds for two years (+0.68, s.e. 0.25)
from .structure import FRAG_MEAN, FRAG_NEW, FRAG_LONG, FRAG_TAU
LEADER_DEATH = (-9.006, 0.0728, -1.277)
SUCC_COUP = 0.68
STRUCT_CONVEXITY = 1.0      # (test switches, code review 2026-10-28)
# structural uncertainty (which risk model is right): "gauss" = one shared draw per world and hazard scaled by the model
# families' disagreement (net.struct_sd); "family" = each world and hazard draws one family with the ensemble's weights
# and takes its log-odds offset from PsychoNet (net.struct_fam, net.struct_w); "familyc" = the same, offsets centred
# on their weighted mean (spread without a level shift); "off" = none (pipeline.struct_inputs)
# Adopted "familyc" 2026-11-01 (scripts/world_hindcast.py --struct, the risk ensemble refitted at each origin, 2,000
# worlds): against "gauss", war onsets CRPS -5.5% at 15 years from 1980-1995 (4/4), -27% from 2000-2009 (3/3), -7% at
# 30 years; conflict share -12% from 2000-2009; the safe-country tail +5-8%. Against "off" within +/-2% except coups.
STRUCT_MODE = "familyc"
LREB_MODE = "age"
SHIFTERS_BEFORE_ALIGN = True   # see the year-1 alignment in simulate()
# limits round 2 (2026-10-30): switches tested and rejected, kept off (docs/world_system.md, "Limits round 2")
# (year shocks per income tier beyond the world one were measured 2026-11-02, empirical-Bayes shrunk: sd 0.1-0.2% a
#  year against 1.2% for the world shock - nothing to replay, not added)
OIL_RESID = False         # oil: replay residuals of a fitted pull + war-entry equation instead of raw moves (tested 2026-11-02)
OIL_EXPORTERS = ["SAU", "IRN", "IRQ", "KWT", "ARE", "QAT", "OMN", "RUS", "LBY", "VEN", "KAZ", "DZA", "NGA"]
DEM_REGIME_SCALE = 1.0     # country-decade democracy regimes x this
GROWTH_REGIME_SCALE = 1.4  # country-decade growth regimes x this (1.8 until 2026-11-02; docs/world_system.md)
DEM_DRIFT = None          # democracy equation's trend: None (1955..origin average), "zero", "longrun", or a share
DEM_SIZE = None           # democracy noise scaled by population like growth's ("both"; tested 2026-11-02)
POOL_STATE = "public_debt"  # growth noise pools split by a state feature: debt top third vs rest (adopted 2026-11-03,
#                             docs/world_system.md; rejected at the 1.8 growth scale on 2026-11-02)
POOL_CENTER = True        # re-centre the two debt sub-pools on their parent bin mean (adopted after review, 2026-11-03)
COUP_GROWTH = None        # extra coup log-odds per unit of last year's growth (recalibration; rejected 2026-11-03)
MOOD_JUMP = None          # (probability per region-year, size in log-odds): rare regional unrest jumps (rejected 2026-11-03)
YEAR_SHOCK_POW = 0.0      # growth year shocks: mean residual weighted by GDP ** this (0: every country alike). Uncentred,
#                           its gains came from a negative drift that hurt 30-year growth; centred, recessions got rarer
CRISIS_COST = None        # (year after, two years after) output cost of a banking crisis beyond its expected rate: no effect
INFL_CONV = 0.0           # yearly pull of each country's inflation norm toward the cross-country median: no effect
SCIENCE_DECAY = 1.0       # yearly factor on the science-capacity residual (world layer): no effect (with INFL_CONV)
# heat mortality (world.WorldConfig.heat_mortality): deaths per person-year per degree C of local warming at a heat/cold
# factor of 1 and the reference income; calibrated so that SSP5-8.5 reaches Carleton et al.'s 73 per 100,000 in 2100
HEAT_MORT_BETA = 0.553e-4    # (1,000 worlds, SSP5-8.5, 2100: 73 per 100,000 a year; SSP2-4.5 then about 17)
HEAT_MORT_POWER = 1.7
HEAT_MORT_INCOME = (15000.0, 0.4)   # reference GDP per person (2011 $) and elasticity: richer countries adapt
POOL_HALFLIFE = None
# align the onset hazard on the previous decade's onset rate (one log-odds shift for every country at peace). Under test
# 2026-11-03: the shift (+0.3 to +0.6 from 1995) went to the safest countries too, which had almost no new wars
ALIGN_ONSET = True        # (rejected off 2026-11-03: total onsets fell to 0.7-0.8x, world conflict share CRPS +11% / +120%)
OIL_INDEP = True          # oil moves replayed from independent historical years, not the world's replayed decade
                          # (adopted 2026-11-03 with WorldConfig.oil_reversion 0.15: within a decade block one era's
                          # direction repeated ten times; 15-year hindcast oil 5-95% band was 0.15-13x, 6 of 7 outcomes
                          # in its middle half)
# the risk-model ensemble's central view in the simulator (adopted 2026-11-03): each country's onset and coup log-odds
# shifted by the ensemble's (G, G60, N, F, equal-weight probabilities) minus the network's at the origin, held for the
# run; coup alignment 0.7 -> 0.66 with it. 10-year hindcasts, 2 network seeds (sim_eval eoc66 vs ao1): onset log loss
# -4.0/-4.5% (1970-1998), -6.1/-7.4% (2000-2012); coup Brier -5.1/-5.6%, -4.4/-3.5%; coup count CRPS -3.8/-5.7%,
# -8.2/-6.5%; world GDP -0.7/-0.4%, -1.2/-1.1%; country GDP +0.0/+0.1%, +0.3/+0.3%. "o" onset, "t" ending, "c" coups
# Withdrawn 2026-11-04: in the forecast's configuration (with the family draw, world_hindcast --struct familyc+ vs
# familyc+oc) new civil wars +11.6% and conflict share +7.2% on 2000-2009 (0/3), share +4.1% and coups +4.1% at 30
# years; coups only ("c"): -1.2% / -9.3% / +5.6%. Not adopted; "" = the network's view, as before.
ENS_OFFSET = ""
# the ensemble's view refreshed from each world's simulated state every this many years (0: the origin's offset for the
# whole run). Offsets at the origin correlate only ~0.6 / 0.45 / 0.25 with the next origin's 5 / 10 / 20 years later
# (outputs/sim_eval/ens_cache), so a fixed offset goes stale. Needs net.ens_models (pipeline.sim_net); under test 2026-11-04
ENS_REFRESH = 0
ONSET_SLOPE = 1.0         # onset log-odds x this before alignment (1.15-1.5 rejected 2026-11-03: worse on 1970-1998)
# 2026-11-02 tests (off = the adopted model):
VOL_NOISE = False         # B: growth noise scaled by the country's own 10-year growth volatility (slope estimated in fit)
ASYM_SHOCK = None         # C: (boom scale, bust scale) for the replayed world shocks, recentred, in place of shock_scale
DEM_INTER = ()            # D: interaction pairs added to the democracy equation, e.g. (("democracy", "freexp_change_5y"),)
DEMOG_COHORT = False
# sovereign debt and currency crises (macro_crisis.py): drawn each year from hazards on simulated debt, growth and
# inflation, each with its estimated output cost; under test 2026-11-02
MACRO_CRISIS = False      # E: hindcasts age the origin's 0-14 and 15-64 stocks forward instead of extending trends      # noise pools resampled with weight 0.5 ** (years before origin / this): widened the spread
SUPPORT_Q = 0.005          # Transition features are clamped to their 0.5-99.5% training range when simulating (None: +/-6 sd)
WORKFORCE_BETA = 1.22     # growth per unit change in log working-age share (scripts/workforce_test.py; s.e. 0.17)
WORKFORCE_MEAN = 0.0019   # mean yearly change 1960-2023, already absorbed by the fitted growth equation


def working_age_share(p):
    """log share of the population aged 15-64 [C, years] (UN WPP estimates, then the medium projection; OWID)"""
    import pandas as pd
    from .data import RAW
    a = pd.read_csv(RAW / "population-young-working-elderly-with-projections.csv")
    v = {k: a[[c for c in a.columns if k in c]].bfill(axis=1).iloc[:, 0] for k in ("age_0_14", "age_15_64", "age_65plus")}
    a["lw"] = np.log(v["age_15_64"] / (v["age_0_14"] + v["age_15_64"] + v["age_65plus"]))
    s = a.dropna(subset=["code", "lw"]).set_index(["code", "year"]).lw
    yi = {y: j for j, y in enumerate(p.years)}
    out = np.full((len(p.codes), len(p.years)), np.nan)
    ci = {c: i for i, c in enumerate(p.codes)}
    for (c, y), x in s.items():
        if c in ci and y in yi:
            out[ci[c], yi[y]] = x
    return out


def exogenous_demography(p, base, origin, T, mode="un"):
    """population and age structure for origin..origin+T.
    'un': the panel's series (UN WPP estimates, then medium-variant projections) - used for forecasts.
    'extrapolate': damped recent trends only - used for hindcasts, which must not see the realised future."""
    t0 = p.yi(origin)
    keys = ["median_age", "youth", "fertility", "lpop", "lwork"]
    base = {**base, "lwork": working_age_share(p)}
    if mode == "un":
        return {k: base[k][:, t0:t0 + T + 1] for k in keys}
    lim = {"median_age": (14, 55), "youth": (0.1, 0.55), "fertility": (1.0, 8.0), "lpop": (-np.inf, np.inf),
           "lwork": (np.log(0.45), np.log(0.75))}
    out = {}
    for k in keys:
        if k == "lwork" and DEMOG_COHORT:
            continue
        rate = np.nan_to_num(_trend_rate(base[k], t0, 5 if k == "lpop" else 10))
        path = [base[k][:, t0]]
        for j in range(1, T + 1):
            path.append(np.clip(path[-1] + rate * (0.98 if k == "lpop" else 0.95) ** j, *lim[k]))
        out[k] = np.stack(path, 1)
    if DEMOG_COHORT:
        # the working-age share from people already born at the origin: each year a fifteenth of the 0-14 stock
        # enters working age and a fiftieth of the 15-64 stock leaves it (uniform ages within each group), over the
        # total population path; after 15 years the share continues its own last-five-year change, damped
        w0 = np.exp(np.nan_to_num(base["lwork"][:, t0], nan=np.log(0.6)))
        y0 = np.clip(np.nan_to_num(base["youth"][:, t0], nan=0.3), 0.05, 0.6)
        lp = out["lpop"]
        path = [np.log(w0)]
        for j in range(1, T + 1):
            if j <= 15:
                stock = w0 + y0 * j / 15 - w0 * j / 50
                path.append(np.log(np.clip(stock / np.exp(lp[:, j] - lp[:, 0]), 0.45, 0.75)))
            else:
                path.append(np.clip(path[-1] + (path[15] - path[10]) / 5 * 0.95 ** (j - 15), np.log(0.45), np.log(0.75)))
        out["lwork"] = np.stack(path, 1)
    return out


# Burke, Hsiang & Miguel (2015, Nature), pooled model: growth effect of annual mean temperature T
BHM = (0.0127, -0.000487)


def climate_inputs(p, base, origin, T, scenario=None, refresh=False):
    """Climate drivers for origin..origin+T, per variable: smooth paths (median, p10, p90 of the
    CMIP6 ensemble) [C, T+1] and the country's interannual noise sd [C].
    scenario=None (hindcasts): no CMIP6, only the country's own 30-year trend continued."""
    from . import climate
    t0 = p.yi(origin)
    out = {}
    paths = None
    # observed series without the panel's carry-forward (ERA5 to 2025):
    # baselines and noise come from observed years only
    raw = {v: p.x[f"c_{v}"] if f"c_{v}" in p.x else base[v] for v in climate.VARS}
    if scenario is not None:
        paths = {st: climate.scenario_paths(p.codes, p.years, scenario, raw, st, refresh) for st in ("median", "p10", "p90")}
    for v in climate.VARS:
        x = raw[v][:, :t0 + 1]
        lastv = int(np.max(np.where(np.isfinite(x).mean(0) > 0.5)[0]))
        x = x[:, :lastv + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # interannual variability: deviation from the trailing 10-year mean
            roll = np.stack([np.nanmean(x[:, max(0, i - 10):i], 1) for i in range(x.shape[1])], 1)
            sd = np.nanstd((x - roll)[:, -40:], 1)
            last10 = np.nanmean(x[:, -10:], 1)
        if paths is None:
            trend = np.nan_to_num(_trend_rate(x, x.shape[1] - 1, 30)) if v in ("tas", "txx") else np.zeros(len(x))
            med = last10[:, None] + trend[:, None] * (np.arange(T + 1)[None, :] + 4.5 + (t0 - lastv))
            q = {"median": med, "p10": med, "p90": med}
        else:
            # smooth the delta-corrected CMIP6 paths (5-yr centred mean): the noise is added separately
            q = {}
            for st in ("median", "p10", "p90"):
                a = paths[st][v][:, t0 - 2:t0 + T + 3]
                if a.shape[1] < T + 5:                  # runs to the end of the projections: hold the last values
                    a = np.pad(a, ((0, 0), (0, T + 5 - a.shape[1])), mode="edge")
                k = np.ones(5) / 5
                sm = np.stack([np.convolve(np.nan_to_num(r, nan=np.nanmean(r)), k, mode="valid") for r in a])
                ramp = np.clip(np.arange(T + 1) / 5, 0, 1)            # blend into the scenario over 5 years
                q[st] = last10[:, None] * (1 - ramp) + sm * ramp
        out[v] = {"median": q["median"], "p10": q["p10"], "p90": q["p90"], "sd": np.nan_to_num(sd), "base": last10}
    out["tas"]["pw_offset"] = climate.popweight_offset(p.codes, p.years, base["tas"], refresh)
    return out


def _ffill_window(x, limit=3):
    """forward-fill NaNs along the last (time) axis, at most `limit` steps."""
    out = x.clone()
    last = out[..., 0].clone()
    age = torch.zeros_like(last)
    for j in range(1, out.shape[-1]):
        v = out[..., j]
        fin = torch.isfinite(v)
        age = torch.where(fin, torch.zeros_like(age), age + 1)
        out[..., j] = torch.where(~fin & (age <= limit), last, v)
        last = torch.where(fin, v, last)
    return out


@torch.no_grad()
def _ens_refresh(models, Xh, haz_net, haz_offset0):
    """[S, C, 3] the ensemble's log-odds (equal-weight family probabilities, as pipeline.attach_ensemble) minus the
    network's at each world's simulated features, clipped to +/-3, on the hazards in ENS_OFFSET; countries the origin's
    offset left at zero (no data) stay at zero"""
    from .features import TASKS
    idx3 = [TASKS.index(t) for t in ("onset_1y", "termination_1y", "coup_1y")]
    S_, C_, F_ = Xh.shape
    Xf_ = Xh.reshape(-1, F_)
    pe = np.empty((S_ * C_, 3))
    for a in range(0, S_ * C_, 200000):                       # chunked to bound host memory
        Xn = Xf_[a:a + 200000].detach().cpu().numpy().astype(np.float32)
        pe[a:a + 200000] = np.mean([np.mean([m.predict(Xn)[:, idx3] for m in ms], 0) for ms in models.values()], 0)
    pe = np.clip(pe, 1e-4, 1 - 1e-4).reshape(S_, C_, 3)
    off = torch.as_tensor(np.log(pe / (1 - pe)), dtype=torch.float32, device=haz_net.device) - haz_net
    off = off.clamp(-3.0, 3.0)
    keep = torch.as_tensor(np.asarray(haz_offset0) != 0, device=haz_net.device)[None]
    return torch.nan_to_num(torch.where(keep, off, torch.zeros_like(off)))


@torch.no_grad()
def simulate(p, base, net: PsychoNet, growth: Transition, ddem: Transition, origin: int, end: int,
             n_scen=4000, seed=0, device=DEVICE, record_paths=8, chronicle=0, mood_sd=(0.10, 0.10), mood_rho=0.9,
             regime_years=10, growth_regime_scale=None, demography="un", clim=None, climate_damage=0.0,
             align=True, world=None, dyads=None, slow_dynamics=True, migration=True, damage_feedback=True,
             align_window=10, align_longrun_w=0.0, year_shock_scale=0.7, align_longrun_targets="otc",
             dem_regime_scale=None, dem_shock_scale=1.0, frailty_sd=0.6, frailty_end=1.0, coup_frailty_sd=0.0, coup_align_window=None, coup_align_mult=None,
             haz_offset=None):
    """`growth_regime_scale` and `mood_sd` are dispersion settings chosen with the hindcasts
    (see README: their coverage is therefore mildly optimistic). mood_sd was lowered from 0.20 to 0.10 on 2026-10-27:
    with 0.20 the world conflict level was over-dispersed (scripts/conflict_clustering_check.py, dev origins 1970-1998:
    CRPS -7.6%, better at 11 of 15 origins; realised value in the middle half of the forecast 53% instead of 67%).
    `clim`: output of climate_inputs (scenario paths); `climate_damage`: multiplier on the
    Burke-Hsiang-Miguel temperature-growth effect of warming beyond the origin (0 = off).
    `align`: shift each hazard's log-odds by a constant so that, at the origin, the average
    predicted onset / termination / coup-attempt rate equals the observed rate of the previous ten
    years (standard microsimulation alignment; uses only pre-origin data).
    `slow_dynamics`: debt, polarization, mobilization, political violence, inequality, poverty and military spending
    follow equations estimated on history (psychohistory/dynamics.py) instead of staying at their origin values.
    `migration`: refugees flee simulated wars and move population to host countries (psychohistory/migration.py).
    `haz_offset`: optional [C, 3] log-odds added to each country's onset / termination / coup hazards for the whole run
    (the risk-model ensemble's view at the origin minus the network's). None: `net.haz_offset` if the caller attached one
    (pipeline.attach_ensemble), restricted to the hazards in ENS_OFFSET.
    `coup_align_mult`: None = 0.66 with the ensemble's coup offset, 0.7 without (both chosen on 1970-1998 hindcasts)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    C = len(p.codes)
    if haz_offset is None and getattr(net, "haz_offset", None) is not None:
        haz_offset = np.array(net.haz_offset, float)
        for j_, t_ in enumerate("otc"):
            if t_ not in ENS_OFFSET:
                haz_offset[:, j_] = 0.0
    if coup_align_mult is None:
        coup_align_mult = 0.66 if haz_offset is not None and bool(np.any(np.asarray(haz_offset)[:, 2] != 0)) else 0.7
    haz_off_t = None if haz_offset is None else torch.as_tensor(np.nan_to_num(np.asarray(haz_offset, float)), dtype=torch.float32, device=device)
    S = n_scen
    t0 = p.yi(origin)
    T = end - origin
    growth.to(device), ddem.to(device)
    # scale of the replayed global year shock (growth only): 0.7 since 2026-10-27, world GDP was over-dispersed
    # (scripts/sim_eval.py: world GDP CRPS -2.2% on dev origins 1970-1998, -3.1% on test origins 2000-2012)
    growth.shock_scale = year_shock_scale
    ddem.shock_scale = dem_shock_scale
    tt = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    regions = torch.as_tensor(p.regions, device=device)
    R = len(p.region_names)
    onehot = torch.nn.functional.one_hot(regions, R).float()

    # --- initial state ---------------------------------------------------------------
    h = {k: tt(base[k][:, t0 - W + 1:t0 + 1])[None].repeat(S, 1, 1) for k in WINDOW_VARS}
    for k in WINDOW_VARS:
        if k != "conflict":
            h[k] = _ffill_window(h[k])          # bridge 1-3 year reporting lags at the origin
    h["conflict"] = torch.nan_to_num(h["conflict"])
    cur = {k: tt(base[k][:, t0])[None].repeat(S, 1) for k in CURRENT_VARS}
    # trends (country-level, deterministic)
    urban_rate = tt(_logistic_rate(base["urban"], t0))
    cm_rate = tt(np.clip(np.nan_to_num(_trend_rate(base["log_cm"], t0)), -0.1, 0.0))
    le_rate = tt(np.clip(np.nan_to_num(_trend_rate(base["life_exp"], t0)), -0.2, 0.5))
    tert_rate = tt(np.clip(np.nan_to_num(_trend_rate(base["tertiary_enrol"], t0)), -1, 2))
    if clim is None:
        clim = climate_inputs(p, base, origin, T)
    cl = {v: {k: tt(a) for k, a in d.items()} for v, d in clim.items()}
    # one climate-sensitivity draw per world, shared by all countries and variables
    z_clim = torch.randn(S, 1, device=device, generator=gen)
    t_off = cl["tas"].pop("pw_offset")        # damage is evaluated at population-weighted temperature
    t_base = cl["tas"]["base"] + t_off
    # heat/cold balance of each country's climate: + where it is hot (more heat deaths), - where it is cold (fewer cold
    # deaths), from its population-weighted baseline temperature
    heat_h = ((t_base - 16.0) / 8.0).clamp(-1.5, 1.5).nan_to_num(0.0)
    tas_0 = cl["tas"]["median"][:, 0]
    dmg_cum = torch.zeros(S, C, device=device)
    # world-system layer (interstate wars, food, energy, pandemics, AI); None = the plain simulator
    WS = None
    if world is not None:
        from .world import World
        WS = World(world, p, base, origin, S, device, gen, dyads)
    # climate (2026-10-29): a heavier-tailed climate-response draw (Student t, scaled to keep the CMIP6 10-90% range),
    # the damage family's state, and global-mean warming for the global-temperature family
    clim_fam = None
    if WS is not None and WS.cfg.sample_params:
        if WS.cfg.clim_tail_df:
            from scipy.stats import t as t_dist, norm as n_dist
            df_ = WS.cfg.clim_tail_df
            g_t = torch.Generator(device=device).manual_seed(seed + 503)
            chi = (torch.randn(S, int(df_), device=device, generator=g_t) ** 2).mean(-1, keepdim=True)
            z_clim = (z_clim / chi.sqrt() * float(n_dist.ppf(0.9) / t_dist.ppf(0.9, df_))).clamp(-4, 4)
        clim_fam = WS.params["dmg_family"] if "dmg_family" in WS.params else torch.full((S, 1), float(WS.cfg.dmg_family), device=device)
        WS.params["clim_response"] = z_clim          # the world's climate-response draw (for the attribution)
    # demographic uncertainty (UN WPP 2024 probabilistic projections, scripts/build_wpp_prob.py): one persistent fertility
    # error per world and country, scaled to the UN's 80% interval for that year, moves births through fert_ratio below
    dem_sig = None
    if WS is not None and WS.cfg.demog_uncertainty and origin >= 2023:
        import os as _os
        _f = _os.path.join(_os.path.dirname(__file__), "..", "data", "wpp_prob.npz")
        if _os.path.exists(_f):
            wz = np.load(_f)
            if list(wz["codes"]) == list(p.codes):
                dem_sig, dem_y0 = tt(np.nan_to_num(wz["sig_tfr"])), int(wz["years"][0])
                g_d = torch.Generator(device=device).manual_seed(seed + 611)
                rho_d = float(wz["rho"])
                z_dem = np.sqrt(rho_d) * torch.randn(S, 1, device=device, generator=g_d) \
                    + np.sqrt(1 - rho_d) * torch.randn(S, C, device=device, generator=g_d)
                # (and the rest of the UN population interval, migration and mortality: a level offset of its own)
                dem_rest, z_rest = tt(np.nan_to_num(wz["sig_rest"])), torch.randn(S, C, device=device, generator=g_d)
                rest_prev = torch.zeros(S, C, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pw_ = np.exp(np.nan_to_num(base["lpop"][:, t0], nan=-np.inf)) * np.isfinite(clim["tas"]["base"])
    pw = tt(pw_ / pw_.sum())
    bhm_cum = torch.zeros(S, C, device=device)
    clim_prev = torch.zeros(S, C, device=device)
    amoc_prev = torch.zeros(S, C, device=device)
    dTg_lag = torch.zeros(S, 1, device=device)
    nuc_cum = torch.zeros(S, C, device=device)          # largest output share lost to nuclear use so far
    clim_ly = torch.zeros(S, C, device=device)          # cumulative climate-driven log harvest change since the origin
    top1_0 = cur["top1"].clone()
    slow = None
    if slow_dynamics:
        from .dynamics import SlowState
        slow = SlowState(_slow_equations(base, p.years, origin), base, t0, cur, device, gen)
        debt_hist = _ffill_window(tt(base["public_debt"][:, t0 - 5:t0 + 1])[None].repeat(S, 1, 1))
    # party systems change at elections (dynamics.fit_party): reversion toward the country's 20-year mean plus a shock
    party = {}
    if slow_dynamics:
        from .dynamics import fit_party, ANCHOR_YEARS
        for k_, e_ in fit_party(p, base, origin).items():
            if k_ in cur:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    anc_ = np.nanmean(base[k_][:, max(0, t0 - ANCHOR_YEARS):t0 + 1], 1)
                party[k_] = (e_, tt(anc_))         # (no V-Party history: stays missing)
    ai_top1_prev = 0.0
    # missing factors (risks.py): internationalized civil wars, correlated weather, population feedbacks, coasts,
    # tail catastrophes
    from . import risks as R_
    wfac = {v: (tt(F_), tt(u_)) for v, (F_, u_) in R_.weather_factors(p, origin).items()}
    intl = torch.zeros(S, C, device=device)
    if WS is not None:
        ix_ = [i for i, c in enumerate(p.codes) if c in R_.intl_at(origin, tuple(p.codes))]
        if ix_:
            intl[:, ix_] = 1.0
    an_prev = {"txx": torch.zeros(S, C, device=device), "pr": torch.zeros(S, C, device=device)}
    pop_off = torch.zeros(S, C, device=device)          # log population vs the UN path (deaths, births)
    youth_off = torch.zeros(S, C, device=device)
    excess_deaths = torch.zeros(S, device=device)
    coast_prev = torch.zeros(S, C, device=device)
    volc_t = torch.full((S,), -99, device=device)
    pand_mort = torch.zeros(S, device=device)
    trans_prev = torch.zeros(S, C, device=device)
    struct_sd = struct_shift = None
    struct_var = None
    if WS is not None and WS.cfg.sample_params and STRUCT_MODE in ("family", "familyc", "ens_onset") and getattr(net, "struct_fam", None) is not None:
        # each world and hazard takes one family's view of every country (within the families' range by construction);
        # "familyc": the offsets centred on their weighted mean, so the draw spreads risk without moving its level;
        # "ens_onset": centred for war ending and coups, uncentred for war onset (the onset level becomes the
        # ensemble's, which puts the safest countries lower than the neural network alone)
        fam_ = np.clip(np.nan_to_num(net.struct_fam), -3.0, 3.0)
        w_ = np.asarray(net.struct_w, float)
        mean_ = np.tensordot(w_, fam_, axes=1)[None]                                      # [1, C, 3]
        cen_ = fam_ - mean_
        # convexity: a log-odds spread that averages zero still raises a low probability on average (Jensen); the
        # same correction as the Gaussian draw, with each country's own spread between the families (review 2026-11-02)
        struct_var = tt(np.tensordot(w_, cen_ ** 2, axes=1))                              # [C, 3]
        if STRUCT_MODE == "familyc":
            fam_ = cen_
        elif STRUCT_MODE == "ens_onset":
            fam_ = cen_.copy()
            fam_[..., 0] = fam_[..., 0] + mean_[..., 0]
        sf = tt(fam_)                                                                     # [F, C, 3]
        g_s = torch.Generator(device=device).manual_seed(seed + 4242)
        fi = torch.multinomial(tt(net.struct_w).expand(S * 3, -1), 1, replacement=True, generator=g_s).view(S, 3)
        struct_shift = torch.stack([sf[fi[:, j], :, j] for j in range(3)], -1)        # [S, C, 3]
    elif WS is not None and WS.cfg.sample_params and STRUCT_MODE == "gauss" and getattr(net, "struct_sd", None) is not None:
        struct_sd = tt(np.clip(net.struct_sd, 0, 1.5))
        g_s = torch.Generator(device=device).manual_seed(seed + 4242)
        z_struct = torch.randn(S, 3, device=device, generator=g_s)
    lgh_ = base["lg"][:, max(0, t0 - 20):t0 + 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g_ref = tt(np.clip(np.nan_to_num((lgh_[:, -1] - lgh_[:, 0]) / max(lgh_.shape[1] - 1, 1), nan=0.015), -0.01, 0.04))
    lg_ref0 = tt(np.nan_to_num(base["lg"][:, t0]))
    storm_expo_ = np.nan_to_num(base["near_coast"][:, t0]) / 100 * (0.5 + 0.5 * np.nan_to_num(base["tropical"][:, t0]) / 100)
    # measured where available: the average share of land under hurricane-force winds per year, IBTrACS 1980-2024
    # (scripts/build_cyclone_exposure.py), rescaled to the proxy's mean across countries
    import os as _os
    _f = _os.path.join(_os.path.dirname(__file__), "..", "data", "cyclone_exposure.npz")
    if _os.path.exists(_f):
        _z = np.load(_f)
        if list(_z["codes"]) == list(p.codes):
            sh_ = _z["area_share"]
            ok_ = np.isfinite(sh_)
            storm_expo_ = np.where(ok_, np.nan_to_num(sh_) * storm_expo_[ok_].mean() / max(sh_[ok_].mean(), 1e-9), storm_expo_)
    storm_expo = tt(storm_expo_)
    mig = None
    if migration:
        from .migration import fit_refugees, host_shares
        # (estimation from 1975, or the 15 years before an earlier hindcast origin: from 1975 alone, origins before
        # 1980 gave no country effects and NaN refugee shares; code review 2026-10-28)
        mig = {"eq": fit_refugees(base, np.asarray(p.years), origin, start=min(1975, origin - 15), codes=p.codes,
                                  expeditionary=WS.cfg.expeditionary if WS is not None else ("USA", "GBR", "FRA", "RUS", "AUS", "CAN", "CHN")),
               "H": tt(host_shares(p.codes, p.regions, origin))}
        mig["stock0"] = torch.nan_to_num(cur["refugees_pc"] * torch.exp(h["lpop"][..., -1]))
        mig["moved"] = torch.zeros(S, C, device=device)              # net people moved since the origin (+ = arrived)
    fd = None
    if WS is not None and WS.cfg.food and "feat__cereal_area_ha" in p.x:
        from .food import fit_calories, fit_demand, fit_world_price, observed_food
        wp = fit_world_price(origin)
        # import dependency: the last food-balance value within 15 years (several countries left the FAO balances
        # after 2013), else the median of the country's region (a missing value is not "self-sufficient")
        dh = p.x["feat__cereal_import_dep"][:, max(t0 - 15, 0):t0 + 1]
        last = np.array([r[np.isfinite(r)][-1] if np.isfinite(r).any() else np.nan for r in dh])
        reg = np.asarray(p.regions)
        for r in np.unique(reg):
            m = (reg == r) & ~np.isfinite(last)
            if m.any():
                last[m] = np.nanmedian(last[(reg == r) & np.isfinite(last)]) if np.isfinite(last[reg == r]).any() else np.nanmedian(last)
        dep0 = np.clip(np.nan_to_num(last, nan=0.25), -4, 1)
        # cereal area: start from the median of the last five observed years (one drought year must not set the
        # level) and keep its 15-year Theil-Sen trend (0 to +3% a year, damped like yields; a falling trend is not
        # extended) until it reaches 90% of the arable land; where it already uses that much (or multi-crops) it
        # cannot grow
        with np.errstate(all="ignore"):
            ah = p.x["feat__cereal_area_ha"][:, t0 - 15:t0]
            a_hist = np.log(np.where(ah > 0, ah, np.nan))
            area0 = np.nan_to_num(np.exp(np.nanmedian(a_hist[:, -5:], 1)))
            i, j = np.triu_indices(a_hist.shape[1], 1)
            a_rate = np.nan_to_num(np.clip(np.nanmedian((a_hist[:, j] - a_hist[:, i]) / (j - i), 1), 0.0, 0.03))
            arable = np.nan_to_num(p.x["feat__arable_ha"][:, t0])
            a_cap = np.maximum(area0, 0.9 * arable)
        area = tt(area0)
        # grain-price residuals for scenario years without one (before 1970, after the data): a fixed draw from the
        # observed residuals, so those years keep the historical volatility
        rv = np.array(list(wp["resid"].values()))
        rr = np.random.default_rng(origin)
        # (an origin before any observed residual, e.g. 1975 hindcasts, replays none: 0)
        res = [wp["resid"].get(int(y) + 1, rv[rr.integers(len(rv))] if len(rv) else 0.0) for y in growth.shock_years]
        fd = {"wp": wp, "kc": fit_calories(origin), "dm": fit_demand(origin), "obs": observed_food(origin), "area": area,
              "a_rate": tt(a_rate), "a_cap": tt(a_cap), "res": tt(res),
              "lp": torch.full((S, 1), wp["p0"], device=device), "s": tt(1 - dep0)[None].repeat(S, 1),
              "wprod": (torch.nan_to_num(cur["cereal_yield"]) * area).sum(-1, keepdim=True),
              "wpop": float(np.nansum(np.exp(base["lpop"][:, t0]))), "sup_dev": torch.zeros(S, 1, device=device)}
        # historical harvest shortfalls by replayed year (year y stands for y -> y+1, so the y+1 harvest); scenario
        # years without one get a fixed draw from the observed years
        from .food import yield_residuals
        yr_ = yield_residuals(p, origin)
        has = np.isfinite(yr_).sum(0) >= 50
        cols = [p.yi(int(y) + 1) if (int(y) + 1) in p.years and has[p.yi(int(y) + 1)] else None for y in growth.shock_years]
        pool = np.where(has)[0]
        rr2 = np.random.default_rng(origin + 17)
        cols = [c if c is not None else int(pool[rr2.integers(len(pool))]) for c in cols]
        fd["yres"] = tt(np.nan_to_num(yr_[:, cols].T))                      # [n shock years, C]
        fd["r_prev"] = torch.zeros(S, C, device=device)
        fd["amp_prev"] = torch.zeros(S, C, device=device)
        idx_ = {c: i for i, c in enumerate(p.codes)}
        fd["i_us"] = idx_.get("USA")
        fd["i_x4"] = [idx_[c] for c in ("USA", "BRA", "ARG", "UKR") if c in idx_]     # top maize exporters
    prev_tas = cl["tas"]["median"][:, 0] + 0 * z_clim
    infl_target = tt(_nanmedian_recent(base["inflation"], t0))   # the country's own 20-year norm
    infl_med = float(torch.nanmedian(infl_target))
    unemp_target = tt(_nanmedian_recent(base["unemployment"], t0))
    cm_trend, le_trend = cur["log_cm"].clone(), cur["life_exp"].clone()
    slopes = comove_slopes(base, origin, p.years)
    oil = base["oil_price"][0]
    # the replayed shock year y stands for the transition y -> y+1 (fit_transitions labels growth and democracy
    # shocks by their start year), so oil and grain-price moves are taken over the same transition
    oil_change_t = tt(np.nan_to_num(np.array([oil[p.yi(y) + 1] - oil[p.yi(y)] for y in growth.shock_years])))
    if OIL_RESID and world is not None:
        # (replay what is left of each historical move after its pull toward the 20-year average and any war entry by a
        #  major exporter, estimated on 1955..origin; the simulator adds its own pull and war spikes, which replaying
        #  the raw moves of 1974, 1979 or 1990 counted twice - independent review, 2026-11-02)
        _ex = [p.codes.index(c) for c in OIL_EXPORTERS if c in p.codes]
        _it = np.nan_to_num(base["interstate"][_ex])
        _new = np.r_[False, ((_it[:, 1:] > 0) & (_it[:, :-1] == 0)).any(0)]
        _rows = {}
        for j in range(len(p.years) - 1):
            y_ = int(p.years[j])
            if 1955 <= y_ < origin and np.isfinite(oil[j + 1]) and np.isfinite(oil[j]):
                _rows[y_] = (oil[j + 1] - oil[j], oil[j] - np.nanmean(oil[max(0, j - 19):j + 1]), float(_new[j + 1]), float(_new[j]))
        if len(_rows) > 15:
            _R = np.array(list(_rows.values()))
            _D = np.c_[np.ones(len(_R)), _R[:, 1:]]
            _b = np.linalg.lstsq(_D, _R[:, 0], rcond=None)[0]
            _res = dict(zip(_rows, _R[:, 0] - _D[:, 1:] @ _b[1:]))       # (the constant stays: the trend is removed later)
            oil_change_t = tt(np.array([_res.get(int(y), float(oil_change_t[i])) for i, y in enumerate(growth.shock_years)]))
    # oil-production exposure (log1p of MWh per person / 10, as in the growth regression) and the major exporters
    oil_expo = tt(np.nan_to_num(np.log1p(base["oil_pc"][:, t0] / 1000) / 10))
    oil_trend = float(np.nanmean(oil_change_t.cpu().numpy()))
    # (from 1995; origins before 2014 use the 20 years to the origin, so pre-1995 hindcasts have an anchor)
    oil_anchor = float(np.nanmean(oil[min(p.yi(1995), p.yi(origin) - 19):p.yi(origin) + 1]))
    oil_spike = torch.zeros(S, 1, device=device)
    site_hist = [torch.zeros(S, C, device=device), torch.zeros(S, C, device=device)]   # war sites 2 and 1 years ago
    war_cum = torch.zeros(S, C, device=device)
    bubble_t = torch.full((S,), -1, device=device, dtype=torch.long)       # year of the AI investment bust (-1: none yet)
    fin_extra = torch.zeros(S, C, device=device)
    fin_extra_prev = torch.zeros(S, C, device=device)
    mc_models, mc_prev = None, {}
    if MACRO_CRISIS and "public_debt" in cur:
        from . import macro_crisis as MC
        mc_models = MC.fit(origin, tuple(p.codes), tuple(int(y) for y in p.years))
        mc_prev = {k: torch.zeros(S, C, device=device) for k in mc_models}
        g_mc = torch.Generator(device=device).manual_seed(seed + 1301)       # (its own draws: other streams unchanged)
        mc_rec = {k: [] for k in mc_models}
    crisis_x = torch.zeros(S, C, device=device)          # banking crises beyond their expected rate, last two years
    crisis_x_prev = torch.zeros(S, C, device=device)
    spill_hist = [torch.zeros(S, C, device=device), torch.zeros(S, C, device=device)]
    lw_hist = [h["lg"][..., -1].clone()]          # GDP with war damage, origin on (last 5 years kept)
    if world is not None and getattr(world, "transition", False):
        tg = torch.Generator(device=device).manual_seed(seed + 7)
        lo_K, hi_K = world.trans_ceiling
        tr_K = lo_K + (hi_K - lo_K) * torch.rand(S, 1, device=device, generator=tg)
        tr_rate = (world.trans_rate[0] + world.trans_rate[1] * torch.randn(S, 1, device=device, generator=tg)).clamp(0.05, 0.3)
        # share at the origin: the fitted S-curve run back from 2024 (wind+solar were ~0 before 2000)
        s0 = world.trans_share_0 * float(np.exp(world.trans_rate[0] * (origin - 2024)))   # 6.4% in 2024
        tr_s0 = torch.full((S, 1), s0, device=device)
        tr_s = tr_s0.clone()
    oil_exporters = torch.as_tensor([p.codes.index(c) for c in OIL_EXPORTERS if c in p.codes], device=device)
    hist_ok = (p.years[None, :] <= origin) & (p.years[None, :] >= 1950)
    # share of coup attempts that succeed (Powell & Thyne), up to the origin
    ca, cs = base["coup_attempts"], base.get("coups_success")
    p_coup_success = float(np.nansum(np.where(hist_ok, cs, 0)) / max(np.nansum(np.where(hist_ok, ca > 0, 0)), 1)) \
        if cs is not None else 0.5
    p_coup_success = min(max(p_coup_success, 0.2), 0.8)
    # leader turnover other than by coup: historical annual rate (V-Dem leader names), by democracy decile
    has_leaders = "leader_log_tenure" in base and "leader_change" in base
    if has_leaders:
        lc, dem_b = base["leader_change"], base["dem"]
        coup_year = np.nan_to_num(cs if cs is not None else ca) > 0
        okc = np.isfinite(lc) & np.isfinite(dem_b) & hist_ok & ~coup_year
        edges_d = np.linspace(0, 1, 11)
        rate = []
        for i in range(10):
            sel = okc & (dem_b >= edges_d[i]) & (dem_b < edges_d[i + 1] + (i == 9))
            rate.append(float(lc[sel].mean()) if sel.sum() > 30 else 0.15)
        turnover = tt(rate)
    # elections: only countries that held one in the 15 years before the origin keep holding them,
    # at their usual interval (median of past gaps; 5 where unknown)
    gaps, holds = [], []
    for r in base["election_year"][:, :t0 + 1]:
        yrs = np.where(r == 1)[0]
        gaps.append(np.median(np.diff(yrs[-8:])) if len(yrs) >= 3 else 5.0)
        holds.append(len(yrs) > 0 and yrs[-1] >= t0 - 15)
    elec_interval = tt(np.clip(gaps, 2, 8))
    holds_elections = torch.as_tensor(holds, device=device)[None].repeat(S, 1)      # per world: regimes change
    bounds = {k: (float(np.nanmin(base[k])), float(np.nanmax(base[k]))) for k in COMOVE + ["admin_impartial"]}
    # 5-year level history of the institutions whose change is a feature
    inst_hist = {lvl: _ffill_window(tt(base[lvl][:, t0 - 5:t0 + 1])[None].repeat(S, 1, 1))
                 for lvl in INST_CHANGE.values()}
    exo = {k: tt(v) for k, v in exogenous_demography(p, base, origin, T, demography).items()}
    sy = list(growth.shock_years)
    # (grid year y holds the shock of the y -> y+1 change: 2019 = the 2020 COVID crash, 2020 = the 2021 rebound;
    # fixed 2026-10-28, the code had replaced the rebound by a second crash)
    i_covid = sy.index(2019) if {2018, 2019, 2020, 2021} <= set(sy) else None
    # coup attempts in the origin year (peer contagion in year 1 sees them, as it sees the simulated ones later)
    coup_last = torch.as_tensor(np.nan_to_num(base["coup_attempts"][:, t0]) > 0, device=device, dtype=torch.float32)[None].repeat(S, 1) \
        if SHIFTERS_BEFORE_ALIGN else torch.zeros(S, len(p.codes), device=device)
    lreb = WS.lreb0[None].repeat(S, 1) if WS is not None else None
    tot_prev = torch.zeros(S, len(p.codes), device=device)          # export-weighted commodity price change, last year
    food_dlp = torch.zeros(S, 1, device=device)
    succ_t = torch.zeros(S, len(p.codes), device=device)            # years left of a post-death succession window
    fish_g = torch.zeros(S, len(p.codes), device=device)            # income change from wild catch (applied next year)
    fish_log = torch.zeros(S, device=device)                        # world wild catch, cumulative log change
    pop = torch.exp(exo["lpop"])

    # each scenario is driven by one ensemble member
    member = torch.arange(S, device=device) % len(net.nets)
    # "world mood": stationary AR(1) shifts of the hazard log-odds, global and regional.
    # They let waves of instability (decolonisation, end of the Cold War, 2014+) emerge that
    # no country-level variable foresees. A convexity correction keeps the mean hazard unchanged.
    mood_world = torch.zeros(S, 1, device=device)
    mood_region = torch.zeros(S, R, device=device)
    mood_coup = torch.zeros(S, R, device=device)
    n_jump = 0
    ever = {k: torch.zeros(S, C, device=device) for k in ("onset", "collapse", "autoc", "coup")}
    p_crisis_low, p_crisis_high = crisis_given_growth(base, origin, p.years)
    # a collapse already under way at the origin is already counted in collapses_total (same rule, features.py)
    was_collapse = (h["lg"][..., -1] - torch.nan_to_num(h["lg"][..., -6:-1], nan=-float("inf")).amax(-1) < np.log(0.9)).float()
    lg0 = h["lg"][..., -1].clone()

    rec = {"conflict": [], "ever_onset": [], "ever_collapse": [], "ever_autoc": [], "gdp_q": [], "dem_q": [],
           "world_conflict": [], "world_gdp": [], "world_dem": [], "onsets": [], "region_conflict": [],
           "ever_coup": [], "coups": [], **({"displaced": []} if migration else {})}
    paths = {"gdp": [], "dem": [], "conflict": [], "coup": [], "pop": [], "war_dmg": [], "clim_dmg": []}   # (pop, damages: scripts/disaster_rca.py)
    sel = slice(0, record_paths)
    chron = []
    var_w = var_r = 0.0
    # country frailty (frailty_sd > 0): a persistent per-world, per-country log-odds shift, + on onsets, - on endings,
    # so war-prone countries stay war-prone (less churn than independent yearly draws)
    frail = frailty_sd * torch.randn(S, C, device=device, generator=gen) if frailty_sd else 0.0
    frail_c = coup_frailty_sd * torch.randn(S, C, device=device, generator=gen) if coup_frailty_sd else 0.0

    # observed rates over the `align_window` years before the origin (alignment targets), optionally blended with the
    # long-run rate since 1946 (weight align_longrun_w): a turbulent recent decade need not persist
    def _rates(n):
        cf = base["conflict"][:, t0 - n:t0 + 1]
        on_ev = (cf[:, 1:] == 1) & (cf[:, :-1] == 0)
        te_ev = (cf[:, 1:] == 0) & (cf[:, :-1] == 1)
        ca = base["coup_attempts"][:, t0 - n + 1:t0 + 1]
        return (float(on_ev.sum() / max((cf[:, :-1] == 0).sum(), 1)), float(te_ev.sum() / max((cf[:, :-1] == 1).sum(), 1)),
                float(np.nanmean(np.where(np.isfinite(ca), ca > 0, np.nan))) if np.isfinite(ca).any() else None)
    r_on, r_te, r_cp = _rates(align_window)
    if not ALIGN_ONSET:
        r_on = None
    # coups have declined almost every decade since the 1960s: the coup target can use its own (shorter) window and a
    # multiplier for that decline
    if coup_align_window and coup_align_window != align_window:
        r_cp = _rates(coup_align_window)[2]
    if align_longrun_w:
        lr = _rates(origin - 1946)
        if "o" in align_longrun_targets:
            r_on = (1 - align_longrun_w) * r_on + align_longrun_w * lr[0]
        if "t" in align_longrun_targets:
            r_te = (1 - align_longrun_w) * r_te + align_longrun_w * lr[1]
        if "c" in align_longrun_targets and r_cp is not None and lr[2] is not None:
            r_cp = (1 - align_longrun_w) * r_cp + align_longrun_w * lr[2]
    if r_cp is not None:
        r_cp = r_cp * coup_align_mult
    shift = torch.zeros(3, device=device)
    # war and climate damage are level factors kept outside the growth dynamics (lg); the political hazards were
    # trained on recorded GDP, which contains such damage, so they see GDP with it (damage_feedback)
    dmg_hist = torch.zeros_like(h["lg"])

    for k in range(1, T + 1):
        if WS is not None and WS.tech is not None:
            WS.tech.set_year(origin + k)               # technology adoption this year (tech.py)
        X = compute_features(h, cur, regions, R)                  # [S, C, F]
        Xh = compute_features({**h, "lg": h["lg"] + dmg_hist}, cur, regions, R) \
            if damage_feedback and bool((dmg_hist != 0).any()) else X
        haz = torch.empty(S, C, 3, device=device)
        for a in range(0, S, 2000):                               # chunked to bound GPU memory
            Z = net.pre.transform_t(Xh[a:a + 2000])
            for m, snaps in enumerate(net.nets):
                i = member[a:a + 2000] == m
                haz[a:a + 2000][i] = torch.stack([snaps[j](Z[i])[..., j] for j in (ON, TERM, COUP)], -1)
        del Z
        if haz_offset is not None:
            if ENS_REFRESH and k > 1 and (k - 1) % ENS_REFRESH == 0 and getattr(net, "ens_models", None):
                haz_off_t = _ens_refresh(net.ens_models, Xh, haz, haz_offset)          # [S, C, 3] from now on
            haz = haz + (haz_off_t if haz_off_t.dim() == 3 else haz_off_t[None])
        Xh_d = Xh if Xh is not X else None          # kept for the democracy step below
        mood_world = mood_rho * mood_world + mood_sd[0] * torch.randn(S, 1, device=device, generator=gen)
        mood_region = mood_rho * mood_region + mood_sd[1] * torch.randn(S, R, device=device, generator=gen)
        mood_coup = mood_rho * mood_coup + mood_sd[1] * torch.randn(S, R, device=device, generator=gen)
        var_w = mood_rho ** 2 * var_w + mood_sd[0] ** 2
        var_r = mood_rho ** 2 * var_r + mood_sd[1] ** 2
        if MOOD_JUMP:
            # belief cascades (1989, the Arab Spring): a rare regional jump in unrest that raises both war and coup
            # hazards and fades with the mood; centred so the average hazard is unchanged
            pj, sj = MOOD_JUMP
            jump = sj * ((torch.rand(S, R, device=device, generator=gen) < pj).float() - pj)
            mood_region = mood_region + jump
            mood_coup = mood_coup + jump
            # the jump is skewed, so the Gaussian variance correction is wrong for it: the accumulated jump state
            # J = sum rho^k j_k has log E[exp(c J)] = sum_k K(c rho^k), K(c) = log(1 - p + p e^(c s)) - c p s,
            # taken at c = +1 (onsets, coups) and c = -1 (endings, where the mood enters with a minus sign)
            n_jump += 1
            K = lambda c: math.log(1 - pj + pj * math.exp(c * sj)) - c * pj * sj
            jcorr = (sum(K(mood_rho ** i) for i in range(n_jump)), sum(K(-mood_rho ** i) for i in range(n_jump)))
        mood = mood_world + mood_region[:, regions]
        if struct_shift is not None:
            haz = haz + struct_shift - STRUCT_CONVEXITY * 0.5 * struct_var[None] * (1 - 2 * torch.sigmoid(haz))
        if struct_sd is not None:
            # structural uncertainty: one error per world and hazard, scaled by the model families' disagreement
            # (with the convexity correction used for mood and frailty, so the average hazard is kept; review 2026-10-28)
            haz = haz + z_struct[:, None, :] * struct_sd[None] - STRUCT_CONVEXITY * 0.5 * struct_sd[None] ** 2 * (1 - 2 * torch.sigmoid(haz))
        if WS is not None:
            # civil wars with outside states fighting in them end less often (UCDP 1946-2023: -0.42 log-odds); the
            # country models already see the average war (23% internationalized), so the term is centred on it.
            # (before the year-1 alignment, which then keeps the overall termination rate on target)
            haz[..., 1] = haz[..., 1] + WS.prm("intl_beta") * (intl - R_.INTL_SHARE)
        def _shifters(haz):
            """succession, peacekeeping, export-price, fragmentation and peer-coup terms"""
            if has_leaders:
                haz[..., 2] = haz[..., 2] + SUCC_COUP * (succ_t > 0).float()        # succession after an autocrat's death
            if WS is not None:
                # peacekeepers already deployed (cut in 2025, winding down) help wars end; falling export prices
                # (last year's simulated oil and food prices x export shares) raise coup attempts (intervention.py)
                pko = WS.pko_n[None] * (1 - WS.pko_cut_on * WS.prm("pko_cut")) * torch.exp(torch.as_tensor(-WS.prm("pko_decay") * (k - 1), device=device, dtype=torch.float32))
                haz[..., 1] = haz[..., 1] + WS.prm("pko_beta") * torch.log1p(pko / 1000)
                haz[..., 2] = haz[..., 2] + WS.prm("commod_coup") * tot_prev
                # wars with more rebel groups end later; coups spread among language / colonial peers (structure.py)
                haz[..., 1] = haz[..., 1] + WS.prm("frag_beta") * (lreb - FRAG_MEAN)
                haz[..., 2] = haz[..., 2] + WS.prm("peer_coup") * (coup_last @ WS.peer_w.T)
            return haz
        # (before the year-1 alignment, like intl_beta, so the alignment absorbs these terms' level at the origin and
        # only their changes move the rates; until 2026-10-28 they were added after it, on top of the aligned rates:
        # SHIFTERS_BEFORE_ALIGN = False restores that)
        if COUP_GROWTH:
            # coup recalibration on last year's growth (scripts/coup_slump_test.py): the network's coup risk rises
            # less after a bad year than real coups did (offset logit on held-out years). Before the alignment, so
            # only the slope changes, not the level
            g1 = torch.nan_to_num(Xh[..., FEATURES.index("growth_1y")], nan=0.0).clamp(-0.3, 0.3)
            haz[..., 2] = haz[..., 2] + COUP_GROWTH * g1
        if ONSET_SLOPE != 1.0:
            # sharper onset risk: the network's log-odds times a slope, before the alignment re-solves the level
            haz[..., 0] = ONSET_SLOPE * haz[..., 0]
        if SHIFTERS_BEFORE_ALIGN:
            haz = _shifters(haz)
        if k == 1 and align:
            # solve for the log-odds shift that matches each target rate on the origin's risk sets
            conf0 = h["conflict"][..., -1] > 0.5
            for j, (target, risk) in enumerate([(r_on, ~conf0), (r_te, conf0), (r_cp, torch.ones_like(conf0))]):
                if target is None or not bool(risk.any()) or not (0 < target < 1):
                    continue
                z = haz[..., j][risk]
                lo_, hi_ = -4.0, 4.0
                for _ in range(40):
                    mid = (lo_ + hi_) / 2
                    if float(torch.sigmoid(z + mid).mean()) < target:
                        lo_ = mid
                    else:
                        hi_ = mid
                shift[j] = (lo_ + hi_) / 2
        haz = haz + shift
        if not SHIFTERS_BEFORE_ALIGN:
            haz = _shifters(haz)
        year = origin + k
        if WS is not None:
            A, dA, fr = WS.adoption(year)
            ai = WS.cfg.ai
            # near-free surveillance: rebellion and coups get harder in autocracies
            autoc = (h["dem"][..., -1] < 0.4).float()
            surv = ai.surveillance * (1 - WS.prm("sat_open") * WS.space_ramp(origin + k))   # satellite internet (cheap space)
            haz[..., 0] = haz[..., 0] - surv * A * autoc
            haz[..., 2] = haz[..., 2] - surv * A * autoc
            if climate_damage > 0:
                # sustained local warming since the origin raises civil-war onsets where farming is a large share of
                # the economy (assumption, range from no effect to a quarter of the untested panel fit)
                dT_l = torch.nan_to_num(prev_tas - cl["tas"]["base"]).clamp(min=0)
                haz[..., 0] = haz[..., 0] + WS.prm("clim_conflict") * dT_l * (WS.agri / 0.3).clamp(max=2)[None]
            if ai.arrival:
                # cheap strike drones in rebel hands make civil wars harder to end (assumption, range includes 0)
                haz[..., 1] = haz[..., 1] - WS.prm("drone_civil") * WS.mil_auto
        # bad mood: more onsets, fewer endings; coups have their own regional waves (e.g. Sahel 2020s)
        # the mood noise would raise average hazards (convexity); the second-order correction keeping the mean
        # probability is var/2 * (1 - 2p), close to var/2 for rare events but smaller for conflict endings (p ~ 0.2)
        var_ = torch.tensor([var_w + var_r + frailty_sd ** 2, var_w + var_r + (frailty_end * frailty_sd) ** 2, var_r + coup_frailty_sd ** 2], device=device)
        corr = var_ / 2 * (1 - 2 * torch.sigmoid(haz))
        if MOOD_JUMP:
            corr = corr + torch.tensor([jcorr[0], jcorr[1], jcorr[0]], device=device)
        haz = torch.sigmoid(haz + torch.stack([mood + frail, -mood - frailty_end * frail, mood_coup[:, regions] + frail_c], -1) - corr)
        if (k - 1) % regime_years == 0:
            g_regime = (GROWTH_REGIME_SCALE if growth_regime_scale is None else growth_regime_scale) * (growth.draw_regime(X, gen) + growth.draw_region_era(S, R, gen)[:, regions])
            d_regime = (DEM_REGIME_SCALE if dem_regime_scale is None else dem_regime_scale) * (ddem.draw_regime(X, gen) + ddem.draw_region_era(S, R, gen)[:, regions])
            era = growth.draw_era(S, gen)
        conf = h["conflict"][..., -1]
        u = torch.rand(S, C, device=device, generator=gen)
        conf_next = torch.where(conf > 0.5, (u >= haz[..., 1]).float(), (u < haz[..., 0]).float())
        onset_next = ((conf < 0.5) & (conf_next > 0.5)).float()
        if WS is not None:
            # a new war starts with the first-year mean number of rebel groups, and groups multiply as it ages (the
            # age profile in structure.py). It used to start at one group and never change, which gave every
            # post-origin war +0.47 log-odds of ending for its whole life (code review 2026-10-28)
            if LREB_MODE == "age":
                lreb = torch.where(onset_next > 0, torch.full_like(lreb, FRAG_NEW),
                                   torch.where(conf_next > 0.5, lreb + (FRAG_LONG - lreb) / FRAG_TAU, lreb))
            else:                           # (for comparisons: "old" = one group forever, "mean" = FRAG_MEAN)
                lreb = torch.where(onset_next > 0, torch.full_like(lreb, float(np.log(2.0)) if LREB_MODE == "old" else FRAG_MEAN), lreb)
        # a continuing war keeps its status; a new one is internationalized with the historical share
        u_i = torch.rand(S, C, device=device, generator=gen)
        intl = torch.where((conf > 0.5) & (conf_next > 0.5), intl,
                           torch.where(onset_next > 0, (u_i < R_.INTL_SHARE).float(), torch.zeros_like(intl)))
        coup_next = (torch.rand(S, C, device=device, generator=gen) < haz[..., 2]).float()
        coup_last = coup_next
        # interstate wars between country pairs (learned hazard, duration table, alliance cascade)
        if WS is not None and WS.dy is not None:
            mil = (A, fr) if WS.cfg.ai.arrival else None     # (military autonomy is read from WS.mil_auto)
            if WS.cfg.trade_gravity and k > 1 and k % 5 == 1:
                WS.update_trade(h["lg"][..., -1], h["lpop"][..., -1])
            inter_next = WS.interstate_step(k, h["lg"][..., -1], h["lpop"][..., -1], h["dem"][..., -1], conf_next,
                                           cur["coup_recent"], mil)
            WS.update_chip_hit()
        else:
            inter_next = torch.zeros(S, C, device=device)
        events = torch.stack([conf_next, onset_next, coup_next, inter_next], -1)

        shock_idx = growth.draw_year(era, gen)
        if WS is not None and WS.cfg.pandemics and i_covid is not None:
            # pandemics are drawn by the world layer: the replayed COVID crash and its rebound would count them twice
            # (they are replaced by the neighbouring years, 2018->19 and 2021->22)
            shock_idx = torch.where(shock_idx == i_covid, torch.full_like(shock_idx, i_covid - 1), shock_idx)
            shock_idx = torch.where(shock_idx == i_covid + 1, torch.full_like(shock_idx, i_covid + 2), shock_idx)
        # with the war-damage layer, the estimated war-site loss (country and year effects, 1951-2023) is the whole
        # cost of an interstate war: the growth equation's own interstate term would count it twice
        ev_g = events
        if WS is not None and WS.dy is not None and WS.cfg.war_damage:
            ev_g = torch.cat([events[..., :3], torch.zeros_like(events[..., 3:])], -1)
        g = growth.step(X, ev_g, shock_idx, g_regime, gen)
        if mc_models is not None:
            # debt and currency crises from last year's state, with their output cost this year and the next
            g_last = torch.nan_to_num(h["lg"][..., -1] - h["lg"][..., -2])
            for k_, mdl in mc_models.items():
                pz = MC.hazard(mdl, h["lg"][..., -1], cur["public_debt"], cur["debt_change_5y"], g_last, cur["inflation"])
                hit = (torch.rand(S, C, device=device, generator=g_mc) < pz).float() * (1 - mc_prev[k_])
                c0, c1 = MC.COST[k_]
                if MACRO_CRISIS == "centered":
                    # (the replayed growth residuals already hold the average crisis, at the historical onset rate pbar:
                    #  charge crises against that rate, so a country above the average risk pays more and one below
                    #  pays less, without a built-in drag. Centring on the country's own pz instead, as first tried,
                    #  makes every country's expected charge zero - found by the independent review, 2026-11-02)
                    pbar = mc_models[k_][2]
                    g = g - c0 * (hit - pbar) - c1 * (mc_prev[k_] - pbar)
                else:
                    g = g - c0 * hit - c1 * mc_prev[k_]
                mc_prev[k_] = hit
                mc_rec[k_].append(hit.sum(-1))
        if CRISIS_COST is not None:
            g = g - CRISIS_COST[0] * crisis_x - CRISIS_COST[1] * crisis_x_prev
        if WS is not None and WS.cfg.dollar_decline:
            g = g - WS.prm("usd_cost") * WS.dollar(origin + k) * WS.is_usa[None]      # the dollar's decline (scenario)
        if WS is not None and k == 1:
            g = g - WS.prm("aid_mult") * WS.aid_cut[None]       # 2026: demand lost to the 2025 USAID cut (assumption)
            # 2026: the 2025 US tariffs (intervention.py, assumption)
            g = g - WS.prm("tariff_pass") * WS.tariff_hit[None] - WS.pko_cut_on * WS.prm("tariff_us") * WS.is_usa[None]
        # workforce: a shrinking (growing) working-age share drags (lifts) output per person
        # (only the change beyond the 1960-2023 mean, which the growth equation already carries; robots offset part
        #  of a shrinking workforce at full adoption, robot_aging = the share offset, assumption)
        dlw = torch.nan_to_num(exo["lwork"][:, k] - exo["lwork"][:, k - 1]) - WORKFORCE_MEAN
        if WS is not None and WS.tech is not None:
            dlw = dlw + WS.tech.lever("work_gain")              # healthy years added (longevity, tech.py)
        if WS is not None:
            off = WS.prm("robot_aging") * WS.A_robot if WS.cfg.ai.arrival else 0.0
            g = g + WS.prm("workforce_beta") * torch.where(dlw < 0, dlw * (1 - off), dlw) + fish_g \
                + WS.prm("articles_beta") * WS.science * SCIENCE_DECAY ** (k - 1)   # science capacity beyond income (innovation.py)
        else:
            g = g + WORKFORCE_BETA * dlw
        # (democracy responds to recorded GDP, damage included, as in its training data; growth keeps X, its own
        # momentum terms must not compound the level damage)
        dd = ddem.step(Xh_d if damage_feedback and Xh_d is not None else X, events, shock_idx, d_regime, gen)
        pandemic = torch.zeros(S, device=device)
        if WS is not None:
            ai = WS.cfg.ai
            # AI dividend, capped by energy per person relative to this year's frontier (physical bottleneck), with
            # data centres taking a share of energy during the steepest adoption; frontier hubs capture extra rents
            if ai.arrival:
                e_pc = cur["energy_pc"].nan_to_num(WS.e_frontier * 0.3)
                e_front = torch.quantile(e_pc, 0.9, dim=1, keepdim=True)
                e_eff = e_pc * (1 - WS.prm("dc_share") * (1 - WS.prm("orbit_dc") * WS.space_ramp(origin + k)) * 4 * A * (1 - A))
                e_cap = (e_eff / e_front).clamp(0.09, 1) ** 0.5
                rent = 1 + WS.prm("hub_rent") * WS.capture() * (WS.hub - 0.3 * (1 - WS.hub))
                # a transfer between hubs and others: rescaled so the output-weighted world dividend is unchanged
                div = A * WS.cog_share * e_cap
                gw = torch.exp(h["lg"][..., -1] + h["lpop"][..., -1]).nan_to_num()
                rent = rent * (div * gw).sum(-1, keepdim=True) / (div * gw * rent).sum(-1, keepdim=True).clamp(min=1e-9)
                # ideas get harder to find: the dividend fades with cumulative adoption (innovation.py)
                WS.cum_A = WS.cum_A + A
                g = g + WS.prm("growth_dividend") * div * rent * torch.exp(-WS.prm("dividend_fade") * WS.cum_A)
                # development ladder: middle-income export manufacturers lose ground as robots reshore production;
                # robots also offset shrinking workforces in ageing societies
                Ar = WS.A_robot
                gap = X[..., FEATURES.index("gdp_gap_frontier")]
                middle = ((gap > -2.5) & (gap < -0.7)).float()
                # reshoring happens when the importing (frontier) economies automate, not when the exporter does
                Rf = WS.robot_frontier
                g = g - ai.ladder_penalty * 4 * Rf * (1 - Rf) * middle * WS.exporter.clamp(max=1)
                # (robots offsetting shrinking workforces: in the workforce term above)
                # physical automation: robots raise output per person in the non-cognitive part of the economy,
                # capped by energy like the cognitive dividend (assumption, 0-3 points a year at full adoption)
                g = g + WS.prm("phys_dividend") * Ar * (1 - WS.cog_share) * e_cap
            # chip chokepoint: a war on Taiwan or South Korea cuts output in proportion to the cognitive share of the
            # economy (with or without an AI shock: today's economies already run on those chips)
            g = g - WS.prm("chip_gdp") * WS.chip_hit * WS.cog_share
            # AI investment boom 2026-2030: a bust (once per world) costs frontier economies over two years
            if 2026 <= year <= 2030:
                p_y = 1 - (1 - WS.prm("bubble_p")) ** (1 / 5)
                p_y = p_y.flatten() if torch.is_tensor(p_y) else p_y
                burst = (torch.rand(S, device=device, generator=gen) < p_y) & (bubble_t < 0)
                bubble_t = torch.where(burst, torch.full_like(bubble_t, year), bubble_t)
            hit_b = ((bubble_t >= 0) & (year - bubble_t < 2)).float()[:, None]
            bc = WS.prm("bubble_cost")
            g = g - bc * 0.5 * hit_b * (0.5 + WS.hub) * WS.cog_share
            WS.rec["bubble"].append(hit_b[:, 0])
            # financial crises added by AI-correlated finance cost output (1952-2024 estimate), the year after
            g = g - WS.cfg.fin_crisis_cost[0] * fin_extra - WS.cfg.fin_crisis_cost[1] * fin_extra_prev
            # pandemics: a common shock to every country in that world
            if WS.cfg.pandemics:
                # natural pandemics, made rarer and milder by AI surveillance and countermeasures; plus, with AI, a
                # heavy tail of engineered ones, likelier in a more violent world
                frf = fr.flatten()
                p_nat = WS.prm("pandemic_p") * (1 - (WS.prm("natural_cut") if ai.arrival else 0.0) * fr)
                p_nat = p_nat.flatten() if torch.is_tensor(p_nat) else p_nat
                pandemic = (torch.rand(S, device=device, generator=gen) < p_nat).float()
                lo_p, hi_p = WS.cfg.pandemic_gdp
                hit = lo_p + (hi_p - lo_p) * torch.rand(S, device=device, generator=gen)
                pcut = (1 - WS.tech.lever("pandemic_cut").clamp(max=1)).flatten() if WS is not None and WS.tech is not None else 1.0   # fast vaccines
                g = g + (pandemic * hit * pcut * (1 - (1 - ai.bio_response) * frf))[:, None]
                pm_ = WS.prm("pandemic_mort")
                pand_mort = pandemic * (pm_.flatten() if torch.is_tensor(pm_) else pm_) * pcut * (1 - (1 - ai.bio_response) * frf)
                u_eng = torch.rand(S, device=device, generator=gen)       # drawn in every world: paired runs stay paired
                u3 = torch.rand(S, device=device, generator=gen) ** 3
                eng_t = WS.tech.lever("eng_pandemic").flatten() if WS is not None and WS.tech is not None else 0.0       # pathogen tools (tech.py)
                if ai.arrival or WS is not None and WS.tech is not None:
                    p_eng = ((WS.prm("eng_pandemic") * fr).flatten() if ai.arrival else 0.0) + eng_t
                    p_eng = p_eng * (1 + 2 * conf_next.mean(-1))
                    eng = (u_eng < p_eng).float()
                    g = g - (eng * (0.05 + 0.20 * u3))[:, None]
                    pand_mort = torch.maximum(pand_mort, eng * (0.002 + 0.05 * u3))     # deaths scale with severity (assumption)
                    pandemic = torch.maximum(pandemic, eng)
                    WS.rec["eng_pandemic"].append(eng)
            if WS.cfg.ai_catastrophe and ai.arrival:
                # AI loss of control (assumption, world.WorldConfig.ai_catastrophe): once per world at most
                u_c = torch.rand(S, device=device, generator=gen)
                p_c = (WS.prm("ai_cat_rate") * fr).flatten()
                cat = ((u_c < p_c) & (WS.ai_cat_done < 0.5)).float()
                WS.ai_cat_done = torch.maximum(WS.ai_cat_done, cat)
                loss_c = torch.as_tensor(WS.prm("ai_cat_loss"), device=device).flatten() * cat
                g = g - loss_c[:, None] + (0.1 * WS.ai_cat_back)[:, None]       # half recovered, a tenth of it a year
                WS.ai_cat_back = 0.9 * WS.ai_cat_back + 0.5 * loss_c
                pand_mort = torch.maximum(pand_mort, cat * torch.as_tensor(WS.prm("ai_cat_deaths"), device=device).flatten())
                WS.rec["ai_catastrophe"].append(cat)
            WS.rec["pandemic"].append(pandemic)
            # geomagnetic storm: grid-dependent economies lose output that year
            p_s = WS.prm("solar_p")
            solar = (torch.rand(S, device=device, generator=gen) < (p_s.flatten() if torch.is_tensor(p_s) else p_s)).float()
            trans_now = WS.prm("solar_loss") * solar[:, None] * WS.readiness[None]
            WS.rec["solar"].append(solar)
            # damaging storms in exposed coastal countries, more frequent with warming (beyond the replayed history)
            # (each country's own warming since the origin decade; none in runs without climate damage)
            dT_w = torch.nan_to_num(prev_tas - cl["tas"]["base"]) * float(climate_damage > 0)
            p_st = 0.05 * WS.prm("storm_trend") * dT_w.clamp(min=0) * storm_expo[None]
            storm = (torch.rand(S, C, device=device, generator=gen) < p_st).float()
            g = g - WS.prm("storm_loss") * storm
            # wartime attacks on infrastructure in this year's great-power wars. Solar storms and cyber attacks are
            # level losses that last while they do (repairs restore output): applied to the damage factor, not growth
            trans_now = trans_now + WS.prm("cyber_loss") * WS.cyber
            war_cum = war_cum - trans_now + trans_prev
            trans_prev = trans_now
            WS.rec["ai_adoption_world"].append((A * pop[:, k]).sum(-1) / pop[:, k].sum())
            WS.rec["robot_adoption_world"].append((WS.A_robot * pop[:, k]).sum(-1) / pop[:, k].sum())
            WS.rec["mil_autonomy"].append(WS.mil_auto[:, 0])
            # surveillance makes autocratization easier outside democracies
            dd = dd - WS.prm("autocracy_drift") * (1 - WS.prm("sat_open") * WS.space_ramp(origin + k)) * A * (h["dem"][..., -1] < 0.5).float()
            if ai.arrival:
                # elite overproduction and synthetic persuasion: credentialed work loses value fastest where many
                # graduates compete, eroding democracies during the transition (weaker with strong fiscal capacity)
                elite = (h["tertiary_enrol"][..., -1].nan_to_num(30) / 60).clamp(0, 1.5)
                dd = dd - WS.prm("dem_erosion") * 4 * A * (1 - A) * elite * (h["dem"][..., -1] >= 0.5).float() * (1 - 0.5 * WS.fiscal)
        # oil price: the scenario's historical year, plus (world system) a weak pull toward the 20-year
        # average, a jump when a major exporter enters a new interstate war, and lower demand as cheap
        # clean energy spreads; exporters gain and importers lose relative to the replayed history
        if OIL_INDEP:
            # (a historical year of its own each year, not the world's replayed decade: a decade block repeats one
            #  era's direction of the oil price ten times; own generator so the other draws stay paired)
            if k == 1:
                gen_oil = torch.Generator(device=device).manual_seed(seed + 7777)
            oil_hist = oil_change_t[torch.randint(len(oil_change_t), (S,), device=device, generator=gen_oil)][:, None]
        else:
            oil_hist = oil_change_t[shock_idx][:, None]
        oil_year = oil_change_t[shock_idx][:, None]    # the move in the year whose growth shocks are replayed
        oil_change = oil_hist
        if WS is not None and WS.cfg.energy:
            # historical year-to-year moves without their 1950-2024 trend, a pull toward the 1995-2024 average,
            # and war spikes that unwind the following year (as in 1990-91)
            oil_change = oil_hist - oil_trend - WS.cfg.oil_reversion * (h["oil_price"][..., -1] - oil_anchor) - oil_spike
            if WS.dy is not None:
                # (a new war-level conflict: a 25-death clash such as US-Venezuela in 2026 does not move the market)
                oil_spike = WS.prm("oil_war_jump") * (WS.new_war_country[:, oil_exporters].sum(-1, keepdim=True) > 0).float()
                oil_change = oil_change + oil_spike
            dlogD = torch.zeros(S, 1, device=device)
            if WS.cfg.transition:
                adopt = (A * pop[:, k]).sum(-1, keepdim=True) / pop[:, k].sum() if WS.cfg.ai.arrival else 0.0
                tr_r = tr_rate * (1 + WS.cfg.ai_transition_boost * adopt) * (1 - WS.prm("china_pause") * WS.china_hit)
                tr_Kt = tr_K
                if WS is not None and WS.tech is not None:
                    # storage, firm clean power, hydrogen: a faster transition and a higher ceiling (tech.py)
                    tr_r = tr_r * (1 + WS.tech.lever("clean_rate"))
                    tr_Kt = (tr_K + WS.tech.lever("clean_cap")).clamp(max=0.95)
                tr_s_new = tr_Kt / (1 + (tr_Kt / tr_s - 1) * torch.exp(-tr_r))          # one logistic step
                F_old = (WS.cfg.fossil_share_0 - (tr_s - tr_s0)).clamp(min=0.05)
                F_new = (WS.cfg.fossil_share_0 - (tr_s_new - tr_s0)).clamp(min=0.05)
                dlogD = torch.log(F_new / F_old)
                # transition minerals (copper, lithium, cobalt, nickel, rare earths): producers gain with the clean share
                # (with new capacity: x4 the change in share gives about the same total as the log growth of the share
                # over the transition, but timed with installations rather than front-loaded)
                g = g + WS.prm("mineral_elast") * WS.rents[None] * 4 * (tr_s_new - tr_s)
                tr_s = tr_s_new
                WS.rec["clean_share"].append(tr_s[:, 0])
                oil_change = oil_change + WS.prm("oil_eta") * dlogD
            oil_noise = torch.randn(S, 1, device=device, generator=gen)      # drawn every year: paired runs stay in step
            obs = dict((y_, (c_, sd_)) for y_, c_, sd_ in WS.cfg.oil_observed).get(origin + k)
            if obs is not None:
                # a year already (partly) observed: the recorded change in the real price, with the unobserved months
                # as noise; the rise above zero unwinds the next year, like a war spike
                oil_change = obs[0] + obs[1] * oil_noise
                oil_spike = torch.full_like(oil_change, max(obs[0], 0.0))
            # (exporters and importers gain or lose against the oil move already inside the replayed growth shocks)
            dev = oil_change - oil_year
            # importers lose only on price rises: the effect of the simulated move minus that of the replayed one (the
            # clamp of their difference would charge importers on average once the two moves are drawn independently)
            rise = oil_change.clamp(min=0) - oil_year.clamp(min=0) if OIL_INDEP else dev.clamp(min=0)
            g = g + WS.cfg.oil_beta * oil_expo * dev + WS.cfg.oil_importer * (oil_expo < 0.05).float() * rise
            g = g + WS.cfg.oil_beta * oil_expo * dlogD                  # exporters also lose volume
            WS.rec["oil"].append(h["oil_price"][:, 0, -1] + oil_change[:, 0])
        oil_next = h["oil_price"][..., -1] + oil_change
        if WS is not None and WS.dy is not None and WS.cfg.war_damage:
            site = WS.war_site
            gdp_w = torch.exp(h["lg"][..., -1] + h["lpop"][..., -1]).nan_to_num()
            share = (site * gdp_w).sum(-1, keepdim=True) / gdp_w.sum(-1, keepdim=True)
            WS.rec["war_site_share"].append(share[:, 0])
            WS.rec["war_sites"].append(site.sum(-1))
            excess = (share - WS.cfg.war_share_norm).clamp(min=0)
            # a level factor on reported GDP, like climate damage: the estimate (-9%, then +4.5% two years on)
            # already includes the aftermath, so it is kept out of the growth dynamics that would amplify it
            # (2026-10-26) the spillover reaches each country through its trade: the gravity-weighted share of its
            # partners at war (risks.trade_weights) above the historical norm, x its trade openness relative to the mean.
            # Same world average as the earlier uniform spillover, concentrated on the war's trading partners
            # (only the excess over the historical norm, already in replayed growth, spills over: each country's
            # exposure is scaled by the world's excess share of output at war sites)
            expo = site @ WS.trade_w.T
            spill = WS.prm("war_spillover") * WS.prm("war_site_loss") * WS.trade_rel[None] \
                * expo * (excess / share.clamp(min=1e-6)) * (1 - site)
            # both war sites and trade partners recover half the loss two years later, as war sites did historically
            site = (site + WS.strike_site).clamp(max=1)                # own output loss includes drone strikes
            war_cum = war_cum - WS.prm("war_site_loss") * site - spill + WS.prm("war_site_loss") * (WS.cfg.war_rebound / WS.cfg.war_site_loss) * site_hist[0] + 0.5 * spill_hist[0]
            # sanctions on a non-Western state at war with the West (x its Western trade share), and nuclear use
            # (sanctions: a growth drag while they last; nuclear damage: a lasting level loss, in log units)
            # (nuclear damage counts once per country: the largest hit so far, not every year of use again)
            nuc_new = torch.maximum(nuc_cum, WS.nuc_hit.clamp(max=0.9))
            war_cum = war_cum - WS.prm("sanction_loss") * (1 - 0.5 * WS.dollar(origin + k)) * WS.west_share[None] * WS.sanctioned \
                + torch.log1p(-nuc_new) - torch.log1p(-nuc_cum)
            nuc_cum = nuc_new
            site_hist = site_hist[1:] + [site]
            spill_hist = spill_hist[1:] + [spill]
        lg_next = h["lg"][..., -1] + g
        dem_prev = h["dem"][..., -1]
        dem_next = (dem_prev + dd).clamp(0, 1)

        u_prev = h["urban"][..., -1]
        urban_next = (u_prev + urban_rate * u_prev * (100 - u_prev) / 100).clamp(max=100)
        smooth, weather = {}, {}
        for v, d in cl.items():
            up, dn = (d["p90"][:, k] - d["median"][:, k]) / 1.2816, (d["median"][:, k] - d["p10"][:, k]) / 1.2816
            smooth[v] = d["median"][:, k] + torch.where(z_clim > 0, z_clim * up, z_clim * dn)
            # shocks shared across countries as observed (risks.weather_factors), not independent
            F_, u_ = wfac[v]
            z_ = torch.randn(S, F_.shape[1], device=device, generator=gen) @ F_.T + u_ * torch.randn(S, C, device=device, generator=gen)
            weather[v] = smooth[v] + d["sd"] * z_
        if WS is not None and WS.tech is not None:
            # carbon removal and geoengineering remove a share of the warming since the origin (all climate channels)
            wc_ = WS.tech.lever("warming_cut").clamp(max=1)
            for v in ("tas", "txx"):
                if v in smooth:
                    dlt = wc_ * (smooth[v] - cl[v]["median"][:, 0])
                    smooth[v], weather[v] = smooth[v] - dlt, weather[v] - dlt
        amoc_r = None
        if WS is not None:
            # AMOC collapse (tail assumption): rain belts shift south, drying the Sahel and West Africa (half as much
            # in South Asia), ramping in over 15 years from the year its effects begin
            amoc_r = ((year - WS.amoc_year) / 15.0).clamp(0, 1)                                     # [S, 1]
            dry_f = 1 - amoc_r * WS.prm("amoc_rain") * (WS.sahel + 0.5 * WS.south_asia)[None]
            smooth["pr"], weather["pr"] = smooth["pr"] * dry_f, weather["pr"] * dry_f
            WS.rec["amoc"].append((amoc_r[:, 0] > 0).float())
        weather["pr"] = weather["pr"].clamp(min=0)
        weather["cdd"] = weather["cdd"].clamp(0, 366)
        if climate_damage:
            # damage from warming beyond the origin decade, a separate log-level factor on reported GDP (not fed back
            # into the growth dynamics, whose momentum terms would otherwise amplify it). Each world draws one family:
            # 1 BHM (2015) growth effect, accumulated as in BHM's own projections; 0 the same function on the level
            # (Newell et al. 2021: level models fit as well and imply 1-3% losses by 2100); 2 global temperature
            # (Bilal & Kaenzig 2024): the same loss everywhere per degC of global warming, after a lag
            ts = smooth["tas"] + t_off
            step = BHM[0] * (ts - t_base) + BHM[1] * (ts ** 2 - t_base ** 2)
            mult = WS.params["climate_damage"] * climate_damage if WS is not None and "climate_damage" in WS.params else climate_damage
            if clim_fam is None:
                dmg_cum = dmg_cum + mult * step
            else:
                from .world import GLOBAL_PER_LAND, BK_PER_DEG, BK_LAG
                bhm_cum = bhm_cum + step
                dTg = (torch.nan_to_num(smooth["tas"] - cl["tas"]["base"]) * pw[None]).sum(-1, keepdim=True) * GLOBAL_PER_LAND
                dTg_lag = dTg_lag + (dTg - dTg_lag) / BK_LAG
                lvl = torch.where(clim_fam == 0, step, torch.where(clim_fam == 1, bhm_cum, -BK_PER_DEG * dTg_lag.expand_as(step)))
                lvl = mult * lvl
                dmg_cum = dmg_cum + lvl - clim_prev
                clim_prev = lvl
        if WS is not None and amoc_r is not None:
            amoc_lvl = -WS.prm("amoc_eu_loss") * amoc_r * WS.nw_europe[None] * float(climate_damage > 0)
            dmg_cum = dmg_cum + amoc_lvl - amoc_prev
            amoc_prev = amoc_lvl
        if WS is not None:
            # sea-level rise: output lost in low-lying areas (people below 5 m), growing with the rise (assumption)
            coast = -WS.prm("slr_cost") * WS.lowpop[None] * WS.prm("slr_mult") * R_.slr(year, origin) / 0.5 * float(climate_damage > 0)
            dmg_cum = dmg_cum + coast - coast_prev
            coast_prev = coast
        tert_next = (h["tertiary_enrol"][..., -1] + tert_rate * 0.97 ** k).clamp(0, 150)
        # oil price moves as it did in the scenario's historical shock year (keeps oil busts and world recessions coupled)
        cur["oil_price_20y"] = cur["oil_price_20y"] + (oil_next - cur["oil_price_20y"]) / 20
        # war-level intensity of civil conflicts (UCDP, 1946-2024 country-years in conflict): a new conflict starts at
        # war level with p = 0.155 (n = 291), a minor one escalates with p = 0.115 (n = 1004), a war stays at war
        # level with p = 0.771 (n = 502)
        u_w = torch.rand(S, C, device=device, generator=gen)
        cur["war"] = conf_next * torch.where(onset_next > 0, u_w < 0.155,
                                             torch.where(cur["war"] > 0.5, u_w < 0.771, u_w < 0.115)).float()
        # battle deaths persist while a conflict lasts; a new conflict starts at a low level
        cur["battle_deaths_pc"] = torch.where(conf_next > 0.5, torch.where(onset_next > 0, torch.full_like(conf_next, 2.0),
                                                                           cur["battle_deaths_pc"]), torch.zeros_like(conf_next))
        # refugees first, so that this year's departures reach this year's population
        if mig is None:
            cur["refugees_pc"] = torch.where(conf_next > 0.5, cur["refugees_pc"], 0.85 * cur["refugees_pc"])
        else:
            # countries with no refugee record at the origin start from none and take the median peacetime level
            # (they must still send refugees if a war reaches them: Taiwan and Serbia are missing from the source)
            r_ = torch.nan_to_num(cur["refugees_pc"]).clamp(min=0)
            L_ = torch.log(r_ + 1e-5)
            fe = mig["fe"] if "fe" in mig else None
            if fe is None:
                fe_ = mig["eq"]["fe"]
                fe = mig["fe"] = tt(np.where(np.isfinite(fe_), fe_, np.nanmedian(fe_)))
            b_c, b_w = mig["eq"]["beta"]
            war_lvl = cur["war"] * conf_next
            if WS is not None and hasattr(WS, "war_site"):
                # a country that is the site of a war-level interstate conflict sends refugees like a civil conflict
                # (as in estimation, where interstate conflict on a country's soil counts as conflict; the war-level
                # indicator there covers civil wars only)
                conf_lvl = torch.maximum(conf_next, WS.war_site)
            else:
                conf_lvl = conf_next
            L_ = L_ + mig["eq"]["lam"] * (fe + b_c * conf_lvl + b_w * war_lvl - L_)
            cur["refugees_pc"] = (torch.exp(L_) - 1e-5).clamp(0, 0.6)
            pop_k = torch.exp(exo["lpop"][:, k])
            stock = torch.nan_to_num(cur["refugees_pc"] * pop_k)
            # only departures beyond the origin stock move people: UN projections already contain both today's
            # refugees and their expected returns, so returns are not added a second time. Capped at half the
            # population before sharing out, so that hosts receive exactly what origins lose
            dep = (stock - mig["stock0"]).clamp(min=0)
            if WS is not None:
                # climate emigration (assumption): a stock that grows with the climate harvest loss x farm share
                c_dep = WS.prm("clim_migr") * WS.agri[None] * (-clim_ly).clamp(min=0) * pop_k
                WS.rec["clim_migrants"].append(c_dep.sum(-1) / 1e6)
                dep = dep + c_dep
            dep = torch.minimum(dep, 0.5 * pop_k)
            mig["moved"] = dep @ mig["H"] - dep
            rec["displaced"].append(stock.sum(-1) / 1e6)
        fert_ratio = None
        if WS is not None:
            # population (2026-10-26): war deaths (battle x civilian multiplier, nuclear) and pandemic deaths come off
            # the UN path; fertility responds to income beyond what the projection assumes (-0.035 per unit change in
            # log GDP per person against the median world, 1970-2023 estimate); births follow fertility
            pop_now = torch.exp(exo["lpop"][:, k]) * torch.exp(pop_off)
            dshare = ((WS.deaths_year / pop_now.clamp(min=1)) + pand_mort[:, None]).clamp(0, 0.5)
            if WS.cfg.heat_mortality and climate_damage:
                # heat and cold deaths from warming since the origin (Carleton et al. 2022), adapting with income
                inc = torch.exp(lg_next + dmg_cum).nan_to_num(HEAT_MORT_INCOME[0])
                adapt = (inc / HEAT_MORT_INCOME[0]).clamp(min=1e-3) ** (-HEAT_MORT_INCOME[1])
                # (convex in warming, as extreme-heat days multiply: power 1.7 makes SSP2-4.5's 2100 toll about a sixth of
                #  SSP5-8.5's, as in Carleton et al., where a linear response gave a third)
                dT_l = torch.nan_to_num(smooth["tas"] - tas_0)
                dT_l = torch.sign(dT_l) * dT_l.abs() ** HEAT_MORT_POWER
                hrate = (HEAT_MORT_BETA * WS.prm("heat_mort") * dT_l * heat_h[None] * adapt.clamp(0.25, 3.0)).clamp(-0.004, 0.004)
                WS.rec["clim_deaths"].append((hrate * pop_now).sum(-1))
                dshare = dshare + hrate
            excess_deaths = excess_deaths + (dshare * pop_now).sum(-1)
            # (income against the country's own pre-origin trend path, which UN projections broadly assume, and after
            # war and climate damage; not against the other simulated worlds)
            fert_ratio = torch.exp(R_.FERT_INCOME * torch.nan_to_num(lg_next + dmg_cum + war_cum - (lg_ref0 + g_ref * k)))
            if dem_sig is not None:
                fert_ratio = fert_ratio * torch.exp(z_dem * dem_sig[:, min(year - dem_y0, dem_sig.shape[1] - 1)][None])
                rest_now = z_rest * dem_rest[:, min(year - dem_y0, dem_rest.shape[1] - 1)][None].nan_to_num(0.0)
                pop_off = pop_off + rest_now - rest_prev
                rest_prev = rest_now
            births = R_.CBR_PER_TFR * exo["fertility"][:, k] * (fert_ratio - 1)
            # (bounded: a population 55x (e^4) above or below the UN path is outside any projection; keeps the world sums finite)
            pop_off = (pop_off + torch.log1p(births - dshare)).nan_to_num(0.0).clamp(-4, 4)
            # (births raise the youth share by b (1 - youth); deaths across all ages leave it unchanged)
            youth_off = youth_off * (14 / 15) + births * (1 - exo["youth"][:, k][None].nan_to_num(0.3))
            pop_un = torch.exp(exo["lpop"][:, k]).nan_to_num()[None]
            WS.rec["pop_dev"].append((pop_un * torch.expm1(pop_off)).nan_to_num().sum(-1) / pop_un.sum(-1))
            WS.rec["excess_deaths"].append(excess_deaths.clone())
        new = {"lg": lg_next, "dem": dem_next, "conflict": conf_next, "urban": urban_next,
               "lpop": exo["lpop"][:, k].expand(S, C) if mig is None else
               torch.log((torch.exp(exo["lpop"][:, k]) + mig["moved"]).clamp(min=0.5 * torch.exp(exo["lpop"][:, k]))),
               "tertiary_enrol": tert_next, "oil_price": oil_next, **weather}
        if WS is not None:
            new["lpop"] = new["lpop"] + pop_off
        for key in WINDOW_VARS:
            h[key] = torch.cat([h[key][..., 1:], new[key][..., None]], -1)
        dmg_hist = torch.cat([dmg_hist[..., 1:], (war_cum + dmg_cum)[..., None]], -1)
        for key in ("median_age", "youth", "fertility"):
            cur[key] = exo[key][:, k].expand(S, C)
        if fert_ratio is not None:
            cur["fertility"] = cur["fertility"] * fert_ratio
            cur["youth"] = (cur["youth"] + youth_off).clamp(0.05, 0.6)
        # health: damped trend, with a transient penalty while at war (not a permanent ratchet)
        cm_trend = cm_trend + cm_rate * 0.97 ** k
        le_trend = (le_trend + le_rate * 0.97 ** k).clamp(max=92)
        cur["log_cm"] = cm_trend + 0.1 * conf_next
        cur["life_exp"] = le_trend - 2.0 * conf_next
        # transitory macro conditions revert toward the country's recent norm
        if INFL_CONV:
            infl_target = infl_target + INFL_CONV * (infl_med - infl_target)
        cur["inflation"] = infl_target + 0.8 * (cur["inflation"] - infl_target)
        u_target = unemp_target
        if WS is not None:
            ai = WS.cfg.ai
            # cognitive-labour displacement: a transitional rise, largest where adoption is steepest
            u_target = unemp_target + ai.displacement_unemp * 4 * A * (1 - A) * WS.cog_share
            cap = WS.capture() if ai.arrival else 1.0      # who keeps the rents: firms (capture) or everyone (world.py)
            ai_top1 = ai.top1_rise * A * WS.cog_share * (1 - 0.5 * WS.fiscal) * cap
            ai_pol = ai.polarization_rise * dA.clamp(min=0) * (dem_next > 0.5).float() * (0.5 + 0.5 * cap)
            if slow is None:
                cur["top1"] = top1_0 + ai_top1
            else:
                # AI shifts move the level and the anchor it reverts to
                cur["top1"] = cur["top1"] + ai_top1 - ai_top1_prev
                slow.shift["top1"] = ai_top1
                slow.shift["polarization"] = slow.shift.get("polarization", 0.0) + ai_pol
                ai_top1_prev = ai_top1
            cur["polarization"] = cur["polarization"] + ai_pol
            for key in ("rule_of_law", "admin_impartial"):
                cur[key] = cur[key] + ai.capacity_gain * A * (bounds[key][1] - cur[key]).clamp(min=0) * 0.5
            cm_trend = cm_trend - ai.health_gain * A
            le_trend = (le_trend + 0.1 * A).clamp(max=95)
            cur["log_cm"] = cm_trend + 0.1 * conf_next
            cur["life_exp"] = le_trend - 2.0 * conf_next - 0.5 * pandemic[:, None]
            if WS.cfg.food:
                # yields: country trend (damped) + warming damage + AI agronomy; calories follow income and yields
                dT = smooth["tas"] - prev_tas
                dly = WS.yield_trend * 0.97 ** k + WS.cfg.yield_per_degC * dT + ai.yield_gain * A
                if WS is not None and WS.tech is not None:
                    dly = dly + WS.tech.lever("yield_gain")          # engineered crops, cheap fertilizer (tech.py)
                # this year's weather against its expected value moves the harvest (1962-2022 estimate: -1.4% per degC
                # of hottest-day anomaly, +5.6% per unit log rain anomaly); the level recovers when the weather does
                an_ = {"txx": torch.nan_to_num(weather["txx"] - smooth["txx"]),
                       "pr": torch.nan_to_num(torch.log(weather["pr"].clamp(min=1) / smooth["pr"].clamp(min=1)))}
                clim_d = WS.cfg.yield_per_degC * dT
                if fd is not None and "yres" in fd:
                    # harvests replay the same historical year as the growth shocks (2012's harvests with 2012's
                    # economy), and bad years deepen with local warming (harvest_amp, assumption); this replaces the
                    # yearly weather terms, which explained ~15% of the US yield swings and gave a 1% spread
                    # against 7.6% observed
                    r_now = fd["yres"][shock_idx]
                    if WS is not None and WS.tech is not None:
                        r_now = r_now * (1 - WS.tech.lever("harvest_cut").clamp(max=1) * (r_now < 0).float())
                    amp = WS.prm("harvest_amp") * torch.nan_to_num(smooth["tas"] - cl["tas"]["base"]).clamp(min=0) * r_now.clamp(max=0)
                    r_tot = r_now + amp
                    dly = dly + r_tot - fd["r_prev"]
                    # (the deepening of this year's bad harvest is weather, not a lasting loss: it stays out of clim_ly)
                    fd["r_prev"], fd["amp_prev"] = r_tot, amp
                    WS.rec["harvest_us"].append(r_tot[:, fd["i_us"]] if fd["i_us"] is not None else torch.zeros(S, device=device))
                    WS.rec["harvest_bad4"].append((r_tot[:, fd["i_x4"]] < -0.10).float().sum(-1))
                else:
                    dly = dly + R_.YIELD_TXX * (an_["txx"] - an_prev["txx"]) + R_.YIELD_LOGPR * (an_["pr"] - an_prev["pr"])
                an_prev = an_
                if WS is not None and amoc_r is not None:
                    a_y = -WS.prm("amoc_eu_yield") * amoc_r * WS.nw_europe[None]
                    dly = dly + a_y - fd.get("amoc_y_prev", 0.0) if fd is not None else dly
                    clim_d = clim_d + a_y - (fd.get("amoc_y_prev", 0.0) if fd is not None else 0.0)
                    if fd is not None:
                        fd["amoc_y_prev"] = a_y
                # a large eruption (VEI 6+) cools the world and cuts harvests for two years
                p_v = WS.prm("volc_p")
                p_v = p_v.flatten() if torch.is_tensor(p_v) else p_v
                volc = (torch.rand(S, device=device, generator=gen) < p_v) & (volc_t < 0)
                volc_t = torch.where(volc, torch.full_like(volc_t, year), volc_t)
                v_dly = WS.prm("volc_yield") * ((volc_t == year).float() - (volc_t == year - 2).float())[:, None]
                dly = dly - v_dly
                WS.rec["volcano"].append((volc_t == year).float())
                volc_t = torch.where(year - volc_t >= 2, torch.full_like(volc_t, -99), volc_t)   # (recovered)
                if WS.cfg.water:
                    # water supply (glacier runoff, precipitation) moves yields where water is binding (assumption)
                    dW = WS.water_step(year, torch.nan_to_num(smooth["pr"] / cl["pr"]["base"], nan=1.0))
                    dly = dly + WS.prm("yield_water") * WS.water_bind * dW
                cur["cereal_yield"] = cur["cereal_yield"] * torch.exp(dly)
                clim_ly = clim_ly + torch.nan_to_num(clim_d)
                if fd is None:
                    cur["calories"] = (cur["calories"] * torch.exp(0.1 * g + 0.3 * dly)).clamp(1400, 3900)
                else:
                    # world grain price: oil pass-through, last year's world harvest shortfall, the scenario year's
                    # residual and a pull toward the 1975..origin average, estimated together; the first year is
                    # observed (FAO cereal index, partial year)
                    wp, kc = fd["wp"], fd["kc"]
                    noise = 0.03 * torch.randn(S, 1, device=device, generator=gen)
                    dlp = wp["const"] + fd["res"][shock_idx][:, None] + wp["oil"] * oil_change + wp["supply"] * fd["sup_dev"] \
                        - wp["revert"] * (fd["lp"] - wp["anchor"])
                    if k == 1 and fd["obs"] is not None:
                        dlp = fd["obs"] + noise
                    if WS is not None and WS.tech is not None:
                        # cheap fertilizer and new protein: the world grain price falls by the adopted share
                        fc_ = torch.log1p(-WS.tech.lever("food_cut").clamp(max=0.9))
                        dlp = dlp + fc_ - fd.get("fcut_prev", 0.0)
                        fd["fcut_prev"] = fc_
                    fd["lp"] = fd["lp"] + dlp
                    food_dlp = dlp
                    # drylands: cereal area follows decade-mean rainfall (estimate, food.dryland), within the arable cap
                    lpr_s = torch.log(torch.nan_to_num(smooth["pr"], nan=1.0).clamp(min=1))
                    d_dry = torch.zeros_like(fd["area"])
                    if "lpr_prev" in fd:
                        d_dry = WS.prm("dry_area_elast") * WS.dry[None] * (lpr_s - fd["lpr_prev"])
                    fd["lpr_prev"] = lpr_s
                    cap_ = torch.maximum(fd["a_cap"], fd["area"])
                    area_new = torch.minimum(fd["area"] * torch.exp(fd["a_rate"] * 0.97 ** k + d_dry), cap_)
                    # (the rainfall-driven part of the area change is a lasting climate harvest change, for migration)
                    area_nc = torch.minimum(fd["area"] * torch.exp(fd["a_rate"] * 0.97 ** k), cap_)
                    clim_ly = clim_ly + torch.nan_to_num(torch.log(area_new.clamp(min=1) / area_nc.clamp(min=1)))
                    dla = torch.log(area_new.clamp(min=1) / fd["area"].clamp(min=1))
                    fd["area"] = area_new
                    wprod = (torch.nan_to_num(cur["cereal_yield"]) * fd["area"]).sum(-1, keepdim=True)
                    wpop = (torch.exp(exo["lpop"][:, k]).nan_to_num()[None] * torch.exp(torch.as_tensor(pop_off, device=device)).nan_to_num(1.0)).sum(-1, keepdim=True)
                    fd["sup_dev"] = torch.log(wprod / fd["wprod"]) - torch.log(wpop / fd["wpop"]) - wp["supply_mean"]
                    fd["wprod"], fd["wpop"] = wprod, wpop
                    # self-sufficiency: harvests against population (after refugee moves) and per-person use, which
                    # grows with income
                    dlpop = torch.nan_to_num(h["lpop"][..., -1] - h["lpop"][..., -2])
                    e_inc = (fd["dm"]["e0"] + fd["dm"]["e1"] * (h["lg"][..., -2].nan_to_num(9.0) - 9)).clamp(min=0)
                    fd["s"] = fd["s"] * torch.exp(torch.nan_to_num(dly) + dla - dlpop - e_inc * g)
                    depp = (1 - fd["s"]).clamp(0, 1)
                    cur["calories"] = (cur["calories"] * torch.exp(kc["g"] * g + kc["own"] * (1 - depp) * (torch.nan_to_num(dly) + dla - dlpop)
                                                                   + kc["price"] * depp * dlp)).clamp(1400, 3900)
                    WS.rec["food_price"].append(fd["lp"][:, 0])
                    WS.rec["import_dep"].append(depp.median(0).values)          # [C], median across worlds
                # wild-catch fisheries: ocean warming by latitude band (ocean ~0.7 x land warming) and the global
                # overfishing / management trend; the change in wild catch moves calories by the fish share of the
                # diet and income (next year) by the catch value share of GDP (ocean.py; assumption)
                band_b = torch.cat([torch.as_tensor(WS.prm(k), device=device, dtype=torch.float32).reshape(-1, 1).expand(S, 1)
                                    for k in ("fish_trop", "fish_mid", "fish_high")], 1)                     # [S, 3]
                dfish = WS.prm("wild_trend") + band_b[:, WS.fish_band] * 0.7 * torch.nan_to_num(dT) * float(climate_damage > 0)
                cur["calories"] = (cur["calories"] * torch.exp(WS.fish_cal * WS.fish_wild * dfish)).clamp(1400, 3900)
                fish_g = WS.fish_gdp * dfish
                fish_log = fish_log + (dfish * WS.fish_capt).sum(-1) / WS.fish_capt.sum().clamp(min=1)
                WS.rec["wild_catch"].append(fish_log.clone())
            # energy per person follows income, plus cheap clean energy once AI arrives
            # (a uniform factor would cancel in the energy cap, which is relative to the frontier: cheap clean energy
            #  instead lets energy-poor countries close part of their gap to the frontier)
            e_rel = (cur["energy_pc"] / torch.quantile(cur["energy_pc"].nan_to_num(WS.e_frontier * 0.3), 0.9, dim=1,
                                                         keepdim=True)).clamp(0, 1).nan_to_num(0.3)
            e_tech = WS.tech.lever("energy_gain") if WS is not None and WS.tech is not None else 0.0
            cur["energy_pc"] = cur["energy_pc"] * torch.exp(0.5 * g + (ai.clean_energy_gain * fr + e_tech) * (1 - e_rel))
            prev_tas = smooth["tas"]
            # (oil exporters' coup risk already moves with oil prices through the country models' oil-exposure
            #  features: only food exporters are added here)
            tot_prev = (WS.food_x[None] * food_dlp).clamp(-0.5, 0.5)
        cur["unemployment"] = u_target + 0.8 * (cur["unemployment"] - u_target)
        # rights and rule-of-law indicators move with the simulated democracy index
        for key, b_ in slopes.items():
            if slow is not None and key in slow.eq:
                continue                                  # (has its own equation, dynamics.INST_SLOW)
            cur[key] = (cur[key] + b_ * (dem_next - dem_prev)).clamp(*bounds[key])
        cur["ysc"] = torch.where(conf_next > 0.5, torch.zeros_like(cur["ysc"]), cur["ysc"] + 1)
        cur["coup_recent"] = DECAY["coup_recent"] * cur["coup_recent"] + coup_next
        cur["coups_total"] = cur["coups_total"] + coup_next
        cur["conflict_years"] = cur["conflict_years"] + conf_next
        cur["obs_years"] = cur["obs_years"] + 1
        # (a deep recession, war and climate damage of the year included, makes a financial crisis likelier)
        g_rec = g + (dmg_hist[..., -1] - dmg_hist[..., -2]) if damage_feedback else g
        p_fc = torch.where(g_rec < -0.05, p_crisis_low, p_crisis_high)
        if WS is not None:
            # short-term external debt against reserves raises the odds of a banking crisis (structure.py)
            odds = p_fc / (1 - p_fc).clamp(min=1e-6) * torch.exp(WS.prm("st_debt_beta") * WS.st_debt[None])
            p_fc = odds / (1 + odds)
            # contagion through trade: a first round at a scaled-down rate, then crises among each country's trade
            # partners raise its chance (fin_contagion), keeping the historical rate at historical exposure
            sc = WS.contagion_scale()
            p1 = p_fc * sc
            c1 = (torch.rand(S, C, device=device, generator=gen) < p1).float()
            e = c1 @ WS.trade_w.T
            p2 = (p1 * (torch.exp(WS.prm("fin_contagion") * e) - 1) / (1 - p1).clamp(min=1e-6)).clamp(0, 1)
            crisis = torch.maximum(c1, (torch.rand(S, C, device=device, generator=gen) < p2).float() * (1 - c1))
        else:
            crisis = (torch.rand(S, C, device=device, generator=gen) < p_fc).float()
        if WS is not None:
            # correlated algorithmic finance: extra crises, which cost output from next year (see above)
            fin_extra_prev = fin_extra
            fin_extra = ((torch.rand(S, C, device=device, generator=gen) < p_fc * (WS.cfg.ai.finance_mult ** A - 1))
                         & (crisis < 0.5)).float()
            crisis = torch.maximum(crisis, fin_extra)
            if WS.tech is not None and "quantum_crypto" in WS.tech.arrival:
                # encryption broken: a one-off wave of extra financial crises in the arrival year and the next, which
                # cost output as the AI-finance crises do (assumption; drawn every year so paired runs stay paired)
                shock_q = (WS.tech.arrived_in("quantum_crypto", year) + WS.tech.arrived_in("quantum_crypto", year - 1)).clamp(max=1)
                m_q = (WS.tech._m["quantum_crypto"]["crisis_mult"] - 1) * shock_q
                q_extra = ((torch.rand(S, C, device=device, generator=gen) < (p_fc * m_q).clamp(max=1)) & (crisis < 0.5)).float()
                fin_extra = torch.maximum(fin_extra, q_extra)
                crisis = torch.maximum(crisis, q_extra)
        if CRISIS_COST is not None:
            crisis_x_prev, crisis_x = crisis_x, crisis - p_fc
        cur["fin_crisis_recent"] = DECAY["fin_crisis_recent"] * cur["fin_crisis_recent"] + crisis
        if slow is not None:
            slow.step(cur, g, conf_next, crisis, dem_next - dem_prev)
            debt_hist = torch.cat([debt_hist[..., 1:], cur["public_debt"][..., None]], -1)
            cur["debt_change_5y"] = debt_hist[..., -1] - debt_hist[..., 0]
        for key in DECAY:
            if key not in ("coup_recent", "fin_crisis_recent") and key not in INST_CHANGE \
                    and not (slow is not None and (key == "debt_change_5y" or key in slow.eq)):
                cur[key] = DECAY[key] * cur[key]
        cur["interstate_recent"] = cur["interstate_recent"] + inter_next
        # 5-year institutional changes, from the simulated levels
        for chg, lvl in INST_CHANGE.items():
            inst_hist[lvl] = torch.cat([inst_hist[lvl][..., 1:], cur[lvl][..., None]], -1)
            cur[chg] = inst_hist[lvl][..., -1] - inst_hist[lvl][..., 0]
        # elections recur at each country's usual interval, only where elections are still held
        elect = (cur["years_since_election"] + 1 >= elec_interval) & torch.isfinite(cur["years_since_election"]) \
            & holds_elections
        cur["election_year"] = torch.where(torch.isfinite(cur["election_year"]), elect.float(), cur["election_year"])
        cur["years_since_election"] = torch.where(elect, torch.zeros_like(cur["years_since_election"]),
                                                  (cur["years_since_election"] + 1).clamp(max=30))
        for k_, (e_, anc_) in party.items():
            x_ = cur[k_]
            # (a value masked at the origin after a leader change V-Party has not coded starts from the country's mean)
            xb_ = torch.where(torch.isfinite(x_), x_, anc_[None].expand_as(x_))
            xn_ = (xb_ + e_["rev"] * (xb_ - anc_) + e_["sd"] * torch.randn(S, C, device=device, generator=gen)).clamp(0, 1)
            cur[k_] = torch.where(elect & torch.isfinite(xb_), xn_, x_)
        # about half of coup attempts succeed (historical share); a success installs a military,
        # unelected leader and restarts the regime clock
        coup_success = (coup_next > 0) & (torch.rand(S, C, device=device, generator=gen) < p_coup_success)
        # regime changes: a successful coup, or crossing the democracy midpoint either way. A coup installs a
        # military, unelected regime that stops holding elections; democratization ends military rule and
        # restores elected government and elections
        democratized = (dem_prev < 0.5) & (dem_next >= 0.5)
        regime_change = coup_success | democratized | ((dem_prev >= 0.5) & (dem_next < 0.5))
        cur["log_regime_duration"] = torch.where(regime_change, torch.zeros_like(cur["log_regime_duration"]),
                                                 torch.log1p(torch.expm1(cur["log_regime_duration"]) + 1))
        cur["regime_military"] = torch.where(coup_success, torch.maximum(cur["regime_military"],
                                                                          torch.full_like(dem_next, 0.75)),
                                             torch.where(democratized, torch.zeros_like(dem_next), cur["regime_military"]))
        cur["elected_regime"] = torch.where(coup_success, torch.zeros_like(dem_next),
                                            torch.where(democratized, torch.ones_like(dem_next), cur["elected_regime"]))
        holds_elections = torch.where(coup_success, torch.zeros_like(holds_elections),
                                      torch.where(democratized, torch.ones_like(holds_elections), holds_elections))
        if has_leaders:
            dbin = (dem_next.nan_to_num(0.5) * 10).long().clamp(0, 9)
            regular = (torch.rand(S, C, device=device, generator=gen) < turnover[dbin]) & ~coup_success
            # death or ill health, by age (on top of the regime-based turnover, which mostly counts political exits)
            a0, a1, a2 = LEADER_DEATH
            p_die = torch.sigmoid(a0 + a1 * cur["leader_age"].nan_to_num(55.0) + a2 * (year - 2000) / 100)
            if WS is not None and WS.tech is not None:
                p_die = p_die * (1 - WS.tech.lever("leader_cut").clamp(max=1))
            died = (torch.rand(S, C, device=device, generator=gen) < p_die) & ~coup_success & ~regular
            succ_t = torch.where(died & (dem_next < 0.5), torch.full_like(succ_t, 2), (succ_t - 1).clamp(min=0))
            regular = regular | died
            new_leader = regular | coup_success
            years_in = torch.expm1(cur["leader_log_tenure"]) / 12 + 1
            cur["leader_log_tenure"] = torch.where(new_leader, torch.full_like(years_in, np.log1p(12.0)),
                                                   torch.log1p(12 * years_in))
            cur["leader_age"] = torch.where(new_leader, torch.full_like(years_in, 55.0), cur["leader_age"] + 1)
            cur["leader_military"] = torch.where(coup_success, torch.ones_like(years_in),
                                                 torch.where(regular, torch.zeros_like(years_in), cur["leader_military"]))
            cur["leader_elected"] = torch.where(coup_success, torch.zeros_like(years_in),
                                                torch.where(regular, (dem_next > 0.5).float(), cur["leader_elected"]))
            cur["months_since_irregular"] = torch.where(coup_success, torch.full_like(years_in, np.log(6)),
                                                        torch.log(torch.exp(cur["months_since_irregular"]) + 12))
            # an electoral loss: a regular change of leader in an election year
            cur["months_since_loss"] = torch.where(regular & elect, torch.full_like(years_in, np.log(6)),
                                                   torch.log(torch.exp(cur["months_since_loss"]) + 12))
            # an election due next year is flagged as anticipated (within ~6 months, REIGN's coding) 30% of the time
            due = (cur["years_since_election"] + 1 >= elec_interval) & holds_elections
            cur["election_anticipated"] = (due & (torch.rand(due.shape, device=device, generator=gen) < 0.3)).float()

        # --- events and aggregates --------------------------------------------------
        # collapse / autocratization: drop vs. the 5-year running peak, counting only years
        # from the origin on (a pre-origin drawdown is not a future crisis)
        lb = max(W - 1 - k, W - 6)
        peak = lambda a: torch.nan_to_num(a[..., lb:], nan=-float("inf")).amax(-1)
        ever["onset"] = torch.maximum(ever["onset"], onset_next)
        ever["coup"] = torch.maximum(ever["coup"], coup_next)
        in_collapse = (lg_next - torch.nan_to_num(h["lg"][..., -6:-1], nan=-float("inf")).amax(-1) < np.log(0.9)).float()
        cur["collapses_total"] = cur["collapses_total"] + torch.clamp(in_collapse - was_collapse, min=0)
        was_collapse = in_collapse
        # economic collapse also counts war damage (a level factor kept outside the growth dynamics)
        lw = lg_next + war_cum + dmg_cum
        ever["collapse"] = torch.maximum(ever["collapse"], (lg_next - peak(h["lg"]) < np.log(0.9)).float())
        peak_w = torch.stack(lw_hist[-5:], -1).nan_to_num(-float("inf")).amax(-1)        # 5-year peak, as for lg
        ever["collapse"] = torch.maximum(ever["collapse"], (lw - peak_w < np.log(0.9)).float())
        lw_hist = lw_hist[-4:] + [lw]
        ever["autoc"] = torch.maximum(ever["autoc"], (dem_next - peak(h["dem"]) <= -0.1).float())
        popk = torch.exp(new["lpop"])                 # (population after simulated deaths, births and migration)
        gdp = torch.exp(lg_next + dmg_cum + war_cum)  # reported GDP includes climate and war damage
        okg = torch.isfinite(gdp) & torch.isfinite(popk)
        world_gdp = (torch.where(okg, gdp, 0) * torch.where(okg, popk, 0)).sum(-1) / torch.where(okg, popk, 0).sum(-1)
        okd = torch.isfinite(dem_next) & torch.isfinite(popk)
        world_dem = (torch.where(okd, dem_next, 0) * torch.where(okd, popk, 0)).sum(-1) / torch.where(okd, popk, 0).sum(-1)
        rec["conflict"].append(conf_next.mean(0))
        for e in ("onset", "collapse", "autoc", "coup"):
            rec[f"ever_{e}"].append(ever[e].mean(0))
        rec["gdp_q"].append(torch.nanquantile(gdp, QS.to(device), dim=0).T)
        rec["dem_q"].append(torch.nanquantile(dem_next, QS.to(device), dim=0).T)
        rec["world_conflict"].append(conf_next.mean(-1))
        rec["region_conflict"].append((conf_next @ onehot) / onehot.sum(0))
        rec["world_gdp"].append(world_gdp)
        rec["world_dem"].append(world_dem)
        rec["onsets"].append(onset_next.sum(-1))
        rec["coups"].append(coup_next.sum(-1))
        if chronicle:
            cw = slice(0, chronicle)
            ev = {"onset": onset_next[cw] > 0, "coup": coup_success[cw], "coup_failed": (coup_next[cw] > 0) & ~coup_success[cw],
                  "dem_drop": (dem_next[cw] - dem_prev[cw]) <= -0.05, "dem_rise": (dem_next[cw] - dem_prev[cw]) >= 0.05,
                  "slump": g[cw] <= -0.08}
            yr = {"year": origin + k, "world_gdp": world_gdp[cw].tolist(), "conflict_share": conf_next[cw].mean(-1).tolist()}
            for kk, m in ev.items():
                yr[kk] = [[p.codes[j] for j in torch.nonzero(m[w]).flatten().tolist()] for w in range(m.shape[0])]
            if WS is not None:
                yr["pandemic"] = WS.rec["pandemic"][-1][cw].tolist() if WS.rec["pandemic"] else None
                yr["oil"] = WS.rec["oil"][-1][cw].tolist() if WS.rec["oil"] else None
                if WS.dy is not None:
                    nw = WS.last_new[cw]
                    yr["new_wars"] = [[(WS.pair_names[j][0], WS.pair_names[j][1], bool(WS.last_war[w, j]))
                                       for j in torch.nonzero(nw[w]).flatten().tolist()] for w in range(nw.shape[0])]
                yr["ai"] = float(WS.rec["ai_adoption_world"][-1][0]) if WS.rec["ai_adoption_world"] else None
            chron.append(yr)
        paths["gdp"].append(gdp[sel].clone())   # (a copy: a slice would keep all worlds alive)
        paths["dem"].append(dem_next[sel].clone())
        paths["conflict"].append(conf_next[sel].clone())
        paths["coup"].append(coup_next[sel].clone())
        paths["pop"].append(popk[sel].clone())
        paths["war_dmg"].append(war_cum[sel].clone())
        paths["clim_dmg"].append(dmg_cum[sel].clone())

    out = {k: torch.stack(v, 1 if k in ("conflict", "ever_onset", "ever_collapse", "ever_autoc", "ever_coup", "gdp_q", "dem_q")
                          else -1).cpu().numpy() for k, v in rec.items()}
    out["paths"] = {k: torch.stack(v, -1).cpu().numpy() for k, v in paths.items()}
    out["years"] = np.arange(origin + 1, end + 1)
    from .dynamics import SPEC as _SLOW_SPEC
    if mc_models is not None:
        out["macro_crises"] = {k: torch.stack(v, -1).cpu().numpy() for k, v in mc_rec.items()}     # [S, T] countries hit
    out["slow_end"] = {k: torch.quantile(cur[k], torch.tensor([.05, .25, .5, .75, .95], device=device), 0).T.cpu().numpy()
                       for k in _SLOW_SPEC if k in cur}
    if WS is not None:
        wr = {k: torch.stack(v, -1).cpu().numpy() for k, v in WS.rec.items() if v}
        if WS.dy is not None:
            wr["watch_names"] = [n for n, _ in WS.watch]
            wr["watch_hits"] = {int(k): v for k, v in (WS.watch_hits or {}).items()}
            wr["blocs"] = {k: float(v.mean()) for k, v in WS.bloc_ever.items()}
            wr["pair_prob"] = WS.pair_any.float().mean(0).cpu().numpy()
            wr["watch_war_hits"] = {int(k): v for k, v in (WS.watch_war_hits or {}).items()}
            wr["watch_war_deaths"] = WS.watch_ep_max.cpu().numpy()          # [S, n_watch] largest war per pair
            wr["blocs_war"] = {k: float(v.mean()) for k, v in WS.bloc_war_ever.items()}
            wr["max_war_deaths"] = WS.max_ep.cpu().numpy()                  # [S] largest single pair-war
        if WS.tech is not None:
            wr["tech_arrival"] = WS.tech.record()
        if WS.params:
            wr["params"] = {k: v[:, 0].cpu().numpy() for k, v in WS.params.items()}
        wr["nuclear_end"] = (WS.nuclear & ~WS.nuclear0).float().mean(0).cpu().numpy()     # [C] share of worlds
        wr["codes"] = list(p.codes)
        out["world"] = wr
    out["gdp0"] = torch.exp(lg0[0]).cpu().numpy()
    if chronicle:
        out["chronicle"] = chron
    return out
