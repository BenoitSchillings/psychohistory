"""Technology breakthroughs as per-world scenario switches (assumptions, 2026-10-31).

Each technology has a chance of working at scale by 2040 and by 2065 (best guesses, docs/methodology.md), an adoption
speed, prerequisites, and per-world magnitude ranges on levers the simulator already has. Per world:
  arrival  inverse of a CDF through (2025, 0), (2040, P40), (2065, P65), (2120, 1), linear in between; drawn
           through a shared "science pace" factor (Gaussian copula, correlation SCIENCE_RHO), so breakthroughs cluster
           in fast-progress worlds; once near-free cognition arrives, the remaining wait shrinks by the world's AI boost
  prereqs  a technology cannot arrive before one of its prerequisites (any of the listed ones)
  adoption a logistic from the arrival year, 10% -> 90% in `years`
  levers   magnitude drawn uniformly per world from the ranges below, scaled by adoption
Levers (read by simulate.py and world.py through TechState.lever):
  clean_rate    extra growth rate of the clean-energy S-curve (fraction of its fitted rate)
  clean_cap     extra ceiling of the clean share (share of primary energy)
  energy_gain   extra yearly growth of energy per person in energy-poor countries (closes the gap to the frontier)
  warming_cut   share of warming beyond the origin removed (all climate channels): carbon removal and geoengineering,
                and clean energy beyond the emissions path's own (the path, SSP2-4.5, already assumes some decarbonisation;
                cheaper oil itself lifts no importer's growth: their 1966-2023 response was to price rises only)
  yield_gain    extra yearly growth of cereal yields
  harvest_cut   share of the bad-harvest swing removed
  food_cut      lower food import prices: share of the world grain-price rise removed
  water_cut     share of water losses removed where water binds
  pandemic_cut  share of pandemic output loss and deaths removed
  eng_pandemic  extra yearly chance of an engineered pandemic (also without near-free cognition)
  work_gain     extra yearly growth of the working-age share (healthy years added; 0.1-0.3% a year at full adoption,
                3-9% over 30 years: people work through part, not all, of their extra healthy years)
  leader_cut    share of leaders' yearly mortality removed
  crisis_mult   banking-crisis odds multiplier in the arrival year and the next (one-off shock)
  geo_tension   log-odds added to war onset between rivals in monsoon Asia (unilateral climate intervention)
  space         deployment of cheap launch (drives the cheap-space channels, world.WorldConfig.cheap_space)
"""
from __future__ import annotations

import numpy as np
import torch

SCIENCE_RHO = 0.5          # correlation of arrival dates through the shared science-pace factor (assumption)
AI_BOOST = (0.0, 1.0)      # near-free cognition divides the remaining wait by 1 + this (uniform per world; assumption)

# name: (P by 2040, P by 2065, adoption years 10->90%, prerequisites, {lever: (lo, hi)})
# Probabilities anchored 2026-10-31 on forecaster markets, expert surveys and agency outlooks where they exist (Metaculus
# values from undated snapshots; Manifold live, several with fewer than 30 forecasters): storage LDES Council 2021, Form
# Energy and first-of-a-kind prices; fission IEA "Path to a New Era for Nuclear" 2025 (120 GW of SMRs by 2050 in its
# optimistic case); geothermal IEA "Future of Geothermal" 2024, Fervo; fusion Metaculus first 100 MW plant (median 2041)
# and >0.1% / >10% of energy (2048 / 2072); launch observed Starship progress; space solar Manifold, ESA/Frazer-Nash 2022;
# air capture Sievert et al. Joule 2024, IEA; geoengineering Metaculus 45% in the 21st century; crops HB4 wheat approvals;
# protein Arthur D Little 2026, Manifold; desalination IDA capacity growth; vaccines CEPI 100 Days Mission report 2026
# ("not yet achievable"); pathogens FRI XPT 2022 and FRI 2025 expert panel; longevity Metaculus; quantum Global Risk
# Institute Quantum Threat Timeline 2025 (51-70% within 15 years), Manifold. Green hydrogen and cheap ammonia: no anchors
# sought (judgement). Not included: room-temperature superconductors and mass brain-computer interfaces (no lever).
TECHS = {
    "long_storage":   (0.35, 0.70, 15, (), {"clean_rate": (0.1, 0.4), "clean_cap": (0.03, 0.08), "warming_cut": (0.02, 0.08)}),
    "adv_fission":    (0.15, 0.55, 20, (), {"clean_rate": (0.05, 0.25), "clean_cap": (0.03, 0.10), "energy_gain": (0.002, 0.006), "warming_cut": (0.03, 0.10)}),
    "geothermal":     (0.15, 0.55, 20, (), {"clean_rate": (0.05, 0.15), "clean_cap": (0.02, 0.06), "energy_gain": (0.001, 0.004), "warming_cut": (0.02, 0.06)}),
    "fusion":         (0.02, 0.25, 20, (), {"clean_rate": (0.2, 0.6), "clean_cap": (0.10, 0.25), "energy_gain": (0.005, 0.015), "warming_cut": (0.05, 0.20)}),
    "green_hydrogen": (0.35, 0.65, 15, (), {"clean_rate": (0.05, 0.2), "clean_cap": (0.03, 0.10), "warming_cut": (0.03, 0.10)}),
    "cheap_ammonia":  (0.25, 0.55, 12, (), {"yield_gain": (0.002, 0.006), "food_cut": (0.1, 0.3)}),
    "cheap_launch":   (0.85, 0.95, 12, (), {"space": (1.0, 1.0)}),
    "space_solar":    (0.03, 0.12, 20, ("cheap_launch",), {"clean_cap": (0.03, 0.10), "energy_gain": (0.002, 0.006), "warming_cut": (0.02, 0.06)}),
    "air_capture":    (0.05, 0.25, 25, (), {"warming_cut": (0.1, 0.3)}),
    "geoengineering": (0.12, 0.35, 5, (), {"warming_cut": (0.4, 0.9), "geo_tension": (0.0, 0.6)}),
    "crops":          (0.45, 0.75, 15, (), {"yield_gain": (0.002, 0.008), "harvest_cut": (0.1, 0.4)}),
    "protein":        (0.45, 0.75, 15, (), {"food_cut": (0.05, 0.25)}),
    "desalination":   (0.55, 0.80, 15, (), {"water_cut": (0.2, 0.6)}),
    "vaccines_100d":  (0.40, 0.70, 8, (), {"pandemic_cut": (0.3, 0.7)}),
    "pathogens":      (0.45, 0.75, 10, (), {"eng_pandemic": (0.001, 0.005)}),
    "longevity":      (0.20, 0.50, 20, (), {"work_gain": (0.001, 0.003), "leader_cut": (0.3, 0.6)}),
    "quantum_crypto": (0.65, 0.92, 2, (), {"crisis_mult": (2.0, 5.0)}),
}
ORIGIN_YEAR = 2025
TAIL_END = 2120


def _arrival_from_u(u, p40, p65):
    """year of arrival for uniform draws u (np, any shape). Beyond 2065 the remaining chance is spread over 2065-2120
    (TAIL_END), so that near-free cognition can pull late breakthroughs into the window; arrivals after the
    simulation's end simply never act"""
    y = 2065 + (TAIL_END - 2065) * (u - p65) / max(1 - p65, 1e-9)
    a = u < p40
    y[a] = ORIGIN_YEAR + (2040 - ORIGIN_YEAR) * u[a] / max(p40, 1e-9)
    b = (u >= p40) & (u < p65)
    y[b] = 2040 + (2065 - 2040) * (u[b] - p40) / max(p65 - p40, 1e-9)
    return y


_CAL = {}


def calibrated(n=40000, seed=5):
    """{name: (b40, b65)} no-AI chances chosen so that, averaged over the model's AI arrival distribution
    (world.ARRIVAL_CDF) and the AI boost, the chance by 2040 and by 2065 equals the anchored TECHS values: the anchors
    are unconditional forecasts, which already count AI's chance of speeding science"""
    if _CAL:
        return _CAL
    from .world import ARRIVAL_CDF
    rng = np.random.default_rng(seed)
    ys, ps = map(np.asarray, zip(*ARRIVAL_CDF))
    ua = rng.random(n)
    ai = np.where(ua <= ps[-1], np.interp(ua, ps, ys), np.inf)
    boost = rng.uniform(*AI_BOOST, n)
    u = rng.random(n)

    def share(b40, b65, year):
        y = _arrival_from_u(u, b40, b65)
        late = y > ai
        y[late] = ai[late] + (y[late] - ai[late]) / (1 + boost[late])
        return float((y <= year).mean())

    def solve(f_, target):
        lo, hi = 1e-4, target
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f_(mid) < target else (lo, mid)
        return (lo + hi) / 2
    for name, (p40, p65, *_r) in TECHS.items():
        b40 = solve(lambda b: share(b, max(b + 1e-3, p65), 2040), p40)
        b65 = solve(lambda b: share(b40, max(b, b40 + 1e-3), 2065), p65)
        _CAL[name] = (b40, max(b65, b40 + 1e-3))
    return _CAL


class TechState:
    """per-world arrival years, adoption and lever magnitudes; `force` = {name: year or None} fixes an arrival
    (None: never) in every world, for paired with/without comparisons"""

    def __init__(self, S, device, seed, ai_arrival=None, techs=None, force=None):
        from scipy.stats import norm
        self.S, self.dev = S, device
        self.techs = dict(TECHS if techs is None else techs)
        rng = np.random.default_rng(seed + 709)
        z_world = rng.standard_normal(S)
        boost = rng.uniform(*AI_BOOST, S)
        ai = np.full(S, np.inf) if ai_arrival is None else np.asarray(ai_arrival, float).reshape(-1) * np.ones(S)
        self.arrival, self.mag = {}, {}
        for name, (p40, p65, yrs, pre, levers) in TECHS.items():
            # every technology consumes its draws in a fixed order, so forcing or dropping one leaves the others alone
            e = rng.standard_normal(S)
            mags = {lv: rng.uniform(lo, hi, S) for lv, (lo, hi) in levers.items()}
            if name not in self.techs:
                continue
            u = norm.cdf(SCIENCE_RHO * z_world + np.sqrt(1 - SCIENCE_RHO ** 2) * e)
            b40, b65 = calibrated()[name]                 # no-AI chances (the anchors are AI-inclusive)
            y = _arrival_from_u(u, b40, b65)
            # near-free cognition shortens the remaining wait
            late = np.isfinite(y) & (y > ai)
            y[late] = ai[late] + (y[late] - ai[late]) / (1 + boost[late])
            self.arrival[name], self.mag[name] = y, mags
        for name, (_, _, _, pre, _) in TECHS.items():
            if name in self.arrival and pre:
                first = np.min([self.arrival.get(q, np.full(S, np.inf)) for q in pre], 0)
                self.arrival[name] = np.maximum(self.arrival[name], first)
        for name, yr in (force or {}).items():
            if name in self.arrival:
                self.arrival[name] = np.full(S, np.inf if yr is None else float(yr))
        self._t = {k: torch.as_tensor(v, dtype=torch.float32, device=device)[:, None] for k, v in self.arrival.items()}
        self._m = {k: {lv: torch.as_tensor(m, dtype=torch.float32, device=device)[:, None] for lv, m in d.items()}
                   for k, d in self.mag.items()}
        self.year = ORIGIN_YEAR
        self._cache = {}

    def adoption(self, name, year=None):
        """[S, 1] share adopted (logistic, 10% -> 90% in the technology's adoption years after arrival)"""
        y = self.year if year is None else year
        yrs = TECHS[name][2]
        t = self._t[name]
        a = torch.sigmoid((y - t - yrs / 2) * (2 * np.log(9) / yrs))
        return torch.where(torch.isfinite(t) & (y >= t), a, torch.zeros_like(a))

    def set_year(self, year):
        self.year, self._cache = year, {}

    def lever(self, lv):
        """[S, 1] sum over technologies of magnitude x adoption for this lever (0 when no technology has it)"""
        if lv not in self._cache:
            out = torch.zeros(self.S, 1, device=self.dev)
            for name, d in self._m.items():
                if lv in d:
                    out = out + d[lv] * self.adoption(name)
            self._cache[lv] = out
        return self._cache[lv]

    def arrived_in(self, name, year):
        """[S, 1] 1 where the technology arrives during this year (one-off shocks)"""
        t = self._t[name]
        return ((t > year - 1) & (t <= year)).float()

    def record(self):
        return {k: v.astype(np.float32) for k, v in self.arrival.items()}
