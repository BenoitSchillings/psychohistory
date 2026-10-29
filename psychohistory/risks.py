"""Further world-system mechanisms added 2026-10-26 after a review of what the model ignored (independent review, outputs/
agy_review_missing_factors.md; docs/world_system.md "Missing factors"). Data builders and constants here; the yearly
steps live in world.World (proliferation, nuclear use, sanctions, alliances) and simulate.simulate (the rest).

Estimated on history (scripts/extra_estimates.py, outputs/extra_estimates.json, and the civil-war check below):
- internationalized civil wars (UCDP/PRIO type 4: an outside state fights on a side) end less often: -0.42 (se 0.20)
  log-odds on next-year termination, 1946-2023, controlling duration, intensity and era (1,747 conflict-years, 23%
  internationalized). The country models already see the average war, so the simulator applies -0.42 x (intl - 0.23)
- cereal yields fall 1.0% per degC of hottest-day anomaly (se 0.25; 1.4% (se 0.35) on the portal's earlier hottest-day series,
  whose anomalies were about 20% smaller; re-estimated 2026-10-28) and rise 5.6% per unit log-precipitation anomaly
  (se 1.6), 1962-2022 with country and year effects; temperature anomalies are shared across countries (mean
  pairwise correlation 0.17 for annual mean, 0.10 for the hottest day, 0.04 rain), so weather noise is drawn from
  the observed cross-country correlation (15 factors + idiosyncratic) rather than independently
- fertility falls with income only slightly beyond what UN projections assume: -0.035 (se 0.021) in log fertility per
  unit 10-year change in log GDP per person
Tested and not adopted: public debt as a financial-crisis predictor (+0.08, se 0.08; growth and inflation matter;
Brier worse out of sample 2000-2024), ageing and war as drivers of debt (out-of-sample R^2 0.132 -> 0.129), mineral
rents as a civil-conflict predictor (coef -0.09, se 0.07; GBM dev score 0.9035 -> 0.9036), alliance erosion with the
regime gap (wrong sign, see world.WorldConfig).

Assumptions (ranges in world.EXTRA_RANGES): proliferation and the preventive-war window, nuclear use, sanctions,
the critical-minerals windfall and China supply chokepoint, war and pandemic deaths, sea-level rise and storms, solar
storms, large eruptions and wartime cyber attacks.
"""
from __future__ import annotations

import functools
import json
import zipfile

import numpy as np
import pandas as pd

from .data import RAW

# estimates (see the docstring)
INTL_TERM = -0.42
INTL_SHARE = 0.227
YIELD_TXX = -0.0103
YIELD_LOGPR = 0.056
FERT_INCOME = -0.035
CBR_PER_TFR = 0.0071         # crude birth rate per child of total fertility (rough: ~35 fertile years x ~25% women)

# countries outside the nuclear club with the means to build a weapon within a few years (Fuhrmann & Tkach 2015
# latency; assumption) and their yearly chance of starting a weapons program, before multipliers. Iran starts in
# pursuit in a share of worlds (its program was struck in 2025 but its enrichment knowledge remains).
LATENT = {"IRN": 0.015, "SAU": 0.004, "KOR": 0.004, "TUR": 0.002, "JPN": 0.001, "EGY": 0.001, "POL": 0.001,
          "TWN": 0.001, "UKR": 0.002, "DEU": 0.0005, "BRA": 0.0005}
# the West for sanctions and the clean-tech chokepoint: NATO, the EU, US treaty allies in Asia-Pacific, Western-backed
# Taiwan, Israel and Ukraine
WEST = {"USA", "CAN", "GBR", "FRA", "DEU", "ITA", "ESP", "PRT", "NLD", "BEL", "LUX", "DNK", "NOR", "ISL", "SWE", "FIN",
        "POL", "CZE", "SVK", "HUN", "SVN", "HRV", "AUT", "IRL", "GRC", "ROU", "BGR", "EST", "LVA", "LTU", "CHE", "ALB",
        "MNE", "MKD", "TUR", "JPN", "KOR", "AUS", "NZL", "TWN", "ISR", "UKR"}


def _iso2to3():
    iso = pd.read_csv(RAW / "iso3166.csv", dtype=str, keep_default_na=False)
    return dict(zip(iso["alpha-2"], iso["alpha-3"]))


def distances(codes):
    c = pd.read_csv(RAW / "country_centroids.csv", keep_default_na=False)
    c = c[c.COUNTRY == c.COUNTRYAFF].drop_duplicates("ISO", keep="first")
    c["iso3"] = c.ISO.map(_iso2to3())
    pos = {r.iso3: (r.latitude, r.longitude) for r in c.itertuples() if isinstance(r.iso3, str)}
    pos.update({"HKG": (22.3, 114.2), "TWN": (23.7, 121.0)})
    lat = np.radians([pos.get(k, (np.nan, np.nan))[0] for k in codes])
    lon = np.radians([pos.get(k, (np.nan, np.nan))[1] for k in codes])
    dlat, dlon = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def trade_weights(codes, gdp):
    """each country's trade by partner, as shares (rows sum to 1): exports plus imports, 2015-2021 mean, from BACI
    via the CEPII Gravity database V202211 (data/raw/trade/bilateral_2015_2021.csv). Countries without trade data
    fall back to gravity shares, GDP_j / distance_ij (Head & Mayer 2014)."""
    d = np.nan_to_num(distances(codes), nan=2e4)
    w = np.nan_to_num(gdp)[None, :] / np.clip(d, 500, None)
    np.fill_diagonal(w, 0.0)
    grav = w / np.clip(w.sum(1, keepdims=True), 1e-12, None)
    f = RAW / "trade" / "bilateral_2015_2021.csv"
    if not f.exists():
        return grav
    t = pd.read_csv(f)
    ci = {c: i for i, c in enumerate(codes)}
    T = np.zeros((len(codes), len(codes)))
    for o, d_, v in zip(t.iso3_o, t.iso3_d, t.flow):
        if o in ci and d_ in ci and o != d_:
            T[ci[o], ci[d_]] += v
            T[ci[d_], ci[o]] += v
    tot = T.sum(1, keepdims=True)
    real = T / np.clip(tot, 1e-12, None)
    return np.where(tot > 0, real, grav)


def _wb(ind, codes, years=5, origin=9999):
    """latest `years`-year mean, up to the origin, of a World Bank indicator (data/raw/wb/<ind>.json)"""
    j = json.load(open(RAW / "wb" / f"{ind}.json"))[1]
    w = pd.DataFrame([(r["countryiso3code"], int(r["date"]), r["value"]) for r in j if r["value"] is not None],
                     columns=["code", "year", "v"])
    w = w[w.year <= origin]
    w = w[w.year > w.groupby("code").year.transform("max") - years]
    m = w.groupby("code").v.mean()
    return np.array([m.get(c, np.nan) for c in codes], float)


def mineral_rents(codes, origin=9999):
    return np.nan_to_num(_wb("NY.GDP.MINR.RT.ZS", codes, origin=origin)) / 100


def low_elevation(codes):
    """share of population living below 5 m elevation (World Bank EN.POP.EL5M.ZS, CIESIN; 1990-2015 estimates)"""
    return np.nan_to_num(_wb("EN.POP.EL5M.ZS", codes, years=30)) / 100


@functools.lru_cache(maxsize=4)
def intl_at(origin, codes):
    """iso3 of countries whose civil conflict was internationalized (UCDP/PRIO type 4) in the last year <= origin"""
    from .interstate import _ccode_to_iso
    from .data import UCDP_FILE
    z = zipfile.ZipFile(RAW / UCDP_FILE)
    a = pd.read_csv(z.open(z.namelist()[0]))
    a = a[(a.type_of_conflict == 4) & (a.year <= origin)]
    a = a[a.year == a.year.max()]
    locs = {int(str(x).split(",")[0]) for x in a.gwno_loc}
    c2i = _ccode_to_iso(set(codes))
    return {c2i[c] for c in locs if c in c2i}


def upstream_precip(codes, pr=None):
    """[C, C] row-stochastic weights: the precipitation a country's water depends on. A country's population share in
    each international basin (TFDD) is spread over the basin's countries by their area in it (the Nile's water falls
    mostly in Ethiopia and the Great Lakes, Iraq's in Turkey); the rest of its water is its own rain. `pr`: each
    country's mean precipitation (mm), so that wet upstream areas weigh more than dry ones (area x rain)."""
    prc = {c: (pr[i] if pr is not None and np.isfinite(pr[i]) else 1.0) for i, c in enumerate(codes)}
    from . import water as W_
    b = W_.bcu()
    pop = W_._pop2020()
    ci = {c: i for i, c in enumerate(codes)}
    M = np.zeros((len(codes), len(codes)))
    for basin, g in b.groupby("Basin_Name"):
        area = g.groupby("iso3").Area_km2.sum()
        area = area[[c in ci for c in area.index]]
        area = area * np.array([prc[c] for c in area.index])
        if len(area) < 2 or area.sum() <= 0:
            continue
        for c, pp in g.groupby("iso3").Pop_2020.sum().items():
            if c not in ci or not np.isfinite(pop.get(c, np.nan)):
                continue
            s = min(pp / pop[c], 1.0)
            for j, a_ in area.items():
                M[ci[c], ci[j]] += s * a_ / area.sum()
    tot = M.sum(1)
    scale = np.where(tot > 1, 1 / np.clip(tot, 1e-9, None), 1.0)
    M = M * scale[:, None]
    M[np.arange(len(codes)), np.arange(len(codes))] += 1 - M.sum(1)
    return M


def weather_factors(p, origin, k=5):
    """per climate variable, loadings F [C, k] and idiosyncratic sd share u [C] such that F F' + diag(u^2)
    approximates the cross-country correlation of yearly anomalies (against each country's previous 10 years),
    1980..origin-1 ERA5. Only ~40 years for 175 countries: the leading k factors are shrunk so that the mean
    off-diagonal correlation they imply equals the observed one (sampling noise would otherwise pass for structure)."""
    years = np.asarray(p.years)
    t = (years >= 1980) & (years < min(origin, 2023))
    out = {}
    for v in ("tas", "pr", "cdd", "txx"):
        a = p.x[f"c_{v}"].astype(float)
        if v == "pr":
            a = np.log(np.clip(a, 1, None))
        ref = pd.DataFrame(a).T.rolling(10, min_periods=5).mean().shift(1).T.to_numpy()
        A = (a - ref)[:, t]
        ok = np.isfinite(A).all(1) & (np.nanstd(A, 1) > 1e-8) if t.sum() else np.zeros(a.shape[0], bool)
        F = np.zeros((a.shape[0], k))
        u = np.ones(a.shape[0])
        # (too few years before an early hindcast origin: independent shocks, as before)
        if t.sum() >= 15 and ok.sum() > k + 5:
            R = np.corrcoef(A[ok])
            ev, V = np.linalg.eigh(R)
            idx = np.argsort(ev)[::-1][:k]
            L = V[:, idx] * np.sqrt(np.clip(ev[idx], 0, None))
            n = ok.sum()
            target = (R.sum() - n) / (n * (n - 1))
            FF = L @ L.T
            implied = (FF.sum() - np.trace(FF)) / (n * (n - 1))
            L = L * np.sqrt(np.clip(target / implied, 0, 1)) if implied > 0 else L * 0
            L = np.nan_to_num(L)
            share = np.clip((L ** 2).sum(1), 0, 0.95)
            F[ok] = L
            u[ok] = np.sqrt(1 - share)
        out[v] = (F, u)
    return out


def slr(year, origin=2025):
    """global mean sea-level rise since the origin, metres (IPCC AR6 SSP2-4.5 median shape: ~0.13 m 2020-2050, ~0.5 m
    by 2100 relative to 2020)"""
    t = max(year - origin, 0)
    return 0.0045 * t + 0.00006 * t * t
