"""Climate: observed history (ERA5, 1950-2025; to 2022 before 2026-10-28) and CMIP6 projections (2015-2100) per country,
from the World Bank Climate Change Knowledge Portal.

Projections are bias-corrected with the delta method: the CMIP6 change relative to its own mean over the last ten
observed years is added to the observed ERA5 mean of those years (ratio for precipitation), so modelled and observed
series join without a jump.
"""
from __future__ import annotations

import json

import numpy as np
import requests

from .data import RAW

API = "https://cckpapi.worldbank.org/cckp/v1"
VARS = ["tas", "pr", "cdd", "txx"]          # mean temp, precipitation, consecutive dry days, hottest day
SCENARIOS = ["ssp126", "ssp245", "ssp370", "ssp585"]
SCENARIO_NAMES = {"ssp126": "SSP1-2.6 (strong mitigation)", "ssp245": "SSP2-4.5 (middle of the road)",
                  "ssp370": "SSP3-7.0 (regional rivalry)", "ssp585": "SSP5-8.5 (fossil-fuelled)"}
CLIM = RAW / "climate"


def _get(url, path, refresh=False):
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    path.write_text(r.text)
    return r.json()


def _to_array(d, codes, years):
    out = np.full((len(codes), len(years)), np.nan)
    idx = {c: i for i, c in enumerate(codes)}
    y0 = years[0]
    for code, series in (d.get("data") or {}).items():
        if code not in idx:
            continue
        for k, v in series.items():
            yi = int(k[:4]) - y0
            if 0 <= yi < len(years) and v is not None:
                out[idx[code], yi] = v
    return out


def observed(codes, years, refresh=False):
    """ERA5 annual values per country: {var: [C, Y]}."""
    out = {}
    for v in VARS:
        # (the portal's 1950-2025 release matches the 1950-2022 one for tas, pr and cdd to 0.01; its hottest day, txx,
        # is 5-7 C higher throughout, apparently a new definition, with year-to-year changes correlated 0.90 and
        # anomalies about 20% larger: the whole series is taken from the new release, never spliced)
        url = f"{API}/era5-x0.25_timeseries_{v}_timeseries_annual_1950-2025_mean_historical_era5_x0.25_mean/all_countries?_format=json"
        out[v] = _to_array(_get(url, CLIM / f"era5_{v}_1950-2025.json", refresh), codes, years)
    return out


def projected(codes, years, scenario, stat="median", refresh=False):
    """CMIP6 multi-model ensemble (`stat`: median, p10, p90) per country: {var: [C, Y]}."""
    out = {}
    for v in VARS:
        url = (f"{API}/cmip6-x0.25_timeseries_{v}_timeseries_annual_2015-2100_{stat}_{scenario}_ensemble_all_mean"
               f"/all_countries?_format=json")
        out[v] = _to_array(_get(url, CLIM / f"cmip6_{v}_{scenario}_{stat}.json", refresh), codes, years)
    return out


def scenario_paths(codes, years, scenario, obs, stat="median", refresh=False):
    """observed series continued by delta-corrected CMIP6 projections after the last observed year."""
    proj = projected(codes, years, scenario, stat, refresh)
    yi = lambda y: int(y - years[0])
    out = {}
    for v in VARS:
        o, p = obs[v].copy(), proj[v]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # anchor on the latest ten observed years (the last year observed for most countries), so the
                # paths start where the simulator starts
                last = int(years[np.max(np.where(np.isfinite(o).mean(0) > 0.5)[0])])
                base_obs = np.nanmean(o[:, yi(last - 9):yi(last) + 1], 1, keepdims=True)
                base_mod = np.nanmean(p[:, yi(last - 9):yi(last) + 1], 1, keepdims=True)
        if v == "pr":
            adj = base_obs * p / base_mod                     # precipitation: relative change
        else:
            adj = base_obs + (p - base_mod)
        fut = years > last
        o[:, fut] = adj[:, fut]
        out[v] = o
    return out


BHM_URL = "https://web.stanford.edu/~mburke/climate/data/GDPcap_ClimateChange_RCP85_SSP5.csv"


def popweight_offset(codes, years, obs_tas, refresh=False):
    """population-weighted minus area-weighted mean temperature per country (degC).
    Burke, Hsiang & Miguel (2015) estimate their damage function on population-weighted
    temperature (column `meantemp` of their projection file, a ~1980-2010 baseline); ERA5
    country means are area-weighted, which makes e.g. China or Canada far too cold."""
    import pandas as pd
    path = CLIM / "bhm" / "GDPcap_ClimateChange_RCP85_SSP5.csv"
    if not path.exists() or refresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(requests.get(BHM_URL, timeout=120).content)
    pw = pd.read_csv(path, usecols=["ISO3", "meantemp"]).set_index("ISO3").meantemp
    yi = lambda y: int(y - years[0])
    area = np.nanmean(obs_tas[:, yi(1980):yi(2010) + 1], 1)
    off = np.array([pw.get(c, np.nan) for c in codes]) - area
    return np.nan_to_num(off, nan=0.0)
