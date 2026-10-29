"""Ocean food: wild-catch fisheries under warming and overfishing, as a sampled assumption on food and income.

Tested first (scripts/ocean_test.py, outputs/ocean_test.json):
- country level, 1965-2023, logit with year effects: fish dependence (fish share of protein) and five-year falls in
  national catch, alone or weighted by dependence, did not predict civil-conflict onset or economic collapse (no
  coefficient significant; out-of-sample Brier unchanged to the 4th decimal);
- pairs of states, International Fishery Conflict Database 1974-2016 (Spijkers et al. 2019): past fisheries conflicts
  did not improve the interstate-conflict, fatal-MID or 1-year onset backtests; a small gain for uses of force (Brier
  skill 0.219 -> 0.224, 90% interval touching zero), an outcome the simulator does not use.
So fish enters only through what is near-mechanical: wild catch moves with ocean warming (by latitude band) and a
global overfishing / management trend, and that moves calories and income in proportion to each country's dependence.
Ranges: catch potential per degree of warming from Cheung et al. (2016, 2018) and IPCC AR6 WG2 ch. 3 (tropics fall,
high latitudes gain); the global trend from FAO SOFIA 2024 (wild catch flat since the 1990s, 38% of stocks overfished).
Aquaculture (over half of fish eaten) is taken as unaffected: only the wild share of supply moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RAW

OC = RAW / "ocean"
KCAL_PER_KG = 600       # fish and seafood, live-weight equivalent (world: 20.7 kg a year ~ 34 kcal a day)
USD_PER_T = 1500        # first-sale value of wild catch, world average order of magnitude


def _owid(name):
    d = pd.read_csv(OC / f"{name}.csv")
    return d.rename(columns={d.columns[3]: "v"}).dropna(subset=["code"]).set_index(["code", "year"]).v


def _last(s, c, year, back=10):
    for y in range(year, year - back, -1):
        v = s.get((c, y))
        if v is not None and np.isfinite(v):
            return float(v)
    return np.nan


def inputs(codes, base, years, origin):
    """per country: fish share of calories, wild share of fish production, wild catch value / GDP, latitude band"""
    from .risks import _iso2to3
    fish, capt, aqua = _owid("fish-and-seafood-consumption-per-capita"), _owid("capture-fishery-production"), \
        _owid("aquaculture-farmed-fish-production")
    t0 = list(years).index(origin)
    c = pd.read_csv(RAW / "country_centroids.csv", keep_default_na=False)
    c = c[c.COUNTRY == c.COUNTRYAFF].drop_duplicates("ISO", keep="first")
    lat = dict(zip(c.ISO.map(_iso2to3()), c.latitude))
    lat.update({"HKG": 22.3, "TWN": 23.7})
    out = {"cal": [], "wild": [], "gdp": [], "band": []}
    for i, k in enumerate(codes):
        kcal = base["calories"][i, t0] if np.isfinite(base["calories"][i, t0]) else 2800.0
        f = _last(fish, k, origin)
        out["cal"].append(0.0 if not np.isfinite(f) else min(f * KCAL_PER_KG / 365 / kcal, 0.25))
        cp, aq = _last(capt, k, origin), _last(aqua, k, origin)
        out["wild"].append(1.0 if not np.isfinite(aq) or not np.isfinite(cp) or cp + aq <= 0 else cp / (cp + aq))
        gdp = np.exp(base["lg"][i, t0] + base["lpop"][i, t0]) if np.isfinite(base["lg"][i, t0]) else np.nan
        out["gdp"].append(0.0 if not np.isfinite(cp) or not np.isfinite(gdp) else min(cp * USD_PER_T / gdp, 0.2))
        la = abs(float(lat.get(k, 30.0)))
        out["band"].append(0 if la < 23.5 else (1 if la < 50 else 2))
    return {k: np.asarray(v, float) for k, v in out.items()}
