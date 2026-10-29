"""Refugee flows in simulation: who flees simulated wars, and where they go.

Tested first as predictors (2026-10-25, data/raw/migration): net migration, migrant share and its 5-year change,
refugees hosted per person and remittances added +0.004 to the dev score (90% interval -0.001 to +0.009, below
the noise floor), and immigration did not robustly predict polarization or anti-democratic mobilization out of
sample. So migration is not a forecasting feature and is not wired to politics. It enters the simulator as a
mechanism:

- refugees from each country (per person, the model's `refugees_pc`) move toward a target level each year:
  log(r + 1e-5) adjusts by `lam` toward  country effect + b_conf * civil conflict + b_war * war-level conflict,
  estimated in one step within countries on 1975..origin (revised 2026-10-25 after review: the earlier two-step
  version fitted the target statically, so refugees still abroad after a war inflated the peacetime level). (A pooled per-person equation gave India-sized countries with minor
  conflicts tens of millions of refugees: 92M simulated for 2015 from 1995 against 16M recorded. This form, run on
  the recorded conflict paths, gives 15.7M against 15.8M for 2015 from 1995, and 13.9M against 30.5M for 2025 from
  2005: refugees from interstate wars, above all Ukraine, are not modelled.)
- the change in each country's refugee stock since the origin moves people from its population into hosts', shared
  out as UNHCR recorded that country's refugees, asylum seekers and other people in need of protection in
  2015..origin (countries with no record: the shares of their region's other origins). UN population projections
  already contain today's flows, so only departures from the origin level are added.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RAW


def fit_refugees(base, years, origin, start=1975, codes=None, expeditionary=("USA", "GBR", "FRA", "RUS", "AUS", "CAN", "CHN")):
    """{"beta": [b_conf, b_war], "lam": adjustment speed, "fe": [C] country effects (NaN where no data)}

    One-step dynamic fixed-effects estimate of the equation the simulator runs,
        dL[t+1] = lam * (fe + b_conf * conf[t+1] + b_war * war[t+1] - L[t]),
    i.e. dL[t+1] = a_i - lam * L[t] + g_c * conf[t+1] + g_w * war[t+1] within countries (T ~ 50 years, so the
    Nickell bias on lam is small). Years a country exists with no refugee record count as zero refugees (the source
    lists only positive stocks). Interstate conflict on a country's soil counts as conflict, as in the simulator
    (approximated by involvement of a country that is not an expeditionary power)."""
    E = 1e-5
    r = base["refugees_pc"].astype(float)
    conf, war = base["conflict"].astype(float), np.nan_to_num(base["war"])
    exists = np.isfinite(conf)
    # (only for countries the source covers at all: Taiwan and Serbia have no record and get the median level)
    covered = np.isfinite(r[:, np.asarray(years) <= origin]).any(1, keepdims=True)
    r = np.where(covered & exists & ~np.isfinite(r) & (np.asarray(years)[None] >= start), 0.0, r)
    L = np.log(r + E)
    conf = np.nan_to_num(conf)
    if codes is not None and "interstate" in base:
        site = np.nan_to_num(base["interstate"]) * ~np.isin(np.asarray(codes), expeditionary)[:, None]
        conf = np.maximum(conf, site)
    t = np.where((np.asarray(years) >= start) & (np.asarray(years) + 1 <= origin))[0]
    dy, Ll, c1, w1 = L[:, t + 1] - L[:, t], L[:, t], conf[:, t + 1], war[:, t + 1]
    ok = np.isfinite(dy) & np.isfinite(Ll)
    n_i = ok.sum(1)
    with np.errstate(all="ignore"):
        dm = lambda a: np.where(ok, a, np.nan) - np.nanmean(np.where(ok, a, np.nan), 1, keepdims=True)
        X = np.stack([dm(Ll)[ok], dm(c1)[ok], dm(w1)[ok]], 1)
        g = np.linalg.lstsq(X, dm(dy)[ok], rcond=None)[0]
        lam = float(-g[0])
        a_i = np.nanmean(np.where(ok, dy + lam * Ll - g[1] * c1 - g[2] * w1, np.nan), 1)
    fe = np.where(n_i >= 5, a_i / lam, np.nan)
    return {"beta": [float(g[1] / lam), float(g[2] / lam)], "lam": lam, "fe": fe}


def host_shares(codes, regions, origin):
    """[C, C] matrix H[o, d]: share of origin o's displaced people hosted by d (rows sum to <= 1)"""
    ci = {c: i for i, c in enumerate(codes)}
    C = len(codes)
    H = np.zeros((C, C))
    try:
        od = pd.read_csv(RAW / "migration" / "refugees_od.csv")
    except FileNotFoundError:
        return H
    lo = 2015 if origin >= 2015 else max(1960, origin - 10)        # (hindcasts: the decade before the origin)
    od = od[(od.year >= lo) & (od.year <= origin)]
    od["n"] = od[["refugees", "asylum_seekers", "oip"]].fillna(0).sum(1)
    g = od.groupby(["iso3_orig", "iso3_asylum"]).n.sum()
    for (o, d), v in g.items():
        if o in ci and d in ci and o != d:
            H[ci[o], ci[d]] += v
    tot = H.sum(1, keepdims=True)
    H = np.divide(H, tot, out=np.zeros_like(H), where=tot > 0)
    regions = np.asarray(regions)
    for r in np.unique(regions):
        rows = (regions == r) & (tot[:, 0] > 0)
        if rows.any():
            reg = H[rows].mean(0)
            for o in np.where((regions == r) & (tot[:, 0] == 0))[0]:
                H[o] = reg
                H[o, o] = 0
                H[o] /= max(H[o].sum(), 1e-12)
    return H
