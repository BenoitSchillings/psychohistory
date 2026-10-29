"""Rebel fragmentation, short-term external debt and coups among language / colonial peers (scripts/structure_test.py,
outputs/structure_test.json; data in data/raw/structure and data/raw/trade).

Kept (year effects, controls, clustered by country):
- civil wars with more rebel groups end later: -1.93 log-odds of ending next year per log(1 + rebel groups) (s.e. 0.41;
  UCDP dyadic 1950-2022). The simulator keeps each ongoing war's 2024 count; a new war starts with one group.
  Confirmed outside state support for rebels also lengthens wars (-0.65, s.e. 0.25) but its data stop in 2017: unused.
- short-term external debt relative to reserves raises the chance of a banking crisis starting next year: +0.60 per
  log(1 + ratio) (s.e. 0.11; WDI IDS, low- and middle-income countries 1975-2022); held at each country's latest value,
  relative to the 1975-2022 mean (high-income countries, not covered, at the mean). The US real-rate interaction was
  not significant.
- a coup attempt in a country sharing an official language or a colonial tie (CEPII Gravity) raises coup attempts next
  year, +2.6 log-odds per unit share of peers with a coup (s.e. 0.86), beyond coups within 1,500 km.
Rejected: sanctions (TIES, IST: no growth or collapse effect), term-limit evasion attempts, economic complexity
(significant in sample, no out-of-sample gain), coup-proofing (counterweights: no effect on attempts, success or onset).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RAW

ST = RAW / "structure"
FRAG_MEAN = 0.936        # mean log(1 + rebel groups) in conflict years, 1950-2022
# rebel groups multiply as a war ages (UCDP 1950-2022, by years since onset: 0.77 in a war's first year, 0.90 at 3-5
# years, 1.00 at 12-20, 1.20 beyond): a war starts at FRAG_NEW and moves toward FRAG_LONG by 1/FRAG_TAU of the gap a
# year (least-squares fit of the age profile)
FRAG_NEW, FRAG_LONG, FRAG_TAU = 0.769, 1.218, 11.7
ST_MEAN = 0.496          # mean log(1 + short-term debt / reserves), 1975-2022


def inputs(codes, origin):
    fr = pd.read_csv(ST / "ucdp_rebel_fragmentation_country_year.csv")
    fr = fr[(fr.year <= origin) & (fr.year >= origin - 1)].sort_values("year").groupby("iso3").n_rebel_groups.last()
    lreb = np.array([np.log1p(float(fr.get(c, 1))) for c in codes])
    wd = pd.read_csv(ST / "wdi_external_debt_country_year.csv")
    wd = wd[(wd.year <= origin) & (wd.year >= origin - 5)].dropna(subset=["st_debt_pct_reserves"])
    wd = wd.sort_values("year").groupby("iso3").st_debt_pct_reserves.last()
    st = np.array([np.log1p(min(float(wd[c]), 1000) / 100) - ST_MEAN if c in wd.index else 0.0 for c in codes])
    peers = pd.read_csv(RAW / "trade" / "peer_pairs_2015.csv")
    ci = {c: i for i, c in enumerate(codes)}
    W = np.zeros((len(codes), len(codes)))
    for r in peers.itertuples():
        if r.a in ci and r.b in ci and r.a != r.b:
            W[ci[r.a], ci[r.b]] = 1
    W = W / np.maximum(W.sum(1, keepdims=True), 1)
    return {"lreb": lreb, "st": st, "peers": W}
