"""How deadly simulated interstate wars are: battle deaths per war-year, drawn from history.

Data: Correlates of War Inter-State War Data v4.0 (95 wars, 1816-2007; battle deaths per participant) and UCDP
Battle-Related Deaths v25.1 (conflict-years 1989-2024, for wars after COW ends and for minor clashes).

Unit: battle deaths per calendar year a pair is at war level (a war's total divided by the calendar years it spans,
as the simulator counts war years). Each simulated war draws its yearly rate when it starts and keeps it while it
lasts (wars have a characteristic scale); a year of minor fighting (25-999 deaths) draws from UCDP's minor
interstate conflict-years.

Two pools, because size depends on who fights far more than on anything else in the data:
- wars between two major powers (COW, 1848-1953: 12 wars including both world wars and Korea; median about 55,000
  battle deaths a year);
- all other wars (COW 1816-2007 plus Russia-Ukraine from UCDP; median about 4,000 a year).
Tested and not used: once major-versus-major is known, neither the belligerents' share of world GDP (+0.09 +/- 0.09
in log10 deaths per unit of log10 share) nor their combined military capability (COW CINC) nor the post-1945 era
(-0.05 +/- 0.14) adds much (R^2 0.25 -> 0.26).
Draws are a smoothed bootstrap on the log scale (bandwidth 0.2 in log10, so about +/-60%), which lets a war exceed the
largest one on record, capped at 3 million a year (above the Second World War's 2.4 million). Nuclear use is outside
this data and not modelled. Battle deaths exclude civilian and indirect deaths, which are often several times larger.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from .data import ROOT

RAW = ROOT / "data" / "raw"
BW = 0.2                    # smoothing bandwidth, log10 units
CAP = 3e6
FLOOR = 300.0


def _is_major(c, y):
    """COW major-power status (codes: USA 2, UK 200, France 220, Russia 365, China 710, Germany 255, Japan 740,
    Italy 325, Austria-Hungary 300)."""
    return (c in (200, 220, 365)) or (c == 2 and y >= 1898) or (c == 710 and y >= 1950) or \
        (c == 740 and 1895 <= y <= 1945) or (c in (255, 325) and y <= 1945) or (c == 300 and y <= 1918)


@functools.lru_cache(maxsize=4)
def pools(origin: int):
    """per-war-year death rates known before `origin`: {"major": [...], "other": [...], "minor": [...],
    "known": {(a, b): last observed war-year deaths}}."""
    w = pd.read_csv(RAW / "wars" / "cow_interstate_v4.csv")
    w["BatDeath"] = w.BatDeath.where(w.BatDeath >= 0)
    rows = []
    for _, s in w.groupby("WarNum"):
        y0 = s.StartYear1[s.StartYear1 > 0].min()
        ends = pd.concat([s.EndYear1, s.EndYear2])
        y1 = ends[ends > 0].max()
        sides = [any(_is_major(c, y0) for c in s.ccode[s.Side == sd]) for sd in s.Side.unique()]
        rows.append({"start": y0, "end": y1, "rate": s.BatDeath.sum() / (y1 - y0 + 1), "mm": sum(sides) >= 2})
    g = pd.DataFrame(rows)
    g = g[(g.end < origin) & (g.rate > 0)]
    major = g[g.mm].rate.to_numpy(float)
    other = g[~g.mm].rate.to_numpy(float)
    b = pd.read_csv(RAW / "ucdp" / "BattleDeaths_v25_1_conf.csv")
    b = b[b.type_of_conflict.astype(str).str.contains("2") & (b.year < origin)]
    minor = b[b.bd_best < 1000].bd_best.to_numpy(float)
    # wars after COW ends (2007): UCDP war-level episodes, one rate per conflict (mean over its war years)
    late = b[(b.bd_best >= 1000) & (b.year > 2007)].groupby("conflict_id").bd_best.mean().to_numpy(float)
    other = np.concatenate([other, late])
    known = {}
    last = b[b.year == b.year.max()]
    if (last.conflict_id == 13243).any():                 # Russia-Ukraine
        known[("RUS", "UKR")] = float(last[last.conflict_id == 13243].bd_best.iloc[0])
    # wars at war level in the origin year itself (UCDP ACD 26.1 dyads, 2025: e.g. Russia-Ukraine at its 2025 rate,
    # Iran-Israel), which BattleDeaths v25.1 does not reach yet (code review 2026-10-28)
    f = RAW / "outcomes" / "ucdp_2025_interstate_dy.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[(d.year == origin) & (d.war_level == 1) & (d.conflict_deaths_best > 0)]
        for r in d.itertuples():
            known[(r.iso3_a, r.iso3_b)] = float(r.conflict_deaths_best)
    return {"major": major, "other": other, "minor": minor if len(minor) else np.array([100.0]), "known": known}


def summary(origin: int = 2025):
    P = pools(origin)
    return {k: {"n": int(len(v)), "median": float(np.median(v)), "max": float(np.max(v))}
            for k, v in P.items() if k != "known"}
