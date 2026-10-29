"""Innovation in the simulator (scripts/innovation_test.py, outputs/innovation_test.json).

Catch-up: with the growth controls and year effects (1996-2018 origins, 5-year growth of GDP per person), countries
doing more science than their income predicts grow faster: +0.23 points a year per log unit of scientific and
technical articles per million people (s.e. 0.11), out-of-sample error 1.7% lower (fit to 2007, scored 2008-2018);
R&D spending, researchers and patents point the same way with smaller out-of-sample gains. The simulator adds the
term for science beyond what income predicts, held at each country's origin level (science capacity moves slowly).

The frontier: US research productivity (TFP growth per effective researcher, Bloom, Jones, Van Reenen & Webb's
measure) has fallen 2.4% a year since 1950. A semi-endogenous fit makes extra research effort raise frontier growth
for a time, then fade as ideas get harder to find (x10 effective researchers: +1.8 points at the peak, half of it gone
25 years later) - but the fade is poorly pinned down by 8 decades. So the AI growth dividend now fades with cumulative
adoption at a sampled rate dividend_fade (0 = no fading, as in Romer-style growth; 0.04 = faster than the fit).
Fast spread of internet and mobile phones did not predict protest, coups, war onset or autocratization (rejected).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RAW

ARTICLES_BETA = 0.0023


def article_capacity(codes, years, lpop, lg, origin):
    """science beyond income: log(1 + articles per million people) minus what income predicts (a quadratic in log
    GDP per person fitted on 1996-2019 country-years), [C] at the origin (mean of the last 3 reported years). The growth
    equation already carries the part of science that goes with income (through the gap to the frontier)."""
    w = pd.read_csv(RAW / "innovation" / "wdi_innovation_cy.csv").set_index(["iso3", "year"]).sci_tech_articles
    yi = {y: j for j, y in enumerate(years)}
    hist, now = [], []
    for i, c in enumerate(codes):
        vals = []
        for y in range(1996, origin + 1):
            v = w.get((c, y), np.nan)
            if np.isfinite(v) and y in yi and np.isfinite(lpop[i, yi[y]]) and np.isfinite(lg[i, yi[y]]):
                x = (np.log1p(v / np.exp(lpop[i, yi[y]]) * 1e6), lg[i, yi[y]])
                if y <= 2019:
                    hist.append(x)
                vals.append(x)
        now.append(vals[-3:])
    h = np.asarray(hist)
    if len(h) < 50:                      # origins before the WDI series (hindcasts from the 1990s): no term
        return np.zeros(len(codes))
    coef = np.polyfit(h[:, 1], h[:, 0], 2)
    out = []
    for v in now:
        out.append(np.mean([a - np.polyval(coef, g) for a, g in v]) if v else 0.0)
    return np.asarray(out)
