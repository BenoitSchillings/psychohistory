"""Long cycles in world history: what Seshat's CrisisDB says about how often and when states break down.

Data (data/raw/seshat/README.md): 169 crisis episodes in 153 polities, 2200 BCE - 1942 CE (CrisisDB, Hoyer,
Turchin et al. 2023), and the polity-century panel of population, territory and hierarchy.

Questions, each a check on the structural-demographic ("secular cycle") theory behind psychohistory:
  1. Are crises within a region periodic (secular cycles of ~200-300 years) or memoryless? The coefficient
     of variation of the gaps between crisis onsets is ~1 for a memoryless (Poisson) process and well
     below 1 for a periodic one.
  2. Does the hazard of crisis rise with a polity's age (a polity "ages into" crisis), as the theory says?
  3. Does population growth over the previous century raise the chance of crisis in the next one?
  4. How severe are crises, and how often do they end the polity?
The per-crisis structural drivers (elite overproduction, fiscal distress) are not publicly released, so
this module can test timing and population pressure only.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import RAW, ROOT

S = RAW / "seshat"


def run(verbose=True, n_perm=2000):
    c = pd.read_csv(S / "crises.csv")
    pol = pd.read_csv(S / "polities.csv")
    pc = pd.read_csv(S / "panel_polity_century.csv")
    out = {"n_crises": int(len(c)), "n_polities": int(c.polity_id.nunique()),
           "span": [int(c.start.min()), int(c.start.max())]}
    # 1. periodicity: gaps between onsets within each region (NGA)
    c = c.sort_values(["home_nga", "start"])
    gaps = c.groupby("home_nga").start.diff().dropna()
    gaps = gaps[gaps > 0]
    cv = float(gaps.std() / gaps.mean())
    # reference: same number of onsets per region placed uniformly at random over the region's span
    rng = np.random.default_rng(0)
    sims = []
    for _ in range(n_perm):
        g_all = []
        for nga, g in c.groupby("home_nga"):
            if len(g) < 2:
                continue
            t = np.sort(rng.uniform(g.start.min(), g.start.max(), len(g)))
            g_all += list(np.diff(t))
        g_all = np.array(g_all)
        sims.append(g_all.std() / g_all.mean())
    out["gaps"] = {"n": int(len(gaps)), "mean": float(gaps.mean()), "median": float(gaps.median()),
                   "q10": float(gaps.quantile(.1)), "q90": float(gaps.quantile(.9)), "cv": cv,
                   "cv_random_mean": float(np.mean(sims)), "p_periodic": float(np.mean(np.array(sims) <= cv))}
    # 2. hazard by polity age (crises per polity-century of exposure), polities covered by CrisisDB
    cov = set(pc.loc[pc.crisisdb_coded_polity == 1, "polity_id"])
    pp = pol[pol.polity_id.isin(cov)].dropna(subset=["start_year", "end_year"])
    bins = [0, 50, 100, 150, 200, 300, 500, 2000]
    expo = np.zeros(len(bins) - 1)
    for s_, e_ in zip(pp.start_year, pp.end_year):
        life = max(e_ - s_, 0)
        for i in range(len(bins) - 1):
            expo[i] += max(0, min(life, bins[i + 1]) - bins[i])
    cc = c[c.polity_id.isin(cov)]
    age = (cc.start - cc.polity_start).clip(lower=0)
    events = np.histogram(age, bins=bins)[0]
    out["hazard_by_age"] = [{"age": f"{bins[i]}-{bins[i + 1]}", "crises": int(events[i]), "polity_years": float(expo[i]),
                             "per_century": float(100 * events[i] / expo[i]) if expo[i] > 0 else None} for i in range(len(bins) - 1)]
    # 3. population growth over the previous century vs crisis onset in this one (CrisisDB-covered polities)
    q = pc[pc.crisisdb_coded_polity == 1].sort_values(["polity_id", "year"]).copy()
    q["growth"] = np.log(q["pop"]) - np.log(q.groupby("polity_id")["pop"].shift(1))
    q = q[q.pop_src == "interp"] if (q.pop_src == "interp").sum() > 50 else q
    q = q.dropna(subset=["growth"])
    if len(q) > 20:
        hi = q.growth > q.growth.median()
        out["pop_growth"] = {"n": int(len(q)), "crisis_rate_high_growth": float(q.crisis_onset[hi].mean()),
                             "crisis_rate_low_growth": float(q.crisis_onset[~hi].mean()),
                             "corr": float(np.corrcoef(q.growth, q.crisis_onset)[0, 1])}
    # 4. severity and consequences
    bins_ = [col for col in c.columns if col.endswith("_bin")]
    out["consequences"] = {b_[:-4]: float(c[b_].mean()) for b_ in bins_}
    out["ends_polity"] = float(c.at_polity_end.mean())
    out["median_duration"] = float(c.duration.median())
    out["severity_mean"] = float(c.severity13.mean())
    # a modern yardstick: crises per polity-century
    tot_expo = float(expo.sum())
    out["crises_per_polity_century"] = float(100 * len(cc) / tot_expo) if tot_expo else None
    (ROOT / "outputs" / "seshat.json").write_text(json.dumps(out, indent=1))
    if verbose:
        print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    run()
