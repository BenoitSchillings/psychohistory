"""Climate damage to 2100: world GDP per person with climate damage against the same worlds without it, by emissions
path and damage family. Only the ratio is reported: absolute GDP that far out is not credible, but the paired ratio
isolates what the damage functions, tails and harvest channels do. Runs the full simulator (AI uncertain, every
assumption sampled) to 2100.
python -P scripts/climate_2100.py -> outputs/climate_2100.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

import disaster_rca as D
from psychohistory import pipeline as PL, world as WM
from psychohistory.simulate import simulate

N, END, SEED = 2000, 2100, 700
YEARS = (2050, 2065, 2075, 2100)


def main():
    p, base, net, growth, ddem, dy, _, O = D.setup()
    clim = PL.climate_summary(p, base, O, END)
    out = {"n": N, "years": list(YEARS), "scenarios": {}}
    for s in ("ssp126", "ssp245", "ssp370", "ssp585"):
        runs = {}
        for dmg in (1.0, 0.0):
            cfg = WM.WorldConfig(ai=WM.AI_SCENARIOS["uncertain"], sample_params=True)
            runs[dmg] = simulate(p, base, net, growth, ddem, O, END, n_scen=N, clim=clim[s]["inputs"], world=cfg,
                                 dyads=dy, seed=SEED, climate_damage=dmg, record_paths=0)
        yrs = list(runs[1.0]["years"])
        ratio = runs[1.0]["world_gdp"] / runs[0.0]["world_gdp"]
        fam = np.asarray(runs[1.0]["world"]["params"]["dmg_family"]).ravel()
        r = {}
        for y in YEARS:
            x = 100 * (ratio[:, yrs.index(y)] - 1)
            r[y] = {"median": round(float(np.median(x)), 1), "p5": round(float(np.quantile(x, .05)), 1),
                    "p95": round(float(np.quantile(x, .95)), 1),
                    "by_family": {n: round(float(np.median(x[fam == f])), 1) for f, n in
                                  enumerate(("level", "bhm_growth", "global_temp"))}}
        k = yrs.index(2100)
        out["scenarios"][s] = {"loss_pct": r,
                               "conflict_2100": {"damage": round(float(np.median(runs[1.0]["world_conflict"][:, k])), 4),
                                                 "no_damage": round(float(np.median(runs[0.0]["world_conflict"][:, k])), 4)}}
        print(s, json.dumps(out["scenarios"][s]), flush=True)
    json.dump(out, open(PL.OUT / "climate_2100.json", "w"), indent=1)


if __name__ == "__main__":
    main()
