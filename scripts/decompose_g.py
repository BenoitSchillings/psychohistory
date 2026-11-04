"""Which of the 2026-10-28 evening equation changes move the headlines? Paired runs (same seed, 2,000 worlds, uncertain
AI, SSP2-4.5), switching one change back at a time: the growth equation without fiscal capacity, the democracy
equation without free-expression erosion and neopatrimonialism. (The personal-rule family changes the risk models'
inputs; its effect is the remainder against the full rerun.) Also the world's fiscal capacity at the origin against
the training mean, which sets the sign of its growth effect.
python -P scripts/decompose_g.py -> outputs/decompose_g.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import pipeline as PL, world as WM, interstate as INTER, simulate as SM
from psychohistory.backtest import flatten, train_mask, ALL_FAMILIES
from psychohistory.models import SeldonEnsemble, load_combine
from psychohistory.simulate import fit_transitions, simulate
from psychohistory.data import LAST_OBS_YEAR

NEW_G, NEW_D = list(SM.GROWTH_FEATURES), list(SM.DEM_FEATURES)
OLD_G = [c for c in NEW_G if c != "fiscal_capacity"]
OLD_D = [c for c in NEW_D if c not in ("freexp_change_5y", "neopatrimonialism")]

p, base, X, Yl = PL.prepare(False)
Xf, Yf, ci, year = flatten(p, X, Yl)
Ytr, tr = train_mask(Yf, year, LAST_OBS_YEAR, first_year=1860)
ens = SeldonEnsemble(load_combine(), members=3, families=ALL_FAMILIES).fit(Xf[tr], Ytr[tr], ci[tr], year[tr])
import copy
net = copy.copy(ens.fam["N"][0])          # (shallow copy: the ensemble's own member keeps its nets)
net.nets = [n for m in ens.fam["N"] for n in m.nets]
dy = INTER.sim_inputs(INTER.run(verbose=False, backtests=False))
clim = PL.climate_summary(p, base, LAST_OBS_YEAR, 2065)
fc = base["fiscal_capacity"]
yrs = np.asarray(p.years)
w = np.exp(base["lpop"][:, p.yi(LAST_OBS_YEAR)])
ok = np.isfinite(fc[:, p.yi(LAST_OBS_YEAR)]) & np.isfinite(w)
info = {"fiscal_capacity_2025_popweighted": float(np.average(fc[ok, p.yi(LAST_OBS_YEAR)], weights=w[ok])),
        "fiscal_capacity_1955_2024_mean": float(np.nanmean(fc[:, (yrs >= 1955) & (yrs < LAST_OBS_YEAR)]))}
print(info, flush=True)


def run(label, g_cols, d_cols):
    SM.GROWTH_FEATURES, SM.DEM_FEATURES = g_cols, d_cols
    growth, ddem = fit_transitions(p, X, base, LAST_OBS_YEAR)
    o = simulate(p, base, net, growth, ddem, LAST_OBS_YEAR, 2065, n_scen=2000, clim=clim["ssp245"]["inputs"],
                 world=WM.WorldConfig(ai=WM.AI_SCENARIOS["uncertain"], sample_params=True), dyads=dy, seed=11)
    wd = o["world"]
    r = {"world_conflict_2065_median": round(float(np.median(np.asarray(o["world_conflict"])[:, -1])), 4),
         "world_gdp_2065_median": round(float(np.median(np.asarray(o["world_gdp"])[:, -1])), 0),
         "world_gdp_2045_median": round(float(np.median(np.asarray(o["world_gdp"])[:, 20])), 0),
         "world_dem_2065_median": round(float(np.median(np.asarray(o["world_dem"])[:, -1])), 4) if "world_dem" in o else None,
         "great_power_war": round(float(np.asarray(wd["great_power_war"]).max(1).mean()), 4),
         "nuclear_use": round(float(np.asarray(wd["nuclear_use"]).max(1).mean()), 4) if "nuclear_use" in wd else None}
    print(label, r, flush=True)
    return r


out = {"info": info, "all new": run("all new", NEW_G, NEW_D), "growth without fiscal capacity": run("growth without fiscal capacity", OLD_G, NEW_D),
       "old democracy equation": run("old democracy equation", NEW_G, OLD_D), "both old": run("both old", OLD_G, OLD_D)}
SM.GROWTH_FEATURES, SM.DEM_FEATURES = NEW_G, NEW_D
json.dump(out, open(PL.OUT / "decompose_g.json", "w"), indent=1)
