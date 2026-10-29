"""How much do the war headlines rest on the great-power rule (COW majors + nuclear states with >= GP_SHARE of world
output)? Paired runs (same seed, 4,000 worlds, uncertain AI) over the threshold. python -P scripts/gp_share_sweep.py
-> outputs/gp_share_sweep.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import pipeline as PL, world as WM, interstate as INTER
from psychohistory.backtest import flatten, train_mask, ALL_FAMILIES
from psychohistory.models import SeldonEnsemble, load_combine
from psychohistory.simulate import fit_transitions, simulate
from psychohistory.data import LAST_OBS_YEAR

p, base, X, Yl = PL.prepare(False)
Xf, Yf, ci, year = flatten(p, X, Yl)
Ytr, tr = train_mask(Yf, year, LAST_OBS_YEAR, first_year=1860)
ens = SeldonEnsemble(load_combine(), members=3, families=ALL_FAMILIES).fit(Xf[tr], Ytr[tr], ci[tr], year[tr])
net = ens.fam["N"][0]
net.nets = [n for m in ens.fam["N"] for n in m.nets]
growth, ddem = fit_transitions(p, X, base, LAST_OBS_YEAR)
dy = INTER.sim_inputs(INTER.run(verbose=False, backtests=False))
clim = PL.climate_summary(p, base, LAST_OBS_YEAR, 2065)
names = list(p.names) if hasattr(p, "names") else None
out = {}
for thr in (0.03, 0.04, 0.05, 0.065, 0.08, 0.10, 9.0):
    WM.GP_SHARE = thr
    o = simulate(p, base, net, growth, ddem, LAST_OBS_YEAR, 2065, n_scen=4000, clim=clim["ssp245"]["inputs"],
                 world=WM.WorldConfig(ai=WM.AI_SCENARIOS["uncertain"], sample_params=True), dyads=dy, seed=11)
    w = o["world"]
    r = {"great_power_war": round(float(w["great_power_war"].max(1).mean()), 4),
         "great_power_war_cow": round(float(w["great_power_war_cow"].max(1).mean()), 4) if "great_power_war_cow" in w else None,
         "nuclear_use": round(float(w["nuclear_use"].max(1).mean()), 4) if "nuclear_use" in w else None}
    if "battle_deaths" in w:
        cum = w["battle_deaths"].sum(-1)
        r.update(battle_deaths_median=round(float(np.median(cum)), 0), battle_deaths_mean=round(float(cum.mean()), 0))
    r["world_gdp_2065_median"] = round(float(np.median(np.asarray(o["world_gdp"])[:, -1])), 0)
    out["none (COW majors only)" if thr > 1 else str(thr)] = r
    print(thr, r, flush=True)
WM.GP_SHARE = 0.05
json.dump(out, open(PL.OUT / "gp_share_sweep.json", "w"), indent=1)
