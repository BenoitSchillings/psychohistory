"""Simulate many worlds (uncertain AI, SSP2-4.5, all assumptions sampled) and keep, per world, every sampled assumption
and the 2065 outcomes, for scripts/uncertainty_attribution.py. python -P scripts/uncertainty_sims.py [chunks]
-> outputs/uncertainty_sims.npz (chunks x 4,000 worlds, a different seed per chunk)"""
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

CH = int(sys.argv[1]) if len(sys.argv) > 1 else 12
p, base, X, Yl = PL.prepare(False)
Xf, Yf, ci, year = flatten(p, X, Yl)
Ytr, tr = train_mask(Yf, year, LAST_OBS_YEAR, first_year=1860)
ens = SeldonEnsemble(load_combine(), members=3, families=ALL_FAMILIES).fit(Xf[tr], Ytr[tr], ci[tr], year[tr])
net = ens.fam["N"][0]
net.nets = [n for m in ens.fam["N"] for n in m.nets]
# as the forecast (code review 2026-10-28: these runs had no structural uncertainty and no climate damage)
from psychohistory.features import TASKS
parts = ens.predict_families(X[:, p.yi(LAST_OBS_YEAR)])
lgf = lambda v: np.log(np.clip(v, 1e-4, 1 - 1e-4) / (1 - np.clip(v, 1e-4, 1 - 1e-4)))
net.struct_sd = np.nan_to_num(np.nanstd(np.stack([lgf(v) for v in parts.values()]), 0)[:, [TASKS.index(t) for t in ("onset_1y", "termination_1y", "coup_1y")]])
net.struct_sd, net.struct_fam, net.struct_w = PL.struct_inputs(ens, X[:, p.yi(LAST_OBS_YEAR)])
growth, ddem = fit_transitions(p, X, base, LAST_OBS_YEAR)
dy = INTER.sim_inputs(INTER.run(verbose=False, backtests=False))
clim = PL.climate_summary(p, base, LAST_OBS_YEAR, 2065)
P, Y = {}, {}
add = lambda d, k, v: d.setdefault(k, []).append(np.asarray(v, float))
for c in range(CH):
    o = simulate(p, base, net, growth, ddem, LAST_OBS_YEAR, 2065, n_scen=4000, clim=clim["ssp245"]["inputs"],
                 world=WM.WorldConfig(ai=WM.AI_SCENARIOS["uncertain"], sample_params=True), dyads=dy, seed=100 + c, climate_damage=1.0)
    w = o["world"]
    for k, v in w["params"].items():
        add(P, k, v)
    add(Y, "log_world_gdp_2065", np.log(o["world_gdp"][:, -1]))
    add(Y, "log_world_gdp_2045", np.log(o["world_gdp"][:, 19]))
    add(Y, "world_conflict_2065", o["world_conflict"][:, -1])
    add(Y, "world_democracy_2065", o["world_dem"][:, -1])
    add(Y, "great_power_war", w["great_power_war"].max(1))
    if "great_power_war_cow" in w:
        add(Y, "great_power_war_cow", w["great_power_war_cow"].max(1))
    if "nuclear_use" in w:
        add(Y, "nuclear_use", w["nuclear_use"].max(1))
    if "battle_deaths" in w:
        add(Y, "log_battle_deaths", np.log1p(w["battle_deaths"].sum(-1)))
    wn, big = w.get("watch_names", []), w.get("watch_war_deaths")
    for nm in ("China-India", "China-Taiwan", "Russia-Poland", "India-Pakistan"):
        if big is not None and nm in wn:
            add(Y, f"war_{nm}", big[:, wn.index(nm)] > 0)
    print("chunk", c, flush=True)
np.savez_compressed(PL.OUT / "uncertainty_sims.npz", **{f"p_{k}": np.concatenate(v) for k, v in P.items()},
                    **{f"y_{k}": np.concatenate(v) for k, v in Y.items()})
print("saved")
