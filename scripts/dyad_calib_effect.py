"""Headline effect of the interstate calibration rule (world.DYAD_CALIB): uncertain-AI worlds, SSP2-4.5, 2 x 4,000 worlds
per rule, same seeds. python -P scripts/dyad_calib_effect.py -> outputs/dyad_calib_effect.json"""
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
import copy
net = copy.copy(ens.fam["N"][0])          # (shallow copy: the ensemble's own member keeps its nets)
net.nets = [n for m in ens.fam["N"] for n in m.nets]
growth, ddem = fit_transitions(p, X, base, LAST_OBS_YEAR)
dy = INTER.sim_inputs(INTER.run(verbose=False, backtests=False))
clim = PL.climate_summary(p, base, LAST_OBS_YEAR, 2065)
out = {"calib_recent": dy["calib_recent"], "calib_long": dy["calib_long"]}
for rule in ("drift", "long", "none"):
    WM.DYAD_CALIB = rule
    gp, gpc, nu, ci_, ip = [], [], [], [], []
    for c in range(2):
        o = simulate(p, base, net, growth, ddem, LAST_OBS_YEAR, 2065, n_scen=4000, clim=clim["ssp245"]["inputs"],
                     world=WM.WorldConfig(ai=WM.AI_SCENARIOS["uncertain"], sample_params=True), dyads=dy, seed=100 + c)
        w = o["world"]
        gp.append(w["great_power_war"].max(1)); gpc.append(w["great_power_war_cow"].max(1)); nu.append(w["nuclear_use"].max(1))
        wn, big = w["watch_names"], w["watch_war_deaths"]
        ci_.append(big[:, wn.index("China-India")] > 0); ip.append(big[:, wn.index("India-Pakistan")] > 0)
    out[rule] = {k: round(float(np.concatenate(v).mean()), 3) for k, v in
                 (("great_power_war", gp), ("great_power_war_cow", gpc), ("nuclear_use", nu), ("china_india_war", ci_),
                  ("india_pakistan_war", ip))}
    print(rule, out[rule], flush=True)
json.dump(out, open(PL.OUT / "dyad_calib_effect.json", "w"), indent=1)
