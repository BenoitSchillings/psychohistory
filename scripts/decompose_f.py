"""Which of round (f)'s changes move the headlines? Paired runs (same seed, 2,000 worlds, uncertain AI), switching one
change back to its old setting at a time. python -P scripts/decompose_f.py -> outputs/decompose_f.json"""
import dataclasses
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

OLD_ARRIVAL = ((2026.75, 0.0), (2027.5, 0.175), (2032.5, 0.375), (2041.5, 0.575), (2047.5, 0.66), (2065.5, 0.775))
OLD_ROBOT = ((2026.75, 0.0), (2027.5, 0.05), (2030.5, 0.20), (2032.5, 0.32), (2040.5, 0.56), (2050.5, 0.70), (2065.5, 0.80))
OLD_RANGES = {"nuclear_war_factor": ("u", 0.10, 0.70), "phys_dividend": ("u", 0.0, 0.05), "nuc_use_one": ("u", 0.0005, 0.01)}

p, base, X, Yl = PL.prepare(False)
Xf, Yf, ci, year = flatten(p, X, Yl)
Ytr, tr = train_mask(Yf, year, LAST_OBS_YEAR, first_year=1860)
ens = SeldonEnsemble(load_combine(), members=3, families=ALL_FAMILIES).fit(Xf[tr], Ytr[tr], ci[tr], year[tr])
import copy
net = copy.copy(ens.fam["N"][0])          # (shallow copy: the ensemble's own member keeps its nets)
net.nets = [n for m in ens.fam["N"] for n in m.nets]
growth, ddem = fit_transitions(p, X, base, LAST_OBS_YEAR)
df = INTER.build(verbose=False)
dy = INTER.sim_inputs(INTER.run(verbose=False, backtests=False, df=df))
INTER.CONT_HALFLIFE = None
dy_old_cont = INTER.sim_inputs(INTER.run(verbose=False, backtests=False, df=df))
INTER.CONT_HALFLIFE = 25
clim = PL.climate_summary(p, base, LAST_OBS_YEAR, 2065)


def run(label, ai=None, dyads=dy):
    o = simulate(p, base, net, growth, ddem, LAST_OBS_YEAR, 2065, n_scen=2000, clim=clim["ssp245"]["inputs"],
                 world=WM.WorldConfig(ai=ai or WM.AI_SCENARIOS["uncertain"], sample_params=True), dyads=dyads, seed=11)
    w = o["world"]
    r = {"world_conflict_2065_median": round(float(np.median(np.asarray(o["world_conflict"])[:, -1])), 4),
         "world_conflict_2035_median": round(float(np.median(np.asarray(o["world_conflict"])[:, 10])), 4),
         "world_gdp_2065_median": round(float(np.median(np.asarray(o["world_gdp"])[:, -1])), 0),
         "great_power_war": round(float(np.asarray(w["great_power_war"]).max(1).mean()), 4),
         "nuclear_use": round(float(np.asarray(w["nuclear_use"]).max(1).mean()), 4) if "nuclear_use" in w else None,
         "coups_per_year_2026_35": round(float(np.asarray(o["coups"])[:, :10].mean()), 3)}
    print(label, r, flush=True)
    return r


out = {"all new": run("all new")}
SM.SHIFTERS_BEFORE_ALIGN = False
out["old shifter order"] = run("old shifter order")
SM.SHIFTERS_BEFORE_ALIGN = True
WM.PAIR_NOISE_CORR = 0.0
out["no pair-noise correction"] = run("no pair-noise correction")
WM.PAIR_NOISE_CORR = 1.0
out["equal-weight continuation"] = run("equal-weight continuation", dyads=dy_old_cont)
saved = dict(WM.PARAM_RANGES), dict(WM.EXTRA_RANGES), WM.ROBOT_CDF
WM.ROBOT_CDF = OLD_ROBOT
out["old AI and robot arrival"] = run("old AI and robot arrival", ai=dataclasses.replace(WM.AI_SCENARIOS["uncertain"], arrival_cdf=OLD_ARRIVAL))
WM.ROBOT_CDF = saved[2]
for k, v in OLD_RANGES.items():
    (WM.PARAM_RANGES if k in WM.PARAM_RANGES else WM.EXTRA_RANGES)[k] = v
out["old nuclear and robot-dividend ranges"] = run("old nuclear and robot-dividend ranges")
WM.PARAM_RANGES.clear(); WM.PARAM_RANGES.update(saved[0]); WM.EXTRA_RANGES.clear(); WM.EXTRA_RANGES.update(saved[1])
json.dump(out, open(PL.OUT / "decompose_f.json", "w"), indent=1)
