"""Cheap space (scenario): the same worlds (SSP2-4.5, every assumption sampled, same seeds) with and without launch
costs falling 10-100x and constellations, space-based sensors and interceptors, satellite internet and orbital compute
deploying from 2028 to 2040 (world.WorldConfig.cheap_space). Uncertain AI and no AI. Reports great-power war, the
United States against China and Russia, nuclear use, civil conflict, coups, democracy and GDP in 2065, and, within the
cheap-space worlds, the split by whether space dominance deters (space_war < 0) or destabilises (> 0).
python -P scripts/space_scenario.py [n_worlds] -> outputs/space_scenario.json"""
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
from psychohistory.features import TASKS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
p, base, X, Yl = PL.prepare(False)
Xf, Yf, ci, year = flatten(p, X, Yl)
Ytr, tr = train_mask(Yf, year, LAST_OBS_YEAR, first_year=1860)
ens = SeldonEnsemble(load_combine(), members=3, families=ALL_FAMILIES).fit(Xf[tr], Ytr[tr], ci[tr], year[tr])
net = ens.fam["N"][0]
net.nets = [n for m in ens.fam["N"] for n in m.nets]
parts = ens.predict_families(X[:, p.yi(LAST_OBS_YEAR)])
lgf = lambda v: np.log(np.clip(v, 1e-4, 1 - 1e-4) / (1 - np.clip(v, 1e-4, 1 - 1e-4)))
net.struct_sd = np.nan_to_num(np.nanstd(np.stack([lgf(v) for v in parts.values()]), 0)[:, [TASKS.index(t) for t in ("onset_1y", "termination_1y", "coup_1y")]])
net.struct_sd, net.struct_fam, net.struct_w = PL.struct_inputs(ens, X[:, p.yi(LAST_OBS_YEAR)])
growth, ddem = fit_transitions(p, X, base, LAST_OBS_YEAR)
dy = INTER.sim_inputs(INTER.run(verbose=False, backtests=False))
clim = PL.climate_summary(p, base, LAST_OBS_YEAR, 2065)
PAIRS = ("China-United States", "Russia-United States", "China-India", "China-Taiwan")


def summary(o, sel=None):
    w = o["world"]
    sel = np.ones(len(o["world_gdp"]), bool) if sel is None else sel
    r = {"great_power_war": float(w["great_power_war"].max(1)[sel].mean()),
         "great_power_war_cow": float(w["great_power_war_cow"].max(1)[sel].mean()),
         "nuclear_use": float(w["nuclear_use"].max(1)[sel].mean()),
         "nuclear_large": float(w["nuclear_large"].max(1)[sel].mean()),
         "conflict_2065": float(np.median(o["world_conflict"][sel, -1])),
         "democracy_2065": float(np.median(o["world_dem"][sel, -1])),
         "gdp_2065": float(np.median(o["world_gdp"][sel, -1]))}
    return r


res = {}
for ai in ("uncertain", "none"):
    for space in (False, True):
        o = simulate(p, base, net, growth, ddem, LAST_OBS_YEAR, 2065, n_scen=N, clim=clim["ssp245"]["inputs"],
                     world=WM.WorldConfig(ai=WM.AI_SCENARIOS[ai], sample_params=True, cheap_space=space, tech=False), dyads=dy,
                     seed=11, climate_damage=1.0)
        key = f"{ai}_{'space' if space else 'base'}"
        res[key] = summary(o)
        w = o["world"]
        hits = w.get("watch_war_hits", {})
        if hits:
            last = hits[max(hits)]
            res[key].update({f"war_{nm}": float(last[w["watch_names"].index(nm)]) for nm in PAIRS if nm in w["watch_names"]})
        if space:
            sw = w["params"]["space_war"].ravel()
            res[key + "_deters"] = summary(o, sw < 0)
            res[key + "_destabilises"] = summary(o, sw >= 0.35)
        print(key, {k: round(v, 4) for k, v in res[key].items()}, flush=True)
json.dump(res, open(os.path.join(PL.OUT, "space_scenario.json"), "w"), indent=1)
for ai in ("uncertain", "none"):
    b, s = res[f"{ai}_base"], res[f"{ai}_space"]
    print(f"\n{ai}: change with cheap space")
    for k in b:
        print(f"  {k:24s} {b[k]:.4f} -> {s[k]:.4f}")
    for t in ("deters", "destabilises"):
        print(f"  [{t}]", {k: round(v, 4) for k, v in res[f"{ai}_space_{t}"].items()})
