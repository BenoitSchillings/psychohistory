"""The dollar's decline (scenario): the same worlds (SSP2-4.5, every assumption sampled, same seeds, technology
registry on) with and without the dollar's reserve role falling by a sampled share (0-100%) by 2045 - sanctions on
states at war with the West losing up to half their bite, and US growth losing up to 0.3% a year at full decline
(world.WorldConfig.dollar_decline). Uncertain AI and no AI. Reports great-power war, the key pairs, nuclear use,
conflict, democracy, world GDP and US and Chinese GDP per person in 2065, and the split by how deep the decline is.
python -P scripts/dollar_scenario.py [n_worlds] -> outputs/dollar_scenario.json"""
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
PAIRS = ("China-United States", "Russia-United States", "China-Taiwan")


def summary(o, sel=None):
    w = o["world"]
    sel = np.ones(len(o["world_gdp"]), bool) if sel is None else sel
    G = o["paths"]["gdp"]                                   # [S, C, T] GDP per person (levels), all worlds recorded
    us, cn = CODES.index("USA"), CODES.index("CHN")
    return {"great_power_war": float(w["great_power_war"].max(1)[sel].mean()),
            "nuclear_use": float(w["nuclear_use"].max(1)[sel].mean()),
            "conflict_2065": float(np.median(o["world_conflict"][sel, -1])),
            "democracy_2065": float(np.median(o["world_dem"][sel, -1])),
            "gdp_2065": float(np.median(o["world_gdp"][sel, -1])),
            "usa_gdppc_2065": float(np.median(G[sel, us, -1])),
            "chn_gdppc_2065": float(np.median(G[sel, cn, -1]))}


CODES = list(p.codes)
res = {}
for ai in ("uncertain", "none"):
    for usd in (False, True):
        o = simulate(p, base, net, growth, ddem, LAST_OBS_YEAR, 2065, n_scen=N, clim=clim["ssp245"]["inputs"], record_paths=N,
                     world=WM.WorldConfig(ai=WM.AI_SCENARIOS[ai], sample_params=True, dollar_decline=usd), dyads=dy,
                     seed=11, climate_damage=1.0)
        key = f"{ai}_{'dollar' if usd else 'base'}"
        res[key] = summary(o)
        hits = o["world"].get("watch_war_hits", {})
        if hits:
            last = hits[max(hits)]
            res[key].update({f"war_{nm}": float(last[o["world"]["watch_names"].index(nm)]) for nm in PAIRS if nm in o["world"]["watch_names"]})
        if usd:
            sh = o["world"]["params"]["dollar_shift"].ravel()
            res[key + "_deep"] = summary(o, sh >= 0.67)
            res[key + "_mild"] = summary(o, sh < 0.33)
        print(key, {k: round(v, 4) for k, v in res[key].items()}, flush=True)
json.dump(res, open(os.path.join(PL.OUT, "dollar_scenario.json"), "w"), indent=1)
for ai in ("uncertain", "none"):
    b, s = res[f"{ai}_base"], res[f"{ai}_dollar"]
    print(f"\n{ai}: change with the dollar's decline")
    for k in b:
        print(f"  {k:24s} {b[k]:.4f} -> {s[k]:.4f}")
    for t in ("deep", "mild"):
        print(f"  [{t}]", {k: round(v, 4) for k, v in res[f"{ai}_dollar_{t}"].items()})
