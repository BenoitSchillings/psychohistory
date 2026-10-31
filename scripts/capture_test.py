"""What does the AI rent-capture assumption do? The same 2,000 worlds (near-free cognition from 2030, SSP2-4.5, every
other assumption sampled, same seed) with the firms' share of AI rents fixed low, at the mean, and high, and with fast
clawback. Reports 2065 world GDP per person, the frontier hubs' GDP against the rest, world democracy, and the
population-weighted top-1% share and polarization.
python -P scripts/capture_test.py -> outputs/capture_test.json"""
import dataclasses
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

p, base, X, Yl = PL.prepare(False)
Xf, Yf, ci, year = flatten(p, X, Yl)
Ytr, tr = train_mask(Yf, year, LAST_OBS_YEAR, first_year=1860)
ens = SeldonEnsemble(load_combine(), members=3, families=ALL_FAMILIES).fit(Xf[tr], Ytr[tr], ci[tr], year[tr])
net = ens.fam["N"][0]
net.nets = [n for m in ens.fam["N"] for n in m.nets]
parts = ens.predict_families(X[:, p.yi(LAST_OBS_YEAR)])
lgf = lambda v: np.log(np.clip(v, 1e-4, 1 - 1e-4) / (1 - np.clip(v, 1e-4, 1 - 1e-4)))
net.struct_sd = np.nan_to_num(np.nanstd(np.stack([lgf(v) for v in parts.values()]), 0)[:, [TASKS.index(t) for t in ("onset_1y", "termination_1y", "coup_1y")]])
growth, ddem = fit_transitions(p, X, base, LAST_OBS_YEAR)
dy = INTER.sim_inputs(INTER.run(verbose=False, backtests=False))
clim = PL.climate_summary(p, base, LAST_OBS_YEAR, 2065)
pop = np.exp(base["lpop"][:, p.yi(LAST_OBS_YEAR)])
hub = np.array([WM.HUBS.get(c, 0.0) >= 0.4 for c in p.codes])
dem0 = base["dem"][:, p.yi(LAST_OBS_YEAR)] > 0.5
res = {}
for name, cap, claw in (("capture 0.2", 0.2, 0.0), ("capture 0.55", 0.55, 0.0), ("capture 0.9", 0.9, 0.0),
                        ("capture 0.9, fast clawback", 0.9, 0.1), ("no AI", None, None)):
    ai = WM.AI_SCENARIOS["none"] if cap is None else dataclasses.replace(WM.AI_SCENARIOS["2030"], firm_capture=cap, clawback=claw)
    o = simulate(p, base, net, growth, ddem, LAST_OBS_YEAR, 2065, n_scen=2000, record_paths=2000, clim=clim["ssp245"]["inputs"],
                 world=WM.WorldConfig(ai=ai, sample_params=True), dyads=dy, seed=7, climate_damage=1.0)
    g = o["paths"]["gdp"][:, :, -1]
    ok = np.isfinite(g).all(0) & np.isfinite(pop)
    w = lambda m: pop[ok & m] / pop[ok & m].sum()
    lw = lambda m: np.log((g[:, ok & m] * w(m)[None]).sum(1))
    se = o["slow_end"]
    pw = lambda k, m: float(np.nansum(se[k][ok & m, 2] * w(m))) if k in se else None
    res[name] = {"world_gdp_2065": float(np.median(o["world_gdp"][:, -1])),
                 "hubs_vs_rest_log": float(np.median(lw(hub) - lw(~hub))),
                 "world_dem_2065": float(np.median(o["world_dem"][:, -1])),
                 "top1_pw": pw("top1", np.ones_like(ok)), "polarization_democracies": pw("polarization", dem0),
                 "conflict_2065": float(np.median(o["world_conflict"][:, -1]))}
    print(name, {k: round(v, 3) if v is not None else None for k, v in res[name].items()}, flush=True)
json.dump(res, open(os.path.join(PL.OUT, "capture_test.json"), "w"), indent=1)
