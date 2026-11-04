"""Which of this round's changes move world GDP per person in 2065? Paired runs (same seed, 2,000 worlds, uncertain AI),
switching one group of parameters off at a time. python -P scripts/decompose_round.py -> outputs/decompose_round.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import pipeline as PL, world as WM, interstate as INTER
from psychohistory.backtest import flatten, train_mask
from psychohistory.models import SeldonEnsemble, load_combine
from psychohistory.backtest import ALL_FAMILIES
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

OFF = {"all on": {}, "no workforce": {"workforce_beta": 0}, "no fisheries": {"fish_trop": 0, "fish_mid": 0, "fish_high": 0, "wild_trend": 0},
       "no science term": {"articles_beta": 0}, "no AI dividend fade": {"dividend_fade": 0},
       "no 2025 tariffs / aid cut": {"tariff_pass": 0, "tariff_us": 0, "aid_mult": 0}}
saved = {k: dict(getattr(WM, k)) for k in ("PARAM_RANGES", "WATER_RANGES", "EXTRA_RANGES")}
out = {}
for name, off in OFF.items():
    for k in saved:
        rng = dict(saved[k])
        for prm, v in off.items():
            if prm in rng:
                rng[prm] = ("u", v, v)
        setattr(WM, k, rng)
    o = simulate(p, base, net, growth, ddem, LAST_OBS_YEAR, 2065, n_scen=2000, clim=clim["ssp245"]["inputs"],
                 world=WM.WorldConfig(ai=WM.AI_SCENARIOS["uncertain"], sample_params=True), dyads=dy, seed=11)
    wg = np.asarray(o["world_gdp"])
    out[name] = {"world_gdp_2065_median": round(float(np.median(wg[:, -1])), 0), "world_gdp_2045_median": round(float(np.median(wg[:, 20])), 0)}
    print(name, out[name], flush=True)
for k, v in saved.items():
    setattr(WM, k, v)
json.dump(out, open(PL.OUT / "decompose_round.json", "w"), indent=1)
