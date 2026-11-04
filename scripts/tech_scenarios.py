"""Technology breakthroughs (psychohistory/tech.py): what they do to the 2025-2065 forecast. SSP2-4.5, every assumption
sampled. Three parts:
  A  all technologies drawn per world against none (same seeds): AI uncertain (2 x 4,000 worlds) and no AI (4,000)
  B  paired forcing: the same 4,000 worlds (AI uncertain, every other technology drawn) with one technology forced to
     arrive in 2035 or never; the difference is that technology's conditional effect
  C  attribution in the part-A worlds: 2065 outcomes regressed on which technologies arrived by 2045 (OLS, all at once)
python -P scripts/tech_scenarios.py [A|B|C ...] -> outputs/tech_scenarios.json (and tech_worlds.npz for C)"""
import json
import os
import sys
import time
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
from psychohistory.tech import TECHS

PARTS = sys.argv[1:] or ["A", "B", "C"]
N = 4000
FORCE = ["long_storage", "fusion", "adv_fission", "geoengineering", "crops", "protein", "vaccines_100d", "pathogens",
         "longevity", "quantum_crypto"]
PETRO = ["SAU", "RUS", "IRQ", "NGA", "AGO", "VEN", "IRN", "KWT", "ARE", "DZA", "LBY", "KAZ", "OMN", "QAT", "AZE"]
SAHEL = ["NER", "TCD", "MLI", "BFA", "SDN", "ETH", "SOM", "MRT", "SSD", "ERI"]
OUTF = os.path.join(PL.OUT, "tech_scenarios.json")

p, base, X, Yl = PL.prepare(False)
Xf, Yf, ci, year = flatten(p, X, Yl)
Ytr, tr = train_mask(Yf, year, LAST_OBS_YEAR, first_year=1860)
ens = SeldonEnsemble(load_combine(), members=3, families=ALL_FAMILIES).fit(Xf[tr], Ytr[tr], ci[tr], year[tr])
net = PL.sim_net(ens, X[:, p.yi(LAST_OBS_YEAR)])     # (as the forecast: pooled seeds, structural inputs, ensemble central offset)
growth, ddem = fit_transitions(p, X, base, LAST_OBS_YEAR)
dy = INTER.sim_inputs(INTER.run(verbose=False, backtests=False))
clim = PL.climate_summary(p, base, LAST_OBS_YEAR, 2065)
pop = np.exp(base["lpop"][:, p.yi(LAST_OBS_YEAR)])
idx = {c: i for i, c in enumerate(p.codes)}
grp = lambda cs: np.array([c in cs for c in p.codes])
petro, sahel = grp(PETRO), grp(SAHEL)


def run(ai, tech, seed, force=None):
    t0 = time.time()
    o = simulate(p, base, net, growth, ddem, LAST_OBS_YEAR, 2065, n_scen=N, record_paths=N, clim=clim["ssp245"]["inputs"],
                 world=WM.WorldConfig(ai=WM.AI_SCENARIOS[ai], sample_params=True, tech=tech, tech_force=force),
                 dyads=dy, seed=seed, climate_damage=1.0)
    w, g = o["world"], o["paths"]["gdp"][:, :, -1]
    lw = lambda m: np.log(np.nansum(g[:, m] * pop[m], 1) / pop[m].sum())
    y = {"log_world_gdp": np.log(o["world_gdp"][:, -1]), "conflict": o["world_conflict"][:, -1],
         "democracy": o["world_dem"][:, -1], "great_power_war": w["great_power_war"].max(1),
         "nuclear_use": w["nuclear_use"].max(1), "log_oil": w["oil"][:, -1], "clean_share": w["clean_share"][:, -1],
         "log_food_price": w["food_price"][:, -1] if "food_price" in w else np.zeros(N),
         # (the record is a running total: its last year is the 2026-2065 count)
         "excess_deaths_m": w["excess_deaths"][:, -1] / 1e6 if "excess_deaths" in w else np.zeros(N),
         "log_gdp_petrostates": lw(petro), "log_gdp_sahel": lw(sahel),
         "coups": o["paths"]["coup"].sum((1, 2)).astype(float)}
    print(f"  run ai={ai} tech={tech} force={force} seed={seed}: {time.time() - t0:.0f}s", flush=True)
    return y, w.get("tech_arrival")


def summ(y):
    r = {}
    for k, v in y.items():
        v = np.asarray(v, float)
        if k in ("great_power_war", "nuclear_use", "conflict", "coups"):
            r[k] = float(np.mean(v))          # (means: the medians of these near-discrete outcomes barely move)
        elif k.startswith("log_"):
            r[k.replace("log_", "") + "_median"] = float(np.exp(np.median(v)))
        else:
            r[k + "_median"] = float(np.median(v))
            if k == "excess_deaths_m":
                r[k + "_mean"] = float(np.mean(v))
    return r


res = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
if "A" in PARTS:
    YA, TA = {}, {}
    for ai, seeds in (("uncertain", (21, 22)), ("none", (21,))):
        for tech in (False, True):
            ys, ts = [], []
            for sd in seeds:
                y, t = run(ai, tech, sd)
                ys.append(y), ts.append(t)
            y = {k: np.concatenate([a[k] for a in ys]) for k in ys[0]}
            res[f"A_{ai}_{'tech' if tech else 'none'}"] = summ(y)
            if tech and ai == "uncertain":
                YA, TA = y, {k: np.concatenate([a[k] for a in ts]) for k in ts[0]}
            json.dump(res, open(OUTF, "w"), indent=1)
    np.savez_compressed(os.path.join(PL.OUT, "tech_worlds.npz"), **{f"y_{k}": v for k, v in YA.items()},
                        **{f"t_{k}": v for k, v in TA.items()})
if "B" in PARTS:
    for name in FORCE:
        for when in (2035, None):
            key = f"B_{name}_{when or 'never'}"
            if key in res:
                continue
            y, _ = run("uncertain", True, 31, force={name: when})
            res[key] = summ(y)
            json.dump(res, open(OUTF, "w"), indent=1)
if "C" in PARTS:
    z = np.load(os.path.join(PL.OUT, "tech_worlds.npz"))
    names = [k[2:] for k in z.files if k.startswith("t_")]
    T = np.stack([(z[f"t_{n}"] <= 2045).astype(float) for n in names], 1)
    D = np.c_[np.ones(len(T)), T]
    out = {}
    for k in [k[2:] for k in z.files if k.startswith("y_")]:
        yv = z[f"y_{k}"].astype(float)
        ok = np.isfinite(yv)
        b, *_ = np.linalg.lstsq(D[ok], yv[ok], rcond=None)
        res_ = yv[ok] - D[ok] @ b
        se = np.sqrt(np.diag(np.linalg.inv(D[ok].T @ D[ok])) * res_.var())
        out[k] = {n: {"coef": round(float(b[i + 1]), 5), "t": round(float(b[i + 1] / se[i + 1]), 1)} for i, n in enumerate(names)}
        out[k]["_r2"] = round(float(1 - res_.var() / yv[ok].var()), 3)
    res["C_attribution_by2045"] = out
    json.dump(res, open(OUTF, "w"), indent=1)
print("done")
