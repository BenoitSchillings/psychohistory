"""Over-clustering check for simulated conflict. If the simulator links countries' wars too tightly, the spread across
worlds of the WORLD conflict level (share of country-years in conflict over the horizon, and number of new onsets
among countries at peace at the origin) is too wide: the realised value then falls in the middle of the simulated
range too often (PIT piles up at 0.5) and the CRPS is worse than that of the same forecasts with countries shuffled
independently (same marginals, no cross-country dependence).
python -P scripts/conflict_clustering_check.py [mood_sd first last tag] -> outputs/conflict_clustering_check[_tag].json
(the mood sweep uses dev origins only: 1970-1998)"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import pipeline as PL
from psychohistory.backtest import flatten, train_mask
from psychohistory.models import PsychoNet, NET_TUNED
from psychohistory.simulate import fit_transitions, simulate
from psychohistory.data import LAST_OBS_YEAR

S, H = 4000, 10


def crps(x, y):
    x = np.sort(x)
    n = len(x)
    return float(np.mean(np.abs(x - y)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2)


p, base, X, Yl = PL.prepare(False)
Xf, Yf, ci, year = flatten(p, X, Yl)
rng = np.random.default_rng(1)
A = sys.argv[1:]
MOOD = float(A[0]) if A else 0.20
ORIG = range(int(A[1]), int(A[2]) + 1, 2) if len(A) > 2 else range(1970, 2014, 2)
TAG = f"_{A[3]}" if len(A) > 3 else ""
out = {}
for O in ORIG:
    E = min(O + H, LAST_OBS_YEAR)
    Ytr, tr = train_mask(Yf, year, O)
    net = PsychoNet(members=8, **NET_TUNED).fit(Xf[tr], Ytr[tr], ci[tr])
    growth, ddem = fit_transitions(p, X, base, O)
    o = simulate(p, base, net, growth, ddem, O, E, n_scen=S, demography="extrapolate", record_paths=S, seed=O + 7, mood_sd=(MOOD, MOOD))
    cf = o["paths"]["conflict"]                                      # [S, C, T]
    i0, i1 = p.yi(O), p.yi(E)
    act = base["conflict"][:, i0:i1 + 1]
    ok = np.isfinite(act).all(1)
    peace = ok & (act[:, 0] == 0)
    full_sim = np.concatenate([np.repeat(act[None, :, :1], S, 0), cf], -1)
    ons_sim = ((full_sim[:, :, 1:] > 0.5) & (full_sim[:, :, :-1] < 0.5)).any(-1)          # [S, C]
    ons_act = ((act[:, 1:] == 1) & (act[:, :-1] == 0)).any(-1)
    stats = {"share": (cf[:, ok].mean((1, 2)), act[ok, 1:].mean(),
                       lambda: np.stack([cf[rng.permutation(S), j].mean(-1) for j in np.where(ok)[0]], 1).mean(1)),
             "onsets": (ons_sim[:, peace].sum(1).astype(float), float(ons_act[peace].sum()),
                        lambda: np.stack([ons_sim[rng.permutation(S), j] for j in np.where(peace)[0]], 1).sum(1).astype(float))}
    r = {}
    for k, (sim, a, shuf) in stats.items():
        sh = shuf()
        r[k] = {"actual": float(a), "sim_mean": float(sim.mean()), "sd_sim": float(sim.std()), "sd_shuffled": float(sh.std()),
                "pit_sim": float((sim < a).mean() + 0.5 * (sim == a).mean()), "pit_shuffled": float((sh < a).mean() + 0.5 * (sh == a).mean()),
                "crps_sim": crps(sim, a), "crps_shuffled": crps(sh, a)}
    out[O] = r
    print(O, json.dumps({k: {m: round(v, 4) for m, v in x.items()} for k, x in r.items()}), flush=True)
summ = {}
for k in ("share", "onsets"):
    g = lambda m: np.array([out[O][k][m] for O in out])
    summ[k] = {"sd_ratio_sim_vs_shuffled": round(float(np.mean(g("sd_sim") / g("sd_shuffled"))), 3),
               "crps_rel_change": round(float(np.mean(g("crps_sim") / g("crps_shuffled") - 1)), 4),
               "pit_sim": g("pit_sim").round(3).tolist(), "pit_shuffled": g("pit_shuffled").round(3).tolist(),
               "share_pit_in_middle_half_sim": round(float(np.mean(np.abs(g("pit_sim") - .5) < .25)), 3),
               "share_pit_in_middle_half_shuffled": round(float(np.mean(np.abs(g("pit_shuffled") - .5) < .25)), 3)}
print(json.dumps(summ, indent=1))
json.dump({"by_origin": {str(k): v for k, v in out.items()}, "summary": summ}, open(PL.OUT / f"conflict_clustering_check{TAG}.json", "w"), indent=1)
