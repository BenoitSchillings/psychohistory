"""Three-way rank-by-rank blend for the monthly country model: LightGBM (production, residual target) + neural model on
the residual target + neural model on the direct target (log deaths). Dev windows 2010-2017 only; the blend weights are
chosen there. python -P scripts/cm_blend3_test.py dev -> outputs/cm_blend3_test.json"""
import itertools
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from psychohistory import cm_fatal as CF
from predcomp_eval import crps_ensemble
import cm_nn_test as NT

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "outputs", "cm_recal_cache")
GRID = [(a, b) for a, b in itertools.product(np.arange(0, 1.01, 0.1), repeat=2) if a + b <= 1.0001]   # (w_nn, w_direct)


def draws(y, kind):
    f = os.path.join(CACHE, f"nn_{kind}_{y}.npz")
    if not os.path.exists(f):
        NT.NN_KW.clear()
        if kind == "direct":
            NT.NN_KW["direct"] = True
        z = np.load(os.path.join(ROOT, "outputs", "predcomp_tune_cache", f"resid3_{y}.npz"))
        nd = NT.nn_draws(CF.mid(y - 1, 10), sorted(int(k) for k in z.files))
        np.savez_compressed(f, **{str(m): v.astype(np.float32) for m, v in nd.items()})
    z = np.load(f)
    return {int(k): np.sort(z[k].astype(float), 1) for k in z.files}


def main():
    truth = CF.deaths(CF.mid(2023, 12))
    tot = {g: 0.0 for g in GRID}
    by_year = {}
    for y in range(2010, 2018):
        z = np.load(os.path.join(ROOT, "outputs", "predcomp_tune_cache", f"resid3_{y}.npz"))
        L = {int(k): np.sort(z[k].astype(float), 1) for k in z.files}
        R, D = draws(y, "resid"), draws(y, "direct")
        yr = {}
        for g in GRID:
            a, b = g
            c = np.mean([crps_ensemble(truth[:, m], np.clip((1 - a - b) * L[m] + a * R[m] + b * D[m], 0, None)).mean() for m in L])
            yr[g] = c
            tot[g] += c / 8
        by_year[y] = {f"{a:.1f},{b:.1f}": round(v, 3) for (a, b), v in yr.items() if (a, b) in [(0.5, 0.0), (0.4, 0.1), (0.4, 0.2), (0.3, 0.2)]}
        print(y, by_year[y], flush=True)
    best = sorted(tot.items(), key=lambda kv: kv[1])[:10]
    res = {"dev_crps": {f"{a:.1f},{b:.1f}": round(v, 4) for (a, b), v in tot.items()},
           "best": [(f"{a:.1f},{b:.1f}", round(v, 4)) for (a, b), v in best], "by_year": by_year}
    print("two-way (0.5, 0):", round(tot[min(GRID, key=lambda g: abs(g[0] - 0.5) + abs(g[1]))], 4))
    print("best:", res["best"])
    json.dump(res, open(os.path.join(ROOT, "outputs", "cm_blend3_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
