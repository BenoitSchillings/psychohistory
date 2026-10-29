"""Quantile recalibration of the monthly country model (LightGBM + neural blend). On the dev windows 2010-2017 the
blended forecast's randomised PIT is collected; a monotone map of quantile levels (the empirical quantiles of the dev
PIT, shrunk toward the identity by weight A) is then applied to every forecast: new draw k = old ensemble quantile at
level g(u_k). Dev score: leave-one-year-out within 2010-2017. Test: the seldon_v7 submissions (2018-2023 and live) remapped
with the map fitted on all of 2010-2017.
python -P scripts/cm_recal_test.py dev              -> outputs/cm_recal_test.json (dev CRPS by shrink weight)
python -P scripts/cm_recal_test.py run NAME A       (write challenge submission NAME from seldon_v7 with weight A)"""
import glob
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
LEVELS = np.linspace(0, 1, 201)
WS = (0.0, 0.25, 0.5, 0.75, 1.0)


def pit(draws, y, rng):
    """randomised PIT of y in each row's ensemble (ties, e.g. at zero, spread uniformly)"""
    lo = (draws < y[:, None]).mean(1)
    hi = (draws <= y[:, None]).mean(1)
    return lo + rng.random(len(y)) * (hi - lo)


def fit_map(p, a):
    """level map g on LEVELS: g(u) = (1 - a) u + a Q_pit(u)"""
    return (1 - a) * LEVELS + a * np.quantile(p, LEVELS)


def remap(draws, g):
    s = np.sort(draws, 1)
    n = s.shape[1]
    u = (np.arange(n) + 0.5) / n
    lev = np.interp(u, LEVELS, g)
    idx = np.clip(np.round(lev * n - 0.5).astype(int), 0, n - 1)
    return s[:, idx]


def dev_blends():
    """blended (w = 0.5) dev draws per year, cached"""
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for y in range(2010, 2018):
        f = os.path.join(CACHE, f"blend_{y}.npz")
        if not os.path.exists(f):
            z = np.load(os.path.join(ROOT, "outputs", "predcomp_tune_cache", f"resid3_{y}.npz"))
            g = {int(k): z[k].astype(float) for k in z.files}
            nd = NT.nn_draws(CF.mid(y - 1, 10), sorted(g))
            np.savez_compressed(f, **{str(m): np.clip(NT.blend(g[m], nd[m], 0.5), 0, None).astype(np.float32) for m in g})
        z = np.load(f)
        out[y] = {int(k): z[k].astype(float) for k in z.files}
        print("dev blend", y, flush=True)
    return out


def dev():
    truth = CF.deaths(CF.mid(2023, 12))
    B = dev_blends()
    rng = np.random.default_rng(0)
    P = {y: np.concatenate([pit(B[y][m], truth[:, m], rng) for m in sorted(B[y])]) for y in B}
    tot = {a: 0.0 for a in WS}
    n = 0
    by_year = {}
    for y in B:
        p_tr = np.concatenate([P[k] for k in P if k != y])
        yr = {}
        for a in WS:
            g = fit_map(p_tr, a)
            c = np.mean([crps_ensemble(truth[:, m], remap(B[y][m], g)).mean() for m in B[y]])
            yr[str(a)] = round(float(c), 3)
            tot[a] += c
        by_year[y] = yr
        print(y, yr, flush=True)
    allp = np.concatenate(list(P.values()))
    res = {"dev_crps": {str(a): round(tot[a] / len(B), 4) for a in WS}, "by_year": by_year,
           "pit_deciles": np.histogram(allp, np.linspace(0, 1, 11))[0].tolist()}
    print(res["dev_crps"], res["pit_deciles"])
    json.dump(res, open(os.path.join(ROOT, "outputs", "cm_recal_test.json"), "w"), indent=1)


def run(name, a):
    import pandas as pd
    from predcomp_run import write
    truth = CF.deaths(CF.mid(2023, 12))
    B = dev_blends()
    rng = np.random.default_rng(0)
    g = fit_map(np.concatenate([pit(B[y][m], truth[:, m], rng) for y in B for m in sorted(B[y])]), a)
    cs = CF.countries()
    for f in sorted(glob.glob(os.path.join(ROOT, "outputs", "predcomp", "seldon_v7", "cm", "window=*", "*.parquet"))):
        wname = os.path.basename(os.path.dirname(f)).split("=")[1]
        P = pd.read_parquet(f).reset_index().pivot_table(index=["month_id", "country_id"], columns="draw", values="outcome")
        dr = {int(m): remap(P.loc[m].reindex(cs.views_country_id).to_numpy(float), g)
              for m in P.index.get_level_values(0).unique()}
        write(name, wname, dr)
        print("wrote", name, wname, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "dev":
        dev()
    else:
        run(sys.argv[2], float(sys.argv[3]))
