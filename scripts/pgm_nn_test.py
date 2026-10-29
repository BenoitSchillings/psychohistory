"""Neural quantile blend for the 55 km grid model (as for the country model, scripts/cm_nn_test.py): LightGBM and the
MLP (cm_fatal.NNQuantile, same weighted rows) averaged rank by rank with weight w, then the local tail bound.
Dev windows 2014-2017 (pgm_tune.py's), CRPS / ignorance / interval score.  python -P scripts/pgm_nn_test.py -> outputs/pgm_nn_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from psychohistory import cm_fatal as CF, pgm_fatal as PG
from predcomp_eval import crps_ensemble, ignorance, interval

ROOT = os.path.join(os.path.dirname(__file__), "..")
WS = (0.0, 0.25, 0.5)


def run_year(y, truth):
    origin = CF.mid(y - 1, 10)
    months = [CF.mid(y, m) for m in range(1, 13)]
    D = PG.deaths(origin)
    Dc = CF.deaths(origin)
    F = PG.features(D, Dc)
    names = list(F)
    rng = np.random.default_rng(0)
    hs_all = sorted({m - origin for m in months})
    act = (F["ld12"] > 0) | (F["wide12"] > 0)
    act24 = act | np.concatenate([np.zeros((act.shape[0], 12), bool), act[:, :-12]], 1)
    Xs, ys, ws = [], [], []
    for t in range(CF.mid(1995, 1), origin - min(hs_all) + 1):
        hs = [h for h in hs_all if t + h <= origin]
        if not hs:
            continue
        a = act24[:, t]
        quiet = ~a & (rng.random(len(a)) < 0.05)
        idx = np.where(a | quiet)[0]
        h = rng.choice(hs, len(idx))
        Xs.append(np.stack([F[f][idx, t] for f in names] + [h.astype(np.float32)], 1))
        ys.append(np.log1p(D[idx, t + h]))
        ws.append(np.where(a[idx], 1.0, 20.0))
    X, yv, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    import lightgbm as lgb
    g = CF.QuantileModel(PG.QUANTS, {"min_child_samples": 200, "n_estimators": 300})
    g.models = [lgb.LGBMRegressor(objective="quantile", alpha=float(q), **g.params).fit(X, yv, sample_weight=w) for q in g.q]
    g.cap = float(np.log1p(1.5 * D.max()))
    nn_ = CF.NNQuantile(PG.QUANTS).fit(X, yv, sample_weight=w)
    nn_.cap = g.cap
    N = D.shape[0]
    bound = np.maximum(PG.TAIL_C0, PG.TAIL_K * PG.local_max(D, origin))[:, None]
    res = {wt: [[], [], []] for wt in WS}
    for tm in months:
        Xt = np.stack([F[f][:, origin] for f in names] + [np.full(N, tm - origin, np.float32)], 1)
        a, b = np.sort(g.sample(Xt), 1), np.sort(nn_.sample(Xt), 1)
        for wt in WS:
            Xb = np.minimum((1 - wt) * a + wt * b, bound)
            for i in range(0, N, 20000):
                yy, xx = truth[i:i + 20000, tm], Xb[i:i + 20000]
                res[wt][0].append(crps_ensemble(yy, xx)); res[wt][1].append(ignorance(yy, xx)); res[wt][2].append(interval(yy, xx))
    return {str(wt): [float(np.concatenate(v).mean()) for v in r] for wt, r in res.items()}


truth = PG.deaths(CF.mid(2023, 12))
out = {}
for y in (2014, 2015, 2016, 2017):
    out[y] = run_year(y, truth)
    print(y, out[y], flush=True)
mean = {str(wt): [round(float(np.mean([out[y][str(wt)][k] for y in out])), 4) for k in range(3)] for wt in WS}
print("mean [crps, ign, mis]", mean)
json.dump({"by_year": out, "mean": mean}, open(os.path.join(ROOT, "outputs", "pgm_nn_test.json"), "w"), indent=1)
