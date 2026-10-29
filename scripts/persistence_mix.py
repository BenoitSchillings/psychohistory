"""High-intensity persistence mix for the country-month fatality model. The quantile model expects very large wars to
fade (most spikes in its training history did); on the live 2024/25 windows Ukraine and Gaza did not, and almost all of
the gap to the "last" and "conflictology" benchmarks is those two countries. Mix: for countries whose mean deaths over
the last 3 months at the origin are >= K, a share pi of the draws is replaced by persistence draws (one of the last 3
months, times exp(N(0, s))). K, pi, s are chosen on the 2010-2017 dev windows (cached draws of the production model,
predcomp_tune resid3); 2018-2023 and the live window are the test.
python -P scripts/persistence_mix.py tune   -> outputs/persistence_mix.json
python -P scripts/persistence_mix.py apply SRC DST   (rewrite submission SRC as DST with the chosen mix)"""
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd

from psychohistory import cm_fatal as CF
from predcomp_eval import crps_ensemble

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "outputs", "persistence_mix.json")


def mix(X, D, origin, K, pi, s, seed=0):
    """X [C, n] draws for one target month; D deaths as known at the origin"""
    if pi == 0:
        return X
    rng = np.random.default_rng(seed)
    last3 = D[:, origin - 2:origin + 1]
    hi = last3.mean(1) >= K
    if not hi.any():
        return X
    X = X.copy()
    n = X.shape[1]
    k = int(round(pi * n))
    pick = last3[hi][:, rng.integers(0, 3, size=(k,))] if False else last3[hi][np.arange(hi.sum())[:, None], rng.integers(0, 3, size=(hi.sum(), k))]
    X[hi, :k] = pick * np.exp(s * rng.standard_normal((hi.sum(), k)))
    return X


def tune():
    truth = CF.deaths(CF.mid(2023, 12))
    cache = os.path.join(ROOT, "outputs", "predcomp_tune_cache")
    dev = {}
    for y in range(2010, 2018):
        z = np.load(os.path.join(cache, f"resid3_{y}.npz"))
        dev[y] = {int(k): z[k].astype(float) for k in z.files}
    res = []
    for K in (100, 300, 1000):
        for pi in (0, 0.25, 0.5, 0.75):
            for s in (0.3, 0.6):
                if pi == 0 and (K, s) != (100, 0.3):
                    continue
                tot, n = 0.0, 0
                for y, dr in dev.items():
                    o = CF.mid(y - 1, 10)
                    for m, X in dr.items():
                        c = crps_ensemble(truth[:, m], np.clip(mix(X, truth, o, K, pi, s), 0, None))
                        tot += c.sum(); n += len(c)
                res.append({"K": K, "pi": pi, "s": s, "dev_crps": round(tot / n, 4)})
                print(res[-1], flush=True)
    best = min(res, key=lambda r: r["dev_crps"])
    json.dump({"grid": res, "chosen": best}, open(OUT, "w"), indent=1)
    print("chosen", best)


def apply(src, dst):
    ch = json.load(open(OUT))["chosen"]
    cs = CF.countries()
    for f in glob.glob(os.path.join(ROOT, "outputs", "predcomp", src, "cm", "window=*", "*.parquet")):
        w = os.path.basename(os.path.dirname(f)).split("=")[1]
        y = int(w[1:])
        if y >= 2024:                      # live: data to April 2024, candidate months as in the run (vintage-corrected)
            o, D = CF.mid(2024, 4), CF.deaths(CF.mid(2024, 4), CF.mid(2023, 12))
        else:
            o = CF.mid(y - 1, 10)
            D = CF.deaths(o)
        s = pd.read_parquet(f)
        P = s.reset_index().pivot_table(index=["month_id", "country_id"], columns="draw", values="outcome")
        rows = []
        for m in sorted(P.index.get_level_values(0).unique()):
            X = P.loc[m].reindex(cs.views_country_id).to_numpy(float)
            X = mix(X, D, o, ch["K"], ch["pi"], ch["s"], seed=m)
            n = X.shape[1]
            rows.append(pd.DataFrame({"month_id": m, "country_id": np.repeat(cs.views_country_id.to_numpy(), n),
                                      "draw": np.tile(np.arange(n), len(cs)), "outcome": X.ravel().astype(np.float32)}))
        d = os.path.join(ROOT, "outputs", "predcomp", dst, "cm", f"window={w}")
        os.makedirs(d, exist_ok=True)
        pd.concat(rows).set_index(["month_id", "country_id", "draw"]).to_parquet(os.path.join(d, f"{dst}_cm_{w}.parquet"))
        print("wrote", dst, w)


if __name__ == "__main__":
    tune() if sys.argv[1] == "tune" else apply(sys.argv[2], sys.argv[3])
