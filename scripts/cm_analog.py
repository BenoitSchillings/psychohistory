"""An analog (trajectory-matching) model for country-month fatalities, alone and blended with the production model.
For each country at the origin, its last L months of log(1 + deaths) (recent months weighted more, half-life 6 months)
are matched against every country's L-month window that ends at least h months before the origin; the K nearest
windows (ties broken at random) each give what happened h months later, as the change from the window's own 3-month
mean (the production model's residual target), applied to the country's own 3-month mean. 1,000 draws from the K
analogs. The idea is ShapeFinder's (the 2023/24 VIEWS challenge winner on CRPS): it follows a war's recent course,
where the production model expects large wars to fade, and the two errors differ (we lead 2018-2021, it leads 2023).
Blend: rank-by-rank average in log(1 + deaths), as the production blend (cm_fatal.train_and_forecast).
Settings (L, K, blend weight) are chosen on the dev windows 2010-2017 only; 2018-2023 and live 2024-25 are the test.

python -P scripts/cm_analog.py prod                  cache the production draws for the dev windows (GPU)
python -P scripts/cm_analog.py dev                   -> outputs/cm_analog_test.json
python -P scripts/cm_analog.py run NAME L K W        challenge submission: production seldon_v9 draws blended with W
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from psychohistory import cm_fatal as CF

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "outputs", "predcomp_tune_cache")
HALF = 6.0          # half-life (months) of the weight of each lag in the distance
START = CF.mid(1990, 1)
N_DRAWS = 1000


def analog_draws(origin, months, final_through=None, L=24, K=50, seed=0):
    """{month_id: [C, N_DRAWS]} analog draws from data known at `origin` (the production function, cm_fatal.analog_draws)"""
    return CF.analog_draws(CF.deaths(origin, final_through), origin, months, L=L, K=K, n_draws=N_DRAWS, seed=seed)


def blend(a, b, w):
    return np.expm1((1 - w) * np.log1p(np.sort(a, 1)) + w * np.log1p(np.sort(b, 1)))


def prod():
    """production (LightGBM + neural, log blend) draws for the dev windows, cached"""
    for y in range(2010, 2018):
        f = os.path.join(CACHE, f"prod9_{y}.npz")
        if os.path.exists(f):
            continue
        o = CF.mid(y - 1, 10)
        blend0, CF.ANALOG_BLEND = CF.ANALOG_BLEND, 0.0       # (the production model without the analog member)
        dr, _ = CF.train_and_forecast(o, [CF.mid(y, m) for m in range(1, 13)])
        CF.ANALOG_BLEND = blend0
        np.savez_compressed(f, **{str(k): v.astype(np.float32) for k, v in dr.items()})
        print("cached", y, flush=True)


def dev():
    from predcomp_eval import crps_ensemble
    truth = CF.deaths(CF.mid(2023, 12))
    cfgs = [(12, 25), (12, 100), (12, 200), (24, 100), (24, 200)]
    ws = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    tot = {(c, w): 0.0 for c in cfgs for w in ws}
    by_year = {}
    n = 0
    for y in range(2010, 2018):
        z = np.load(os.path.join(CACHE, f"prod9_{y}.npz"))
        g = {int(k): z[k].astype(float) for k in z.files}
        o = CF.mid(y - 1, 10)
        yr = {}
        for c in cfgs:
            ad = analog_draws(o, sorted(g), L=c[0], K=c[1])
            for w in ws:
                s = sum(crps_ensemble(truth[:, m], blend(g[m], ad[m], w)).sum() for m in g)
                tot[(c, w)] += s
                yr[f"L{c[0]}K{c[1]}w{w}"] = round(s / (len(g) * len(truth)), 3)
        n += len(g) * len(truth)
        by_year[y] = yr
        print(y, {k: v for k, v in yr.items() if k.endswith("w0.0") or k.endswith("w0.5") or k.endswith("w1.0")}, flush=True)
    res = {f"L{c[0]}K{c[1]}w{w}": round(v / n, 4) for (c, w), v in tot.items()}
    best = min(res, key=res.get)
    print("dev CRPS:", json.dumps(res, indent=0)[:2000])
    print("best:", best, res[best], "production alone:", res["L12K25w0.0"])
    json.dump({"dev_crps": res, "best": best, "by_year": by_year}, open(os.path.join(ROOT, "outputs", "cm_analog_test.json"), "w"), indent=1)


def run(name, L, K, w, src="seldon_v9"):
    """challenge submission: the production draws of `src` blended with analog draws"""
    import glob
    import pandas as pd
    from predcomp_run import write
    cs = CF.countries()
    windows = [(f"Y{y}", CF.mid(y - 1, 10), None, [f"Y{y}"]) for y in range(2018, 2024)]
    windows.append(("LIVE", CF.mid(2024, 4), CF.mid(2023, 12), ["Y2024", "Y2025"]))
    for wname, origin, ft, wys in windows:
        g = {}
        for wy in wys:
            f = glob.glob(os.path.join(ROOT, "outputs", "predcomp", src, "cm", f"window={wy}", "*.parquet"))[0]
            P = pd.read_parquet(f).reset_index().pivot_table(index=["month_id", "country_id"], columns="draw", values="outcome")
            for m in P.index.get_level_values(0).unique():
                g[int(m)] = P.loc[m].reindex(cs.views_country_id).to_numpy(float)
        ad = analog_draws(origin, sorted(g), ft, L=L, K=K)
        dr = {m: blend(g[m], ad[m], w) for m in g}
        if wname == "LIVE":
            write(name, "Y2024", {m: v for m, v in dr.items() if m <= CF.mid(2024, 12)})
            write(name, "Y2025", {m: v for m, v in dr.items() if m > CF.mid(2024, 12)})
        else:
            write(name, wname, dr)
        print("wrote", name, wname, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prod":
        prod()
    elif sys.argv[1] == "dev":
        dev()
    else:
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
