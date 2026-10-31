"""Full-system hindcasts: the simulator with the world layer on (interstate wars, finance, oil, food, migration,
pandemics; every assumption sampled per world, AI off) from past origins, with models trained only on data before each
origin. Saves, per origin, each world's sampled assumptions and summary statistics of its path next to what happened,
for testing the assumption ranges against history (scripts/world_history_match.py).

python -P scripts/world_hindcast.py TAG HORIZON ORIGIN [ORIGIN ...] [--n N] [--cfg '{"WorldConfig field": value}']
    [--consts '{"module.CONST": value}']
-> outputs/world_hindcast/TAG_<origin>.npz
python -P scripts/world_hindcast.py --compare TAG_A TAG_B ORIGIN [ORIGIN ...]   (CRPS of each statistic, B against A)"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "world_hindcast")


def world_series(lg, w):
    """population-weighted world GDP per person [.., T] from country log GDP per person [.., C, T]"""
    return np.log((np.exp(lg) * w[:, None]).sum(-2))


def crps(x, y):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float(np.mean(np.abs(x - y)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2)


def compare(a, b, origins):
    rows = {}
    for O in origins:
        A, B = (np.load(os.path.join(OUT, f"{t}_{O}.npz")) for t in (a, b))
        for j, k in enumerate(A["names"]):
            if k not in B["names"]:
                continue
            jb = list(B["names"]).index(k)
            y = float(A["obs"][j])
            rows.setdefault(str(k), []).append((crps(A["sim"][:, j], y), crps(B["sim"][:, jb], y),
                                                float(np.median(A["sim"][:, j])), float(np.median(B["sim"][:, jb])), y))
    print(f"{b} vs {a}, origins {origins}")
    for k, r in rows.items():
        r = np.array(r)
        print(f"  {k:16s} CRPS {r[:, 0].mean():.4f} -> {r[:, 1].mean():.4f} ({100 * (r[:, 1].mean() / r[:, 0].mean() - 1):+.1f}%, "
              f"B better at {(r[:, 1] < r[:, 0]).sum()}/{len(r)})  median {np.round(r[:, 2], 3).tolist()} -> "
              f"{np.round(r[:, 3], 3).tolist()}  actual {np.round(r[:, 4], 3).tolist()}")


def main():
    from psychohistory import interstate as INTER, pipeline as PL, simulate as SIM, world as WM
    from psychohistory.backtest import flatten, train_mask
    from psychohistory.data import LAST_OBS_YEAR
    from psychohistory.models import NET_TUNED, PsychoNet

    import json
    argv = list(sys.argv[1:])
    n = int(argv.pop(argv.index("--n") + 1)) if "--n" in argv else 2000
    wkw = json.loads(argv.pop(argv.index("--cfg") + 1)) if "--cfg" in argv else {}      # WorldConfig settings to test
    consts = json.loads(argv.pop(argv.index("--consts") + 1)) if "--consts" in argv else {}   # {"module.CONST": value}
    import importlib
    for k, v in consts.items():
        mod, name = k.rsplit(".", 1)
        setattr(importlib.import_module(f"psychohistory.{mod}"), name, v)
    args = [a for a in argv if not a.startswith("--")]
    tag, H, origins = args[0], int(args[1]), [int(a) for a in args[2:]]
    os.makedirs(OUT, exist_ok=True)
    p, base, X, Yl = PL.prepare(False)
    Xf, Yf, ci, year = flatten(p, X, Yl)
    df = INTER.build(verbose=False)
    for O in origins:
        f = os.path.join(OUT, f"{tag}_{O}.npz")
        if os.path.exists(f):
            continue
        E = min(O + H, LAST_OBS_YEAR)
        Ytr, tr = train_mask(Yf, year, O)
        net = PsychoNet(members=8, **NET_TUNED).fit(Xf[tr], Ytr[tr], ci[tr])
        growth, ddem = SIM.fit_transitions(p, X, base, O)
        dy = INTER.sim_inputs(INTER.run(verbose=False, origin=O, backtests=False, df=df))
        cfg = WM.WorldConfig(ai=WM.AI_SCENARIOS["none"], sample_params=True, **wkw)
        o = SIM.simulate(p, base, net, growth, ddem, O, E, n_scen=n, demography="extrapolate", record_paths=n,
                         seed=O + 11, world=cfg, dyads=dy)
        i0, i1 = p.yi(O), p.yi(E)
        T = i1 - i0
        cf, gdp, dem = o["paths"]["conflict"], o["paths"]["gdp"], o["paths"]["dem"]           # [S, C, T]
        act = base["conflict"][:, i0:i1 + 1]
        lga = base["lg"][:, i0:i1 + 1]
        pop0 = np.exp(base["lpop"][:, i0])
        okc = np.isfinite(act).all(1)
        okg = np.isfinite(lga).all(1) & np.isfinite(pop0) & np.isfinite(np.log(gdp)).all((0, 2))
        okd = np.isfinite(base["dem"][:, i1]) & np.isfinite(base["dem"][:, i0]) & np.isfinite(dem[:, :, -1]).all(0)
        w = np.where(okg, pop0, 0.0)
        w = w / w.sum()
        sim, obs = {}, {}
        # world GDP per person: growth over the horizon, how often it falls, its yearly volatility
        ws = world_series(np.concatenate([np.repeat(lga[None, okg, :1], n, 0), np.log(gdp[:, okg])], -1), w[okg])   # [S, T+1]
        wa = world_series(lga[okg], w[okg])
        gs, ga = np.diff(ws, axis=1), np.diff(wa)
        sim["gdp_growth"], obs["gdp_growth"] = ws[:, -1] - ws[:, 0], wa[-1] - wa[0]
        sim["gdp_falls"], obs["gdp_falls"] = (gs < 0).mean(1), (ga < 0).mean()
        sim["gdp_vol"], obs["gdp_vol"] = gs.std(1), ga.std()
        # spread of country growth (convergence / divergence)
        cg = np.log(gdp[:, okg, -1]) - lga[okg, 0][None]
        sim["gdp_dispersion"], obs["gdp_dispersion"] = cg.std(1), (lga[okg, -1] - lga[okg, 0]).std()
        # civil conflict: share of country-years, new conflicts, endings
        full = np.concatenate([np.repeat(act[None, okc, :1], n, 0), cf[:, okc]], -1)
        sim["conflict_share"], obs["conflict_share"] = cf[:, okc].mean((1, 2)), act[okc, 1:].mean()
        sim["onsets"] = ((full[:, :, 1:] > 0.5) & (full[:, :, :-1] < 0.5)).sum((1, 2)).astype(float)
        obs["onsets"] = float(((act[okc, 1:] == 1) & (act[okc, :-1] == 0)).sum())
        # coups
        ca = base["coup_attempts"][:, i0 + 1:i1 + 1]
        okp = np.isfinite(ca).all(1)
        sim["coups"], obs["coups"] = o["paths"]["coup"][:, okp].sum((1, 2)).astype(float), float((ca[okp] > 0).sum())
        # democracy: population-weighted change
        wd = np.where(okd, pop0, 0.0)
        wd = wd / wd.sum()
        sim["dem_change"] = ((dem[:, okd, -1] - base["dem"][okd, i0][None]) * wd[okd]).sum(1)
        obs["dem_change"] = float(((base["dem"][okd, i1] - base["dem"][okd, i0]) * wd[okd]).sum())
        # interstate conflicts active per year (UCDP), oil price change
        wr = o["world"]
        real = df[(df.year > O) & (df.year <= E)].groupby("year").ev_ucdp.apply(lambda s_: (s_ == 1).sum())
        sim["interstate"], obs["interstate"] = wr["active"].mean(1), float(real.reindex(range(O + 1, E + 1)).fillna(0).mean())
        if "oil" in wr:
            oil_hist = base["oil_price"][0]                                     # log real price
            sim["oil_change"] = wr["oil"][:, -1] - oil_hist[i0]
            obs["oil_change"] = float(oil_hist[i1] - oil_hist[i0])
        # country level: where each country's realised GDP and democracy fall in its simulated band
        cg_act = lga[okg, -1] - lga[okg, 0]
        q = np.quantile(cg, [0.05, 0.25, 0.5, 0.75, 0.95], 0)
        dsim = dem[:, okd, -1]
        qd = np.quantile(dsim, [0.05, 0.95], 0)
        dact = base["dem"][okd, i1]
        country = {"gdp_in90": float(((cg_act >= q[0]) & (cg_act <= q[4])).mean()),
                   "gdp_in50": float(((cg_act >= q[1]) & (cg_act <= q[3])).mean()),
                   "gdp_pit_low": float((cg_act < q[0]).mean()), "gdp_pit_high": float((cg_act > q[4]).mean()),
                   "gdp_median_bias": float(np.median(q[2] - cg_act)),
                   "dem_in90": float(((dact >= qd[0]) & (dact <= qd[1])).mean())}
        params = wr.get("params", {})
        np.savez_compressed(f, origin=O, end=E, T=T, names=np.array(sorted(sim)),
                            sim=np.stack([sim[k] for k in sorted(sim)], 1), obs=np.array([obs[k] for k in sorted(sim)]),
                            pnames=np.array(sorted(params)), params=np.stack([params[k] for k in sorted(params)], 1),
                            country=json.dumps(country))
        print(O, E, {k: (round(float(np.median(sim[k])), 4), round(float(obs[k]), 4),
                         round(float((sim[k] < obs[k]).mean()), 2)) for k in sorted(sim)}, country, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3], [int(o) for o in sys.argv[4:]])
    else:
        main()
