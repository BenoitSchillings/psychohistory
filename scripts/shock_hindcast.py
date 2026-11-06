"""Does the simulator propagate a named shock correctly? (second debate, 6 November 2026: the 1-10 year coupled-shock
service is worth offering only if conditional hindcasts pass.) From the year before four historical wars, the
forecast's own simulator (pipeline.build_sim_env at that origin, world layer on, AI off) runs twice with the same
seed: as is, and with the war forced in its year (WorldConfig.forced, as the crisis explorer does). Over the first
three years, the oil price change and the GDP-per-person change of the countries most affected are scored (CRPS)
against what happened, with and without the forced war.

Pass (pre-registered before the first run): forcing the war brings both the oil score and the affected countries'
GDP score closer to what happened in at least 3 of the 4 episodes. Otherwise the shock explorer stays an illustration.
python -P scripts/shock_hindcast.py [--n 1000] -> outputs/shock_hindcast.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

EPISODES = {
    "yom_kippur_1973": {"origin": 1972, "forced": [("EGY", "ISR", 1973), ("ISR", "SYR", 1973)],
                        "affected": ["EGY", "ISR", "SYR", "JOR"]},
    "kuwait_1990": {"origin": 1989, "forced": [("IRQ", "KWT", 1990)], "affected": ["IRQ", "KWT", "SAU", "JOR"]},
    "iraq_2003": {"origin": 2002, "forced": [("IRQ", "USA", 2003)], "affected": ["IRQ", "KWT", "JOR", "SYR"]},
    "ukraine_2022": {"origin": 2021, "forced": [("RUS", "UKR", 2022)], "affected": ["UKR", "RUS", "MDA", "BLR"]},
}
H = 3


def crps(x, y):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float(np.mean(np.abs(x - y)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2)


def main():
    from psychohistory import pipeline as PL, simulate as SIM, world as WM, interstate as INTER
    from psychohistory.backtest import flatten
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1000
    p, base, X, Yl = PL.prepare(False)
    Xf, Yf, ci, year = flatten(p, X, Yl)
    df = INTER.build(verbose=False)
    oil = base["oil_price"][0]
    res = {}
    for key, ep in EPISODES.items():
        O = ep["origin"]
        i0, i1 = p.yi(O), p.yi(O + H)
        net = PL.build_sim_env(p, X, Xf, Yf, ci, year, O, base=base)
        growth, ddem = SIM.fit_transitions(p, X, base, O)
        dy = INTER.sim_inputs(INTER.run(verbose=False, origin=O, backtests=False, df=df))
        run = lambda forced: SIM.simulate(p, base, net, growth, ddem, O, O + H, n_scen=n, demography="extrapolate",
                                          record_paths=n, seed=O + 23, dyads=dy,
                                          world=WM.WorldConfig(ai=WM.AI_SCENARIOS["none"], sample_params=True, forced=tuple(forced)))
        r = {}
        for tag, forced in (("as_is", []), ("forced", ep["forced"])):
            o = run(forced)
            so = o["world"]["oil"][:, -1] - oil[i0]
            ao = float(oil[i1] - oil[i0])
            cs = []
            for c in ep["affected"]:
                j = p.codes.index(c) if c in p.codes else None
                if j is None or not np.isfinite(base["lg"][j, i1]) or not np.isfinite(base["lg"][j, i0]):
                    continue
                sg = np.log(o["paths"]["gdp"][:, j, -1]) - base["lg"][j, i0]
                if np.isfinite(sg).all():
                    cs.append((c, crps(sg, float(base["lg"][j, i1] - base["lg"][j, i0])),
                               float(np.median(sg)), float(base["lg"][j, i1] - base["lg"][j, i0])))
            r[tag] = {"oil_crps": round(crps(so, ao), 4), "oil_median": round(float(np.median(so)), 3), "oil_actual": round(ao, 3),
                      "gdp_crps_mean": round(float(np.mean([x[1] for x in cs])), 4) if cs else None,
                      "gdp": {c: {"crps": round(a, 4), "median": round(m, 3), "actual": round(y, 3)} for c, a, m, y in cs}}
        r["oil_better"] = r["forced"]["oil_crps"] < r["as_is"]["oil_crps"]
        r["gdp_better"] = (r["forced"]["gdp_crps_mean"] or 9) < (r["as_is"]["gdp_crps_mean"] or 9)
        res[key] = r
        print(key, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "gdp"}) for k, v in r.items()}), flush=True)
    ok = sum(r["oil_better"] and r["gdp_better"] for r in res.values())
    res["verdict"] = {"episodes_both_better": ok, "of": len(EPISODES), "pass": ok >= 3}
    print(res["verdict"])
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "shock_hindcast.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
