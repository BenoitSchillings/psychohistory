"""Working-age drag on falling shares as a sampled assumption (simulate.WORKFORCE_FALL = (0.57, 1.22); third debate,
9 November 2026). Not testable on history: the 1970-2009 hindcasts saw rising working-age shares. Here its effect on the
2025-origin forecast: same seed and network, 2,000 worlds, with AI uncertain (the headline) and without AI.
python -P scripts/workforce_fall_test.py [--n 2000] -> outputs/workforce_fall_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np


def main():
    from psychohistory import pipeline as PL, simulate as SIM, interstate as INTER
    from psychohistory.backtest import flatten
    from psychohistory.world import AI_SCENARIOS, WorldConfig
    from psychohistory.data import LAST_OBS_YEAR
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 2000
    O = LAST_OBS_YEAR
    p, base, X, Yl = PL.prepare(False)
    Xf, Yf, ci, year = flatten(p, X, Yl)
    net = PL.build_sim_env(p, X, Xf, Yf, ci, year, O, base=base)
    growth, ddem = SIM.fit_transitions(p, X, base, O)
    clim = PL.climate_summary(p, base, O, 2065)
    dy = INTER.sim_inputs(INTER.run(verbose=False))
    codes = list(p.codes)
    g0 = np.exp(base["lg"][:, p.yi(O)])
    rich = np.where(np.nan_to_num(g0) > 40000)[0]
    res = {}
    for ai in ("uncertain", "none"):
        for tag, wf in (("beta_1.22", None), ("fall_0.57_1.22", (0.57, 1.22))):
            SIM.WORKFORCE_FALL = wf
            o = SIM.simulate(p, base, net, growth, ddem, O, 2065, n_scen=n, clim=clim["ssp245"]["inputs"], climate_damage=1.0,
                             world=WorldConfig(ai=AI_SCENARIOS[ai], sample_params=True), dyads=dy, seed=2, record_paths=n)
            gdp = o["paths"]["gdp"]                                  # [S, C, T]
            med30 = np.median(gdp[:, :, 2030 - O - 1], 0)
            w = o["world_gdp"]
            r = {"world_2030_median": round(float(np.median(w[:, 2030 - O]))), "world_2065_median": round(float(np.median(w[:, -1]))),
                 "world_2065_q05_q95": [round(float(x)) for x in np.quantile(w[:, -1], [0.05, 0.95])],
                 "rich_shrinking_2030": int((med30[rich] < g0[rich]).sum()), "rich_n": int(len(rich)),
                 **{f"{c}_2030_median": round(float(med30[codes.index(c)])) for c in ("USA", "DEU", "JPN", "KOR", "CHN")},
                 "civil_conflict_2065": round(float(np.median(o["world_conflict"][:, -1])), 4)}
            res[f"{ai}:{tag}"] = r
            print(ai, tag, r, flush=True)
    SIM.WORKFORCE_FALL = None
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "workforce_fall_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
