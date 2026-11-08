"""Two-regime AI growth, an exploration (state-of-the-art survey, 7 November 2026): task-exposure models give AI +0.05 to
+1.3 points of growth a year; substitution models give explosive growth once most tasks are automated. Seldon's
headline dividend sits in the first camp. Here the headline world (2025 origin, AI arrival uncertain) is run with and
without an explosive regime in 10% of worlds (WorldConfig.explosive_p; the dividend x3-6 without fading, energy cap
kept), same seed, to show what the excluded worlds look like. Not the headline.
python -P scripts/two_regime_ai.py [--n 4000] -> outputs/two_regime_ai.json"""
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
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 4000
    O = LAST_OBS_YEAR
    p, base, X, Yl = PL.prepare(False)
    Xf, Yf, ci, year = flatten(p, X, Yl)
    net = PL.build_sim_env(p, X, Xf, Yf, ci, year, O, base=base)
    growth, ddem = SIM.fit_transitions(p, X, base, O)
    clim = PL.climate_summary(p, base, O, 2065)
    dy = INTER.sim_inputs(INTER.run(verbose=False))
    res = {}
    for tag, ep in (("headline", 0.0), ("two_regime", 0.1)):
        o = SIM.simulate(p, base, net, growth, ddem, O, 2065, n_scen=n, clim=clim["ssp245"]["inputs"], climate_damage=1.0,
                         world=WorldConfig(ai=AI_SCENARIOS["uncertain"], sample_params=True, explosive_p=ep), dyads=dy, seed=2)
        gdp = o["world_gdp"][:, -1]
        r = {"gdp_2065_q": [round(float(x)) for x in np.quantile(gdp, [0.05, 0.25, 0.5, 0.75, 0.95])],
             "gdp_2065_mean": round(float(gdp.mean())), "share_over_100k": round(float((gdp > 100000).mean()), 4),
             "great_power_war": round(float(o["world"]["great_power_war"].max(-1).mean()), 4),
             "civil_conflict_2065": round(float(np.median(o["world_conflict"][:, -1])), 4),
             "democracy_2065": round(float(np.median(o["world_dem"][:, -1])), 4)}
        res[tag] = r
        print(tag, r, flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "two_regime_ai.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
