"""Effect of the catastrophe tails (simulate.CATASTROPHE_TAILS: nuclear winter and VEI 7 eruptions; 10 November 2026).
Assumptions, not estimates, so there is no test to pass: this reports what switching them on does to the headline
worlds, paired (same seed, same worlds, the tails' own generator), in the forecast's configuration
(pipeline.build_sim_env at 2025, AI uncertain, parameters sampled, SSP2-4.5).
python -P scripts/catastrophe_tails.py [--n 2000] -> outputs/catastrophe_tails.json"""
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
    O, E = LAST_OBS_YEAR, 2065
    p, base, X, Yl = PL.prepare(False)
    Xf, Yf, ci, year = flatten(p, X, Yl)
    net = PL.build_sim_env(p, X, Xf, Yf, ci, year, O, base=base)
    growth, ddem = SIM.fit_transitions(p, X, base, O)
    clim = PL.climate_summary(p, base, O, E)
    dy = INTER.sim_inputs(INTER.run(verbose=False))
    q = lambda a, qs=(0.5, 0.9, 0.99): [round(float(x), 4) for x in np.quantile(a, qs)]
    res = {"n_worlds": n}
    for on in (False, True):
        SIM.CATASTROPHE_TAILS = on
        o = SIM.simulate(p, base, net, growth, ddem, O, E, n_scen=n, clim=clim["ssp245"]["inputs"], climate_damage=1.0,
                         world=WorldConfig(ai=AI_SCENARIOS["uncertain"], sample_params=True), dyads=dy, seed=2, record_paths=0)
        W = o["world"]
        w = np.asarray(o["world_gdp"])
        pop = np.asarray(W["pop_dev"])
        ed = np.asarray(W["excess_deaths"])[:, -1] / 1e6
        large = np.asarray(W["nuclear_large"]).reshape(n, -1).max(-1) > 0
        g40 = (np.log(w[:, -1]) - np.log(w[:, 0])) / (w.shape[1] - 1)
        r = {"world_gdp_pc_2065_q05_50_95": [round(float(x)) for x in np.quantile(w[:, -1], [0.05, 0.5, 0.95])],
             "world_gdp_pc_2065_q01": round(float(np.quantile(w[:, -1], 0.01))),
             "pop_dev_2065_q01_50": [round(float(x), 4) for x in np.quantile(pop[:, -1], [0.01, 0.5])],
             "excess_deaths_m_q50_90_99": q(ed), "worlds_shrinking_40y": round(float((g40 < 0).mean()), 4),
             "large_nuclear_exchange": round(float(large.mean()), 4)}
        if on:
            fam = np.asarray(W["famine_deaths"]).reshape(n, -1).sum(-1) / 1e6
            v7 = np.asarray(W["vei7"]).reshape(n, -1).max(-1) > 0
            r["vei7_worlds"] = round(float(v7.mean()), 4)
            r["famine_deaths_m_given_large_q10_50_90"] = q(fam[large], (0.1, 0.5, 0.9)) if large.any() else None
            r["famine_deaths_m_all_q50_95_99"] = q(fam, (0.5, 0.95, 0.99))
            r["peak_winter_harvest_loss_given_large_q10_50_90"] = q(np.asarray(W["winter_loss"]).reshape(n, -1).max(-1)[large], (0.1, 0.5, 0.9)) if large.any() else None
        res["on" if on else "off"] = r
        print("tails", on, json.dumps(r), flush=True)
    SIM.CATASTROPHE_TAILS = False
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "catastrophe_tails.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
