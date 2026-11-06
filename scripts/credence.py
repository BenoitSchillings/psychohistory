"""Long-run numbers as credences with a range (second debate, 6 November 2026). The headline integrates over every
assumption at once, so "45% great-power war" mixes what the assumptions say with chance. Here the forecast's own
simulator (pipeline.build_sim_env at 2025, the headline world: AI arrival uncertain, every assumption sampled) runs in
blocks of worlds that share one draw of every assumption (WorldConfig.param_block; also the model-family draw and the
pair shift). Each block's share is a probability given its assumptions; the spread of those probabilities across
blocks is the range the assumptions alone produce. The attribution is then scored on block probabilities, not on
single yes-or-no worlds, so the assumptions' share is not drowned by chance (the binary-R^2 artefact).
python -P scripts/credence.py [--blocks 200 --per 50] -> outputs/credence.json"""
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
    a = sys.argv
    M = int(a[a.index("--blocks") + 1]) if "--blocks" in a else 200
    K = int(a[a.index("--per") + 1]) if "--per" in a else 50
    O, E = LAST_OBS_YEAR, 2065
    p, base, X, Yl = PL.prepare(False)
    Xf, Yf, ci, year = flatten(p, X, Yl)
    net = PL.build_sim_env(p, X, Xf, Yf, ci, year, O, base=base)
    growth, ddem = SIM.fit_transitions(p, X, base, O)
    clim = PL.climate_summary(p, base, O, E)
    dy = INTER.sim_inputs(INTER.run(verbose=False))
    cfg = WorldConfig(ai=AI_SCENARIOS["uncertain"], sample_params=True, param_block=K)
    o = SIM.simulate(p, base, net, growth, ddem, O, E, n_scen=M * K, clim=clim["ssp245"]["inputs"], climate_damage=1.0,
                     world=cfg, dyads=dy, seed=2)
    w = o["world"]
    blk = lambda x: np.asarray(x, float)[:M * K].reshape(M, K)
    outs = {
        "great_power_war": blk(w["great_power_war"].max(-1)),
        "great_power_war_majors": blk(w["great_power_war_cow"].max(-1)),
        "nuclear_use": blk(w["nuclear_use"].max(-1)),
        "civil_conflict_2065_above_25pct": blk(o["world_conflict"][:, -1] > 0.25),
    }
    gdp = blk(o["world_gdp"][:, -1])
    res = {"blocks": M, "worlds_per_block": K, "origin": O}
    for k, v in outs.items():
        pb = v.mean(1)                                     # probability given each block's assumptions
        res[k] = {"overall": round(float(v.mean()), 4),
                  "range_across_assumptions_5_95": [round(float(x), 4) for x in np.quantile(pb, [0.05, 0.95])],
                  "quartiles": [round(float(x), 4) for x in np.quantile(pb, [0.25, 0.5, 0.75])],
                  # share of the spread of single worlds that the assumptions explain, measured properly: variance
                  # of block probabilities (less its sampling noise) over the total Bernoulli variance
                  "assumption_share": round(float(max(pb.var() - (pb * (1 - pb)).mean() / (K - 1), 0) / max(v.var(), 1e-12)), 4)}
    med = np.median(gdp, 1)
    res["world_gdp_per_person_2065"] = {"overall_median": round(float(np.median(gdp))),
                                        "block_medians_5_95": [round(float(x)) for x in np.quantile(med, [0.05, 0.95])]}
    # which assumptions move the block probability of great-power war: rank correlation of each sampled assumption
    # (one value per block) with the block probability
    from scipy.stats import spearmanr
    pb = outs["great_power_war"].mean(1)
    drivers = {}
    for k, v in w.get("params", {}).items():
        v = np.asarray(v, float).ravel()[:M * K].reshape(M, K)[:, 0]
        if np.isfinite(v).all() and v.std() > 0:
            drivers[k] = round(float(spearmanr(v, pb)[0]), 3)
    res["great_power_war_drivers_rank_corr"] = dict(sorted(drivers.items(), key=lambda kv: -abs(kv[1]))[:8])
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "credence.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
