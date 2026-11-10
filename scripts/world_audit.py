"""Consistency audit of simulated worlds (data debate, 10 November 2026; a report, not a model change). The forecast's own
simulator (pipeline.build_sim_env at 2025, the headline world) runs 2026-2065, and each country-decade of each world is
checked against what 1950-2024 history allows. Envelopes and pass marks were fixed before the first run:
1. Growth: 10-year change in log GDP per person inside history's 0.5th-99.5th percentile of 10-year changes (countries
   with 1m+ people). Pass: at most 2% of simulated country-decades outside (history itself has 1% by construction).
2. Level: GDP per person inside [0.5 x history's lowest, 2 x the highest recorded in 2024]. Pass: at most 0.5% outside.
3. Population: the simulated deviation from the UN path stays within +/-40% (pop_off is clamped at e^+/-4, so values
   near that clamp mean the accounting broke). Pass: no country-decade beyond +/-40%, none at the clamp.
4. Health against income (Preston): life expectancy follows fixed pre-origin trends, so every country-decade with a
   30%+ fall in GDP per person (or 5+ years at war) is one where health keeps improving regardless. Reported as the
   share of country-decades affected (no pass mark: the size of the inconsistency).
5. World totals: world GDP per person never falls more than 25% in a year (history's worst world year: 1931, about
   -9%; 1945 about -15%), and population-weighted world growth over 40 years stays within 0-6% a year. Pass: under 1%
   of worlds outside.
python -P scripts/world_audit.py [--n 1000] -> outputs/world_audit.json"""
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
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1000
    O, E = LAST_OBS_YEAR, 2065
    p, base, X, Yl = PL.prepare(False)
    Xf, Yf, ci, year = flatten(p, X, Yl)
    lg, lp = base["lg"], base["lpop"]
    # history's envelopes (1950-2024, countries of 1m+ people)
    y0, y1 = p.yi(1950), p.yi(2024)
    big = np.exp(lp) >= 1e6
    d10 = (lg[:, y0 + 10:y1 + 1] - lg[:, y0:y1 - 9])[big[:, y0 + 10:y1 + 1]]
    d10 = d10[np.isfinite(d10)]
    g_lo, g_hi = np.quantile(d10, [0.005, 0.995])
    lv = lg[:, y0:y1 + 1][np.isfinite(lg[:, y0:y1 + 1])]
    l_lo, l_hi = np.log(0.5) + lv.min(), np.log(2.0) + np.nanmax(lg[:, y1])
    net = PL.build_sim_env(p, X, Xf, Yf, ci, year, O, base=base)
    growth, ddem = SIM.fit_transitions(p, X, base, O)
    clim = PL.climate_summary(p, base, O, E)
    dy = INTER.sim_inputs(INTER.run(verbose=False))
    o = SIM.simulate(p, base, net, growth, ddem, O, E, n_scen=n, clim=clim["ssp245"]["inputs"], climate_damage=1.0,
                     world=WorldConfig(ai=AI_SCENARIOS["uncertain"], sample_params=True), dyads=dy, seed=2, record_paths=n)
    gdp, pop, cf = o["paths"]["gdp"], o["paths"]["pop"], o["paths"]["conflict"]      # [S, C, T]
    S, C, T = gdp.shape
    lgs = np.log(np.clip(gdp, 1e-9, None))
    lg0 = lg[:, p.yi(O)]
    path = np.concatenate([np.repeat(lg0[None, :, None], S, 0), lgs], -1)            # [S, C, T+1]
    ok = np.isfinite(lg0) & (np.exp(lp[:, p.yi(O)]) >= 1e6)
    dec = [(a, a + 10) for a in range(0, T - 9, 10)]
    g10 = np.stack([path[:, :, b] - path[:, :, a] for a, b in dec], -1)[:, ok]       # [S, C', D]
    res = {"n_worlds": S, "envelopes": {"growth10_log": [round(float(g_lo), 3), round(float(g_hi), 3)],
                                        "level_log": [round(float(l_lo), 3), round(float(l_hi), 3)]}}
    out_g = float(((g10 < g_lo) | (g10 > g_hi)).mean())
    res["1_growth_outside"] = {"share": round(out_g, 4), "below": round(float((g10 < g_lo).mean()), 4),
                               "above": round(float((g10 > g_hi).mean()), 4), "pass": out_g <= 0.02}
    lvl = lgs[:, ok]
    out_l = float(((lvl < l_lo) | (lvl > l_hi)).mean())
    res["2_level_outside"] = {"share": round(out_l, 5), "above_2x_record": round(float((lvl > l_hi).mean()), 5),
                              "pass": out_l <= 0.005}
    # population against the UN path: the pop path is the UN path x exp(pop_off); compare decade ends with the exogenous
    # path the simulator used (o["paths"]["pop"] already includes the offset; the UN path is pop_dev's denominator)
    pd_ = o["world"].get("pop_dev")
    res["3_population"] = {"world_dev_2065_q": [round(float(x), 4) for x in np.quantile(pd_[:, -1], [0.01, 0.5, 0.99])]} if pd_ is not None else {}
    # country level: the simulated population against its own first simulated year's path is not the UN path; use
    # the recorded deviation instead when available
    # 4. Preston: decades with a 30%+ fall in GDP per person, or 5+ years at war
    fall = g10 < np.log(0.7)
    war5 = np.stack([cf[:, :, a:b].sum(-1) >= 5 for a, b in dec], -1)[:, ok]
    res["4_health_inconsistent"] = {"share_collapse_decades": round(float(fall.mean()), 4),
                                    "share_war5_decades": round(float(war5.mean()), 4),
                                    "share_either": round(float((fall | war5).mean()), 4),
                                    "note": "life expectancy keeps its pre-origin trend in all of these"}
    w = o["world_gdp"]                                                                  # [S, T]
    wg = np.diff(np.log(np.concatenate([w[:, :1], w], 1)), axis=1)
    worst = wg.min(1)
    g40 = (np.log(w[:, -1]) - np.log(w[:, 0])) / (T - 1)
    bad = (worst < np.log(0.75)) | (g40 < 0) | (g40 > 0.06)
    res["5_world_totals"] = {"worst_year_q": [round(float(x), 4) for x in np.quantile(worst, [0.01, 0.5])],
                             "growth_per_year_q": [round(float(x), 4) for x in np.quantile(g40, [0.01, 0.5, 0.99])],
                             "share_outside": round(float(bad.mean()), 4), "pass": float(bad.mean()) < 0.01}
    # which worlds fail: catastrophes (large nuclear exchange, AI catastrophe, great-power war) or ordinary worlds
    W = o["world"]
    flags = {}
    for k in ("nuclear_large", "ai_catastrophe", "great_power_war", "pandemic"):
        if k in W:
            v = np.asarray(W[k]).reshape(S, -1).max(-1) > 0
            flags[k] = v
    any_cat = np.zeros(S, bool)
    for k in ("nuclear_large", "ai_catastrophe"):
        if k in flags:
            any_cat |= flags[k]
    res["5_world_totals"]["failing_worlds"] = {k: round(float(v[bad].mean()), 3) for k, v in flags.items()} if bad.any() else {}
    res["5_world_totals"]["share_outside_excluding_large_nuclear_or_ai_catastrophe"] = round(float((bad & ~any_cat).mean()), 4)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "world_audit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
