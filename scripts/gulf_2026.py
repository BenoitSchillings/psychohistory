"""What the recorded 2026 Gulf war does inside the model: paired runs (same seed) from the 2025 origin.

A: the forecast as published -- all fighting recorded in UCDP candidate events to September 2026 happens in 2026,
   including Iran vs Israel and the United States (war level) and Iraq vs Israel and the United States, and the
   2026 oil price is the observed January-September average (World Bank Pink Sheet: +22% real on 2025).
B: the same worlds without the Gulf pairs (Russia-Ukraine, Afghanistan-Pakistan and US-Venezuela kept; the Gulf
   pairs, including Iran-Israel already at war in 2025, held at peace in 2026) and with the 2026 oil price simulated
   instead of observed.
C: A with the 2026 oil price simulated (what the model alone would have said about oil, given the wars).
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import interstate as I
from psychohistory import pipeline as PL
from psychohistory.data import LAST_OBS_YEAR
from psychohistory.simulate import climate_inputs, simulate
from psychohistory.world import AI_SCENARIOS, WorldConfig

O, E = LAST_OBS_YEAR, 2035
p, base, X, Yl = PL.prepare(False)
Xf, Yf, ci, year = PL.flatten(p, X, Yl)
Ytr, tr = PL.train_mask(Yf, year, O)
net = PL.PsychoNet(members=8, **PL.NET_TUNED).fit(Xf[tr], Ytr[tr], ci[tr])
growth, ddem = PL.fit_transitions(p, X, base, O)
clim = climate_inputs(p, base, O, E - O, "ssp245")
dy = I.sim_inputs(I.run(verbose=False, backtests=False))
GULF = {("IRN", "ISR"), ("IRN", "USA"), ("IRQ", "ISR"), ("IRQ", "USA")}
dyB = {**dy, "observed": [o for o in dy["observed"] if (o[0], o[1]) not in GULF]}
cfg = lambda **k: WorldConfig(ai=AI_SCENARIOS["uncertain"], sample_params=True, **k)
run = lambda d, c: simulate(p, base, net, growth, ddem, O, E, n_scen=4000, clim=clim, climate_damage=1.0, seed=17, world=c, dyads=d)
R = {"A_recorded": run(dy, cfg()), "B_no_gulf_war": run(dyB, cfg(oil_observed=(), peace=tuple((a, b, O + 1) for a, b in GULF))), "C_oil_simulated": run(dy, cfg(oil_observed=()))}
years = R["A_recorded"]["years"].tolist()
oil0 = base["oil_price"][0][p.yi(O)]
ix = lambda c: p.codes.index(c)
out = {"years": years, "note": __doc__.strip()}
for k, o in R.items():
    w = o["world"]
    out[k] = {"oil_median_2025eq100": np.round(100 * np.exp(np.median(w["oil"], 0) - oil0), 1).tolist(),
              "oil_p90": np.round(100 * np.exp(np.quantile(w["oil"], 0.9, 0) - oil0), 1).tolist(),
              "world_gdp_median": np.round(np.median(o["world_gdp"], 0), 0).tolist(),
              "world_conflict_mean": np.round(o["world_conflict"].mean(0), 4).tolist(),
              "gpw_by": np.round(np.maximum.accumulate(w["great_power_war"], axis=1).mean(0), 4).tolist(),
              "gdp_country_median": {c: np.round(o["gdp_q"][ix(c), :, 2], 0).tolist() for c in
                                     ("USA", "CHN", "IND", "JPN", "KOR", "DEU", "PAK", "EGY", "BGD", "TUR", "SAU", "ARE", "IRQ", "KWT", "QAT", "NOR", "RUS", "IRN", "ISR")
                                     if c in p.codes},
              "conflict_country": {c: np.round(o["conflict"][ix(c)], 3).tolist() for c in ("IRN", "IRQ", "LBN", "SYR", "YEM", "JOR", "SAU", "EGY") if c in p.codes},
              "pairs_any_by_end": {f"{a}-{b}": float(w["pair_prob"][i]) for i, (a, b) in enumerate(zip(dy["now"].a, dy["now"].b))
                                   if (a, b) in {("IRN", "SAU"), ("ARE", "IRN"), ("IRN", "QAT"), ("IRN", "KWT"), ("IRN", "IRQ"), ("ISR", "LBN"), ("ISR", "SYR"), ("IRN", "TUR")}}}
json.dump(out, open(os.path.join(PL.OUT, "gulf_2026.json"), "w"), indent=1)
A, B, C = (out[k] for k in ("A_recorded", "B_no_gulf_war", "C_oil_simulated"))
for y in (0, 1, 4, 9):
    print(years[y], "oil A/B/C", A["oil_median_2025eq100"][y], B["oil_median_2025eq100"][y], C["oil_median_2025eq100"][y],
          "| world GDP A/B-1", round(A["world_gdp_median"][y] / B["world_gdp_median"][y] - 1, 4), "C/B-1", round(C["world_gdp_median"][y] / B["world_gdp_median"][y] - 1, 4))
for c in A["gdp_country_median"]:
    print(c, "GDP 2027 vs no Gulf war: A", round(A["gdp_country_median"][c][1] / B["gdp_country_median"][c][1] - 1, 4), "C", round(C["gdp_country_median"][c][1] / B["gdp_country_median"][c][1] - 1, 4),
          "| 2030 A", round(A["gdp_country_median"][c][4] / B["gdp_country_median"][c][4] - 1, 4))
print("pairs by 2035 A:", A["pairs_any_by_end"]); print("pairs by 2035 B:", B["pairs_any_by_end"])
print("conflict 2030 A vs B:", {c: (A["conflict_country"][c][4], B["conflict_country"][c][4]) for c in A["conflict_country"]})
print("GPW by 2035 A/B/C", A["gpw_by"][-1], B["gpw_by"][-1], C["gpw_by"][-1])
