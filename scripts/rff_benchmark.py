"""External benchmark for the long-run GDP fans: the RFF Socioeconomic Projections (Rennert et al. 2022, Nature; CC BY 4.0;
probabilistic GDP per person from Mueller, Stock & Watson's Bayesian dynamic factor model and expert elicitation, and
UN-style probabilistic population, Raftery & Sevcikova 2021). RFF-SPs contain no climate damage and no AI scenario.

World GDP per person (market exchange rates) relative to 2025, 2045 and 2065, from the RFF trajectories extracted in
data/raw/benchmarks/rffsps/extracted/pop_income (a random subset of the 10,000 is enough for quantiles), against this
model's fans in outputs/forecast.json: the headline (AI uncertain) and the no-AI case, and the same without climate
damage where the forecast has it.
python -P scripts/rff_benchmark.py -> outputs/rff_benchmark.json"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
Q = [0.05, 0.25, 0.5, 0.75, 0.95]


def main():
    fs = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "benchmarks", "rffsps", "extracted", "pop_income", "*.feather")))
    W = []
    for f in fs:
        d = pd.read_feather(f)
        g = d[d.Year.isin([2025, 2045, 2065])].groupby("Year")[["GDP", "Pop"]].sum()
        W.append((g.GDP / g.Pop).to_dict())
    W = pd.DataFrame(W)
    out = {"rff_n": len(W), "rff": {}, "model": {}}
    for y in (2045, 2065):
        out["rff"][y] = [round(float(v), 3) for v in np.quantile(W[y] / W[2025], Q)]
    fc = json.load(open(os.path.join(ROOT, "outputs", "forecast.json")))
    g0 = float(fc["history"]["world_gdp"][-1])
    years = fc["world"].get("years", list(range(2026, 2066)))
    for name, q in (("headline_ai_uncertain", fc["ai"]["uncertain"]["world_gdp_q"]), ("no_ai", fc["ai"]["none"]["world_gdp_q"]),
                    ("ssp245_no_climate_damage", fc["scenarios"].get("ssp245_nodamage", {}).get("world_gdp_q"))):
        if q is None:
            continue
        q = np.asarray(q)
        out["model"][name] = {y: [round(float(v) / g0, 3) for v in q[:, years.index(y)]] for y in (2045, 2065)}
    out["quantiles"] = Q
    json.dump(out, open(os.path.join(ROOT, "outputs", "rff_benchmark.json"), "w"), indent=1)
    print("world GDP per person relative to 2025 (5/25/50/75/95%)")
    print("  RFF-SPs (no climate damage, no AI):", out["rff"])
    for k, v in out["model"].items():
        print(f"  {k}:", v)


if __name__ == "__main__":
    main()
