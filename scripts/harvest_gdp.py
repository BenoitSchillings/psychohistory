"""Harvests and output (10 November 2026). In the simulator, harvest losses beyond history (warming, eruptions,
nuclear winter, AMOC) reach output only through food prices; a world that loses half its harvest keeps its income. This
estimates how much a bad harvest costs a country's growth, from 1961-2024.

Specification (fixed before the run): annual growth of GDP per person on the cereal-yield residual (log yield minus its
centred 11-year median, food.yield_residuals), alone and interacted with the farm share of GDP (World Bank via OWID,
the previous available year), with country and year fixed effects, errors clustered by country; countries of 1m+
people, |growth| < 0.3. Year effects absorb world-wide harvest years, so the estimate is the country's own loss.
Also reported: the same with the residual of the year before (lasting losses).
Use: if the interaction is positive at p < 0.05, the simulator's channel is d log GDP = beta x farm share x (the part of
the harvest change not replayed from history), as an estimated elasticity applied out of sample (an assumption about
size beyond history's range); otherwise the channel stays out.
python -P scripts/harvest_gdp.py -> outputs/harvest_gdp.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    from psychohistory import pipeline as PL
    from psychohistory.food import yield_residuals
    from psychohistory.data import RAW
    p, base, X, Yl = PL.prepare(False)
    r = yield_residuals(p, 2024)
    a = pd.read_csv(RAW / "agriculture-share-gdp.csv")
    ag = a.pivot_table(index="code", columns="year", values=a.columns[-1]).reindex(index=list(p.codes), columns=list(p.years))
    ag = ag.T.ffill(limit=5).T.to_numpy(float) / 100
    rows = []
    for i, c in enumerate(p.codes):
        for y in range(1962, 2025):
            t = p.yi(y)
            rows.append((c, y, base["lg"][i, t] - base["lg"][i, t - 1], r[i, t], r[i, t - 1], ag[i, t - 1], base["lpop"][i, t]))
    d = pd.DataFrame(rows, columns=["code", "year", "g", "r", "r1", "ag", "lpop"]).dropna()
    d = d[(np.exp(d.lpop) >= 1e6) & (d.g.abs() < 0.3)]
    d["r_ag"], d["r1_ag"] = d.r * d.ag, d.r1 * d.ag
    out = {}
    for name, cols in (("contemporaneous", ["r", "r_ag"]), ("with_lag", ["r", "r_ag", "r1", "r1_ag"])):
        Xm = pd.concat([d[cols], pd.get_dummies(d.year, prefix="y", drop_first=True).astype(float),
                        pd.get_dummies(d.code, prefix="c", drop_first=True).astype(float)], axis=1).assign(const=1.0)
        m = sm.OLS(d.g, Xm).fit(cov_type="cluster", cov_kwds={"groups": d.code.astype("category").cat.codes})
        out[name] = {k: [round(float(m.params[k]), 4), round(float(m.bse[k]), 4), round(float(m.pvalues[k]), 4)] for k in cols}
        out[name]["n"] = int(len(d))
        print(name, json.dumps(out[name]), flush=True)
    b = out["contemporaneous"]["r_ag"]
    out["channel_supported"] = bool(b[0] > 0 and b[2] < 0.05)
    out["median_farm_share"] = round(float(d.ag.median()), 3)
    # the implied loss of a 10% harvest shortfall at a 25% farm share
    out["loss_10pct_harvest_at_25pct_farm"] = round(float(b[0] * 0.25 * np.log(0.9)), 4)
    print(json.dumps({k: out[k] for k in ("channel_supported", "median_farm_share", "loss_10pct_harvest_at_25pct_farm")}))
    json.dump(out, open(os.path.join(ROOT, "outputs", "harvest_gdp.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
