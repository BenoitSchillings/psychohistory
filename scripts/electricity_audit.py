"""Do autocracies overstate growth? An electricity cross-check (data debate, 10 November 2026; a report, not a model
change). Martinez (2022) finds GDP rising more steeply with night lights in autocracies, as if reported growth were
inflated; scripts/pe_test.py reproduced the sign on his 1992-2013 data with a wide interval. Electricity use is a second
physical measure that national statistics offices do not compile into GDP, available to 2024 (Ember, CC BY 4.0).

Specification (Martinez's, with electricity in place of lights): annual change in log real GDP (the panel's GDP per
person times population) on the annual change in log electricity demand (generation plus net imports), the same
interacted with autocracy, an autocracy dummy, country and year fixed effects, errors clustered by country.
Countries of 1m+ people, 2001-2024, rows with |dGDP| < 0.3 and |dElec| < 1.0 (Martinez's trimming in spirit).
Autocracy: the panel's democracy index (V-Dem electoral democracy) below 0.4 in the previous year.
The same regime coding is applied to Martinez's lights for comparison.

Pre-registered (10 November 2026, before the first run): overstatement is "supported" if the electricity interaction is
positive at p < 0.05 AND the lights interaction under the same coding is positive; "not supported" otherwise. Either way
nothing in the model changes here: a correction of autocracies' growth would need its own design, scored against more
than the national accounts it corrects.
python -P scripts/electricity_audit.py -> outputs/electricity_audit.json"""
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


def fe_ols(m, x, y="dgdp"):
    m = m.copy()
    m["x_aut"] = m[x] * m.aut
    X = pd.concat([m[[x, "x_aut", "aut"]], pd.get_dummies(m.year, prefix="y", drop_first=True).astype(float),
                   pd.get_dummies(m.code, prefix="c", drop_first=True).astype(float)], axis=1).assign(const=1.0)
    r = sm.OLS(m[y], X).fit(cov_type="cluster", cov_kwds={"groups": m.code.astype("category").cat.codes})
    b, c = float(r.params[x]), float(r.params["x_aut"])
    return {"elasticity_democracies": [round(b, 4), round(float(r.bse[x]), 4)],
            "interaction_autocracy": [round(c, 4), round(float(r.bse["x_aut"]), 4)], "p_interaction": round(float(r.pvalues["x_aut"]), 4),
            "autocracy_intercept": [round(float(r.params["aut"]), 4), round(float(r.bse["aut"]), 4)],
            "implied_overstatement": round((b + c) / b - 1, 3) if b > 0 else None, "n": int(len(m)),
            "countries": int(m.code.nunique()), "autocracy_rows": int(m.aut.sum())}


def main():
    from psychohistory import pipeline as PL
    p, base, X, Yl = PL.prepare(False)
    yrs = list(range(1990, 2025))
    rows = []
    for i, c in enumerate(p.codes):
        for y in yrs:
            t = p.yi(y)
            rows.append((c, y, base["lg"][i, t] + base["lpop"][i, t], base["lpop"][i, t], base["dem"][i, t - 1]))
    pan = pd.DataFrame(rows, columns=["code", "year", "lgdp", "lpop", "dem_prev"])
    e = pd.read_csv(os.path.join(ROOT, "data", "raw", "energy", "ember", "release_generation_yearly_global.csv"))
    e = e[(e["Area type"] == "Country or economy") & (e["Electricity source"] == "Demand")]
    e = e.rename(columns={"ISO 3 code": "code", "Year": "year", "Generation (TWh)": "twh"})[["code", "year", "twh"]]
    d = pan.merge(e, on=["code", "year"], how="inner").sort_values(["code", "year"])
    d = d[(np.exp(d.lpop) >= 1e6) & (d.twh > 0)]
    d["dgdp"] = d.groupby("code").lgdp.diff()
    d["dlpop"] = d.groupby("code").lpop.diff()
    d["delec"] = d.groupby("code").twh.transform(lambda s: np.log(s).diff())
    d["aut"] = (d.dem_prev < 0.4).astype(float)
    d = d.dropna(subset=["dgdp", "delec", "dem_prev"])
    d = d[(d.year >= 2001) & (d.dgdp.abs() < 0.3) & (d.delec.abs() < 1.0)]
    res = {"electricity": fe_ols(d, "delec")}
    print("electricity", json.dumps(res["electricity"]), flush=True)
    # lights, same regime coding
    m = pd.read_csv(os.path.join(ROOT, "data", "raw", "political_economy", "martinez_lights_gdp.csv")).sort_values(["iso3", "year"])
    m = m.rename(columns={"iso3": "code"})
    m["dgdp"] = m.groupby("code").ln_gdp_const_lcu.diff()
    m["dlight"] = m.groupby("code").ln_avdn_dmsp.diff()
    m = m.merge(pan[["code", "year", "dem_prev", "lpop"]], on=["code", "year"], how="inner")
    m = m[np.exp(m.lpop) >= 1e6]
    m["aut"] = (m.dem_prev < 0.4).astype(float)
    m = m.dropna(subset=["dgdp", "dlight", "dem_prev"])
    m = m[(m.dgdp.abs() < 0.3) & (m.dlight.abs() < 1.5)]
    res["lights_same_coding"] = fe_ols(m, "dlight")
    print("lights", json.dumps(res["lights_same_coding"]), flush=True)
    # what it would mean: mean reported growth of GDP per person in autocracies 2001-2024 and the implied physical rate
    a = d[d.aut == 1]
    g = float((a.dgdp - a.dlpop).mean())
    res["autocracies_mean_reported_growth_pc"] = round(g, 4)
    E, L = res["electricity"], res["lights_same_coding"]
    res["supported"] = bool(E["interaction_autocracy"][0] > 0 and E["p_interaction"] < 0.05 and L["interaction_autocracy"][0] > 0)
    print("supported:", res["supported"])
    json.dump(res, open(os.path.join(ROOT, "outputs", "electricity_audit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
