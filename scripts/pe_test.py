"""Political-economy candidates from the independent review (data in data/raw/political_economy).

1. Sovereign defaults (Bank of Canada - Bank of England database, defaults of $100m or more, 1975-2023): what follows a
   default start (coups, autocratization, protest waves, war onset, collapse, within 2 years) and what predicts one
   (short-term debt / reserves, external debt / GNI, debt service / exports, growth, banking crisis, income, regime).
2. Autocracies' GDP (Martinez 2022 data: night lights and GDP, 1992-2013): the elasticity of official GDP growth to
   lights growth in Not Free vs Free countries (country and year effects); the ratio is the overstatement factor.
3. Alliance reliability (Leeds 2003; COW/IWD wars, CJL 2026): share of invoked alliances honored, by era.
4. Public trust (IVS via OWID: confidence in government, armed forces; trust in people): does it predict coups,
   autocratization, protest waves or war onset (sparse survey years carried forward up to 5 years)?
python -P scripts/pe_test.py  -> outputs/pe_test.json
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import statsmodels.api as sm

from psychohistory.data import RAW, ROOT

PE = RAW / "political_economy"
src = open(os.path.join(os.path.dirname(__file__), "structure_test.py")).read().split("# 1. sanctions")[0]
exec(compile(src, "structure_test_head", "exec"))        # panel, rows_for(), fit() as in structure_test.py
out = {}

# 1. sovereign defaults
sd = pd.read_csv(PE / "sovereign_defaults_boc_boe.csv").set_index(["iso3", "year"])
wd = pd.read_csv(RAW / "structure" / "wdi_external_debt_country_year.csv").set_index(["iso3", "year"])
def dflt(i, c, t, ti):
    s_ = sd.default_start_ge100mn.get((c, t), np.nan) if (c, t) in sd.index else np.nan
    s1 = sd.default_start_ge100mn.get((c, t + 1), np.nan) if (c, t + 1) in sd.index else np.nan
    ind = sd.in_default_ge100mn.get((c, t), np.nan) if (c, t) in sd.index else np.nan
    g = lambda col: wd[col].get((c, t), np.nan) if (c, t) in wd.index else np.nan
    st, ed, ds = g("st_debt_pct_reserves"), g("ext_debt_pct_gni"), g("debt_service_pct_exports")
    return {"default_start": s_, "default_next": s1 if ind == 0 else np.nan,
            "st_res": np.log1p(min(st, 1000) / 100) if np.isfinite(st) else np.nan,
            "ext_gni": np.log1p(ed / 100) if np.isfinite(ed) else np.nan,
            "ds_exp": np.log1p(ds / 100) if np.isfinite(ds) else np.nan,
            "fin_now": float(np.nan_to_num(b["fin_crisis"][i, ti]) > 0)}
d = rows_for(1975, 2022, dflt)
CT = ["lg", "dem", "conflict", "g_prev"]
out["default_consequences"] = {y: fit(d, y, ["default_start"], CT if y != "onset2" else ["lg", "dem", "g_prev"])
                               for y in ("coup2", "autoc2", "mob_up", "onset2", "collapse2")}
out["default_hazard"] = fit(d, "default_next", ["st_res", "ext_gni", "ds_exp", "fin_now"], CT)
out["default_rate"] = round(float(d.default_next.mean()), 4)
print("1. defaults", json.dumps(out["default_consequences"]), "\n hazard", json.dumps(out["default_hazard"]), flush=True)

# 2. Martinez: lights elasticity by regime
m = pd.read_csv(PE / "martinez_lights_gdp.csv").sort_values(["iso3", "year"])
m["dgdp"] = m.groupby("iso3").ln_gdp_const_lcu.diff()
m["dlight"] = m.groupby("iso3").ln_avdn_dmsp.diff()
m["nf"] = (m.fh_status.astype(str).str.upper().str.startswith("N")).astype(float)
m["pf"] = (m.fh_status.astype(str).str.upper().str.startswith("P")).astype(float)
m = m.dropna(subset=["dgdp", "dlight"])
m = m[(m.dgdp.abs() < 0.3) & (m.dlight.abs() < 1.5)]
m["dl_nf"], m["dl_pf"] = m.dlight * m.nf, m.dlight * m.pf
X = pd.concat([m[["dlight", "dl_nf", "dl_pf", "nf", "pf"]], pd.get_dummies(m.year, prefix="y", drop_first=True).astype(float),
               pd.get_dummies(m.iso3, prefix="c", drop_first=True).astype(float)], axis=1).assign(const=1.0)
r = sm.OLS(m.dgdp, X).fit(cov_type="cluster", cov_kwds={"groups": m.iso3.astype("category").cat.codes})
e_f = float(r.params["dlight"])
out["martinez"] = {k: [round(float(r.params[k]), 4), round(float(r.bse[k]), 4)] for k in ("dlight", "dl_nf", "dl_pf")}
out["martinez"]["overstatement_not_free"] = round((e_f + float(r.params["dl_nf"])) / e_f - 1, 3)
out["martinez"]["n"] = int(len(m))
print("2. martinez", json.dumps(out["martinez"]), flush=True)

# 3. alliance reliability
a = pd.read_csv(PE / "alliance_reliability_ally_war.csv")
a["era"] = (a.war_start_year // 25 * 25).astype("Int64")
out["alliance_reliability"] = {s_: {str(k): [round(float(v["honor"].mean()), 3), int(len(v))] for k, v in g.groupby("era")}
                               for s_, g in a.groupby("source")}
out["alliance_reliability_post1945"] = {s_: [round(float(g[g.war_start_year >= 1946].honor.mean()), 3), int((g.war_start_year >= 1946).sum())]
                                        for s_, g in a.groupby("source")}
print("3. reliability", json.dumps(out["alliance_reliability_post1945"]), flush=True)

# 4. public trust
tr = pd.read_csv(PE / "public_trust_owid.csv").sort_values(["iso3", "year"])
cols = ["ivs_confidence_government_pct", "ivs_confidence_armed_forces_pct", "ivs_trust_most_people_pct"]
filled = {}
for c_, g in tr.groupby("iso3"):
    g = g.set_index("year")[cols].reindex(range(1980, 2024)).ffill(limit=5)
    for y, row in g.iterrows():
        filled[(c_, y)] = row.to_dict()
def trust(i, c, t, ti):
    v = filled.get((c, t), {})
    return {"conf_gov": v.get(cols[0], np.nan) / 100, "conf_army": v.get(cols[1], np.nan) / 100, "trust": v.get(cols[2], np.nan) / 100}
d = rows_for(1981, 2022, trust)
CT = ["lg", "dem", "conflict", "g_prev"]
out["trust"] = {y: fit(d, y, ["conf_gov", "conf_army", "trust"], CT if y != "onset2" else ["lg", "dem", "g_prev"])
                for y in ("coup2", "autoc2", "mob_up", "onset2")}
print("4. trust", json.dumps(out["trust"]))
json.dump(out, open(ROOT / "outputs" / "pe_test.json", "w"), indent=1)
