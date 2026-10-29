"""Structural candidates suggested by the independent review, tested on history (data in data/raw/structure).

Country-year logits/OLS with year effects and controls, clustered by country (helpers as in intervention_test.py):
1. sanctions (TIES 1945-2005 imposed cases, then IST 1990-2018 economic sanctions) -> growth, collapse, autocratization
2. rebel fragmentation (UCDP dyadic: number of rebel groups) -> war ending next year
3. outside state support for rebels (UCDP ESD 1975-2017, confirmed) -> war ending next year
4. term-limit evasion attempts (Ginsburg, Melton & Elkins, to 2006) -> coups, autocratization, protest waves
5. economic complexity (Atlas ECI, SITC) -> 5-year growth, with out-of-sample check
6. coup-proofing (De Bruin SSF counterweights, 1960-2010) -> coup attempts, coup success given an attempt, war onset
7. external debt fragility (WDI IDS short-term debt / reserves, debt service / exports) x the US real rate change
   -> banking-crisis start, collapse
8. coups in same-language / colonial-tie countries last year (CEPII Gravity) beyond coups in nearby countries
   (within 1,500 km) -> coup attempt next year
python -P scripts/structure_test.py  -> outputs/structure_test.json
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

from psychohistory.data import RAW, ROOT, get_panel
from psychohistory.features import base_arrays
from psychohistory.simulate import working_age_share
from psychohistory import risks

ST = RAW / "structure"
p = get_panel()
b = base_arrays(p)
Y = list(p.years)
codes = list(p.codes)
lg, conf = b["lg"], np.nan_to_num(b["conflict"])
coups = np.nan_to_num(b["coup_attempts"])
front = np.nanmax(np.where(np.isfinite(lg), lg, -np.inf), 0)
lw = working_age_share(p)
out = {}


def fit(d, y, cols, ctrl, kind="logit", split=None):
    sub = d.dropna(subset=[y] + cols + ctrl)
    if kind == "logit":
        sub = sub[sub.groupby("year")[y].transform("max") > 0]
    fe = pd.get_dummies(sub.year, prefix="y", drop_first=True).astype(float)
    X = pd.concat([sub[cols + ctrl], fe], axis=1).assign(const=1.0)
    X = X.loc[:, X.std() > 0].assign(const=1.0)
    kw = dict(cov_type="cluster", cov_kwds={"groups": sub.c.astype("category").cat.codes})
    try:
        m = (sm.Logit(sub[y], X).fit(disp=0, maxiter=300, **kw) if kind == "logit" else sm.OLS(sub[y], X).fit(**kw))
    except Exception as e:
        return repr(e)[:100]
    r = {k: [round(float(m.params[k]), 4), round(float(m.bse[k]), 4)] for k in cols if k in m.params}
    r["n"] = int(len(sub))
    if kind == "logit":
        r["events"] = int(sub[y].sum())
    if split:
        tr, te = sub.year <= split, sub.year > split + 4
        for tag, cc in (("without", ctrl), ("with", cols + ctrl)):
            Z = sm.add_constant(sub[cc])
            mo = sm.OLS(sub[y][tr], Z[tr]).fit()
            r[f"oos_rmse_{tag}"] = round(float(np.sqrt(np.mean((mo.predict(Z[te]) - sub[y][te]) ** 2))), 5)
    return r


def rows_for(t0, t1, extra):
    rows = []
    for i, c in enumerate(codes):
        for t in range(t0, t1):
            ti = Y.index(t)
            if not np.isfinite(lg[i, ti]) or not np.isfinite(b["dem"][i, ti]) or not np.isfinite(lg[i, ti - 1]):
                continue
            r = {"c": c, "year": t, "lg": lg[i, ti], "dem": b["dem"][i, ti], "conflict": float(conf[i, ti] > 0),
                 "g_prev": float(np.clip(lg[i, ti] - lg[i, ti - 1], -.3, .3)),
                 "conf_years10": int(np.sum(conf[i, max(0, ti - 9):ti + 1] > 0)),
                 "bd": np.log1p(np.nan_to_num(b["battle_deaths_pc"][i, ti]) * 1e5),
                 "coup_recent": float(np.nan_to_num(b["coup_recent"][i, ti]) > 0),
                 "g1": float(np.clip(lg[i, ti + 1] - lg[i, ti], -.3, .3)) if np.isfinite(lg[i, ti + 1]) else np.nan,
                 "end1": float(conf[i, ti + 1] == 0) if conf[i, ti] > 0 else np.nan,
                 "onset2": float(np.sum(conf[i, ti + 1:ti + 3]) > 0) if conf[i, ti] == 0 else np.nan,
                 "coup1": float(coups[i, ti + 1] > 0), "coup2": float(np.sum(coups[i, ti + 1:ti + 3]) > 0),
                 "collapse2": float(np.nanmin(lg[i, ti + 1:ti + 3]) - lg[i, ti] < -0.05) if np.isfinite(lg[i, ti + 2]) else np.nan,
                 "autoc2": float(b["dem"][i, ti + 2] - b["dem"][i, ti] < -0.1) if np.isfinite(b["dem"][i, ti + 2]) else np.nan,
                 "coup_success1": float(np.nan_to_num(b["coups_success"][i, ti + 1]) > 0) if coups[i, ti + 1] > 0 else np.nan,
                 "fin_start": float(np.nan_to_num(b["fin_crisis"][i, ti + 1]) > 0) if np.nan_to_num(b["fin_crisis"][i, ti]) == 0 else np.nan,
                 "mob_up": float(np.nanmax(b["mass_mobilization"][i, ti + 1:ti + 3]) - b["mass_mobilization"][i, ti] > 1.0)
                 if np.isfinite(b["mass_mobilization"][i, ti]) else np.nan}
            r.update(extra(i, c, t, ti))
            rows.append(r)
    return pd.DataFrame(rows)


def lookup(path, cols):
    x = pd.read_csv(ST / path).set_index(["iso3", "year"])
    return {k: x[k].to_dict() for k in cols}


# 1. sanctions
ties = lookup("sanctions_ties_country_year.csv", ["n_imposed_active", "comprehensive"])
ist = lookup("sanctions_ist_country_year.csv", ["economic_sanctions", "comprehensive_trade_embargo"])
def sanc(i, c, t, ti):
    if t < 1990:
        v = ties["n_imposed_active"].get((c, t))
        return {"sanct": np.nan if v is None else float(v > 0), "sanct_comp": float(ties["comprehensive"].get((c, t), 0) or 0)}
    v = ist["economic_sanctions"].get((c, t))
    return {"sanct": np.nan if v is None else float(v > 0), "sanct_comp": float(ist["comprehensive_trade_embargo"].get((c, t), 0) or 0)}
d = rows_for(1950, 2018, sanc)
CT = ["lg", "dem", "conflict", "g_prev"]
out["sanctions"] = {"growth": fit(d, "g1", ["sanct", "sanct_comp"], CT, "ols"),
                    "collapse2": fit(d, "collapse2", ["sanct", "sanct_comp"], CT),
                    "autoc2": fit(d, "autoc2", ["sanct", "sanct_comp"], CT),
                    "onset2": fit(d, "onset2", ["sanct"], ["lg", "dem", "g_prev"])}
print("1. sanctions", json.dumps(out["sanctions"]), flush=True)

# 2-3. rebels: fragmentation and outside support
frag = lookup("ucdp_rebel_fragmentation_country_year.csv", ["n_rebel_groups"])
esd = lookup("ucdp_external_support_country_year.csv", ["rebel_ext_support_state", "rebel_ext_support_alleged_only"])
def rebels(i, c, t, ti):
    n = frag["n_rebel_groups"].get((c, t), 0)
    s_ = esd["rebel_ext_support_state"].get((c, t))
    al = esd["rebel_ext_support_alleged_only"].get((c, t), 0)
    return {"lrebels": np.log1p(n or 0), "ext_sup": np.nan if s_ is None or t > 2017 or t < 1975 else float(s_ > 0 and not al)}
d = rows_for(1950, 2023, rebels)
dc = d[d.conflict == 1]
CT = ["lg", "dem", "conf_years10", "bd"]
out["rebels"] = {"fragmentation": fit(dc, "end1", ["lrebels"], CT), "external_support": fit(dc, "end1", ["ext_sup"], CT),
                 "both": fit(dc, "end1", ["lrebels", "ext_sup"], CT)}
print("2-3. rebels", json.dumps(out["rebels"]), flush=True)

# 4. term-limit evasion
ev = lookup("ccp_term_limit_evasion_country_year.csv", ["leader_attempted_evasion"])
def evasion(i, c, t, ti):
    v = [ev["leader_attempted_evasion"].get((c, t - k)) for k in range(3)]
    vv = [x for x in v if x is not None and np.isfinite(x)]
    return {"evasion3": float(any(x > 0 for x in vv)) if t <= 2006 else np.nan}
d = rows_for(1950, 2007, evasion)
CT = ["lg", "dem", "conflict", "g_prev", "coup_recent"]
out["term_limit_evasion"] = {y: fit(d, y, ["evasion3"], CT if y != "onset2" else ["lg", "dem", "g_prev"])
                             for y in ("coup2", "autoc2", "mob_up", "onset2")}
print("4. evasion", json.dumps(out["term_limit_evasion"]), flush=True)

# 5. economic complexity
eci = lookup("atlas_eci_country_year.csv", ["eci_sitc"])
def complexity(i, c, t, ti):
    ok = all(np.isfinite(lg[i, ti - k]) for k in range(6)) and np.isfinite(lg[i, ti + 5]) if ti + 5 < len(Y) else False
    return {"eci": eci["eci_sitc"].get((c, t), np.nan), "g5": (lg[i, ti + 5] - lg[i, ti]) / 5 if ok else np.nan,
            "gap": front[ti] - lg[i, ti], "g_prev5": (lg[i, ti] - lg[i, ti - 5]) / 5 if np.isfinite(lg[i, ti - 5]) else np.nan,
            "popg": (b["lpop"][i, ti] - b["lpop"][i, ti - 5]) / 5, "dw5": (lw[i, ti + 5] - lw[i, ti]) / 5 if ti + 5 < len(Y) else np.nan}
d = rows_for(1965, 2019, complexity)
out["eci"] = fit(d, "g5", ["eci"], ["gap", "g_prev", "g_prev5", "conflict", "dem", "popg", "dw5"], "ols", split=2000)
print("5. eci", json.dumps(out["eci"]), flush=True)

# 6. coup-proofing
ssf = lookup("ssf_counterbalancing_country_year.csv", ["counterbalancing", "n_counterweights"])
def proof(i, c, t, ti):
    v = ssf["n_counterweights"].get((c, t))
    return {"counterweights": np.nan if v is None or not np.isfinite(v) else np.log1p(v)}
d = rows_for(1960, 2011, proof)
CT = ["lg", "dem", "conflict", "g_prev", "coup_recent"]
out["coup_proofing"] = {"coup_attempt": fit(d, "coup1", ["counterweights"], CT),
                        "coup_success_given_attempt": fit(d, "coup_success1", ["counterweights"], ["lg", "dem"]),
                        "onset2": fit(d, "onset2", ["counterweights"], ["lg", "dem", "g_prev"])}
print("6. coup-proofing", json.dumps(out["coup_proofing"]), flush=True)

# 7. external debt fragility x US rates
wd = lookup("wdi_external_debt_country_year.csv", ["st_debt_pct_reserves", "debt_service_pct_exports"])
us = pd.read_csv(ST / "us_rates_year.csv").set_index("year")
rr = us.real_fedfunds_expost
def debt(i, c, t, ti):
    st = wd["st_debt_pct_reserves"].get((c, t), np.nan)
    ds = wd["debt_service_pct_exports"].get((c, t), np.nan)
    drr = rr.get(t + 1, np.nan) - rr.get(t - 1, np.nan)
    st_ = np.log1p(min(st, 1000) / 100) if np.isfinite(st) else np.nan
    return {"st_res": st_, "ds_exp": np.log1p(ds / 100) if np.isfinite(ds) else np.nan,
            "st_x_rate": st_ * drr / 10 if np.isfinite(st_) and np.isfinite(drr) else np.nan}
d = rows_for(1975, 2023, debt)
CT = ["lg", "dem", "conflict", "g_prev"]
out["external_debt"] = {y: fit(d, y, ["st_res", "ds_exp", "st_x_rate"], CT) for y in ("fin_start", "collapse2")}
print("7. external debt", json.dumps(out["external_debt"]), flush=True)

# 8. coups among language / colonial peers
peers = pd.read_csv(RAW / "trade" / "peer_pairs_2015.csv")
ci = {c: i for i, c in enumerate(codes)}
Wp = np.zeros((len(codes), len(codes)))
for r in peers.itertuples():
    if r.a in ci and r.b in ci and r.a != r.b:
        Wp[ci[r.a], ci[r.b]] = 1
Wp = Wp / np.maximum(Wp.sum(1, keepdims=True), 1)
D = risks.distances(codes)
Wn = ((D < 1500) & (D > 0)).astype(float)
Wn = Wn / np.maximum(Wn.sum(1, keepdims=True), 1)
def peer(i, c, t, ti):
    cp = (coups[:, ti] > 0).astype(float)
    return {"peer_coups": float(Wp[i] @ cp), "near_coups": float(Wn[i] @ cp), "has_peers": float(Wp[i].sum() > 0)}
d = rows_for(1950, 2023, peer)
d = d[d.has_peers == 1]
CT = ["lg", "dem", "conflict", "g_prev", "coup_recent", "near_coups"]
out["peer_coups"] = fit(d, "coup1", ["peer_coups"], CT)
print("8. peer coups", json.dumps(out["peer_coups"]), flush=True)
json.dump(out, open(ROOT / "outputs" / "structure_test.json", "w"), indent=1)
