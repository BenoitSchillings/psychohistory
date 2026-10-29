"""Peacekeeping, aid shocks and commodity-price shocks, tested on history before they enter the simulator.

A. UN peacekeeping (UN Peace and Security Data Hub, uniformed personnel by host, 1990-2025). Among country-years in
   civil conflict: does the log number of peacekeepers predict the war ending next year? Among country-years at peace
   within 10 years of a war: does it predict a relapse within 2 years? Logit with year effects and controls (income,
   democracy, conflict years, battle deaths), clustered by country. Missions go to the hardest cases, so a
   prediction test is biased against finding a benefit; the published causal estimates (Hegre, Hultman & Nygard 2019)
   are reported beside ours.
B. Aid shocks (WDI/OECD DAC net ODA % of GNI, 1975-2023). Does a fall in aid of more than 2% of GNI from its previous
   3-year mean (Nielsen et al. 2011 use a similar threshold) predict civil-conflict onset, a coup attempt or an
   economic collapse within 2 years?
C. Commodity-price shocks (WDI export shares x World Bank Pink Sheet real indices, 1965-2023). Exposure-weighted
   price change: shares of fuel, ores & metals, food and raw materials in merchandise exports (5-year lag) x the
   yearly log change of the matching index x exports % GDP. Predicts growth, collapse, onset, coups?
python -P scripts/intervention_test.py  -> outputs/intervention_test.json
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

IV = RAW / "intervention"
p = get_panel()
b = base_arrays(p)
Y = list(p.years)
lg, conf = b["lg"], np.nan_to_num(b["conflict"])
out = {}


def logit(d, y, cols, ctrl):
    sub = d.dropna(subset=[y] + cols + ctrl)
    sub = sub[sub.groupby("year")[y].transform("max") > 0]
    fe = pd.get_dummies(sub.year, prefix="y", drop_first=True).astype(float)
    X = pd.concat([sub[cols + ctrl], fe], axis=1).assign(const=1.0)
    X = X.loc[:, X.std() > 0].assign(const=1.0)
    try:
        m = sm.Logit(sub[y], X).fit(disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": sub.c.astype("category").cat.codes})
        return {k: [round(float(m.params[k]), 3), round(float(m.bse[k]), 3)] for k in cols} | {"events": int(sub[y].sum()), "n": int(len(sub))}
    except Exception as e:
        return repr(e)[:100]


def ols(d, y, cols, ctrl):
    sub = d.dropna(subset=[y] + cols + ctrl)
    fe = pd.get_dummies(sub.year, prefix="y", drop_first=True).astype(float)
    m = sm.OLS(sub[y], pd.concat([sub[cols + ctrl], fe], axis=1).assign(const=1.0)).fit(
        cov_type="cluster", cov_kwds={"groups": sub.c.astype("category").cat.codes})
    return {k: [round(float(m.params[k]), 4), round(float(m.bse[k]), 4)] for k in cols} | {"n": int(len(sub))}


def base_row(i, c, t):
    ti = Y.index(t)
    yrs_conf = int(np.sum(conf[i, max(0, ti - 9):ti + 1] > 0))
    return {"c": c, "year": t, "lg": lg[i, ti], "dem": b["dem"][i, ti], "conflict": float(conf[i, ti] > 0),
            "conf_years10": yrs_conf, "bd": np.log1p(np.nan_to_num(b["battle_deaths_pc"][i, ti]) * 1e5),
            "g_prev": float(np.clip(lg[i, ti] - lg[i, ti - 1], -.3, .3)) if np.isfinite(lg[i, ti - 1]) else np.nan,
            "end1": float(conf[i, ti + 1] == 0) if conf[i, ti] > 0 else np.nan,
            "onset2": float(np.sum(conf[i, ti + 1:ti + 3]) > 0) if conf[i, ti] == 0 else np.nan,
            "coup2": float(np.nansum(b["coup_attempts"][i, ti + 1:ti + 3]) > 0),
            "collapse2": float(np.nanmin(lg[i, ti + 1:ti + 3]) - lg[i, ti] < -0.05) if np.isfinite(lg[i, ti + 2]) else np.nan,
            "g1": float(np.clip(lg[i, ti + 1] - lg[i, ti], -.3, .3)) if np.isfinite(lg[i, ti + 1]) else np.nan,
            "post_war": float(conf[i, ti] == 0 and yrs_conf > 0)}


# ---------- A. peacekeeping
pk = pd.read_csv(IV / "un_pko_personnel_cy.csv").set_index(["iso3", "year"])
rows = []
for i, c in enumerate(p.codes):
    for t in range(1990, 2024):
        if not np.isfinite(lg[i, Y.index(t)]):
            continue
        r = base_row(i, c, t)
        n = pk.pko_total.get((c, t), 0.0) if (c, t) in pk.index else 0.0
        r["lpko"] = np.log1p(n / 1000)                    # thousands of uniformed personnel
        r["pko_any"] = float(n > 100)
        rows.append(r)
d = pd.DataFrame(rows)
CT = ["lg", "dem", "conf_years10", "bd"]
out["peacekeeping"] = {
    "termination (in conflict)": logit(d[d.conflict == 1], "end1", ["lpko"], CT),
    "relapse (post-war peace)": logit(d[d.post_war == 1], "onset2", ["lpko"], ["lg", "dem", "conf_years10"]),
    "literature": "Hegre, Hultman & Nygard (2019, J. Politics): a large PKO budget and strong mandate cut major "
                  "conflict by about two-thirds in simulations; Walter, Howard & Fortna (2021) review: PKOs reduce "
                  "relapse and civilian killing"}
print("A. peacekeeping:", json.dumps(out["peacekeeping"], indent=1), flush=True)

# ---------- B. aid shocks
aid = pd.read_csv(IV / "aid_wdi_cy.csv").set_index(["iso3", "year"]).oda_pct_gni
rows = []
for i, c in enumerate(p.codes):
    for t in range(1975, 2022):
        if not np.isfinite(lg[i, Y.index(t)]):
            continue
        a_now = aid.get((c, t), np.nan)
        a_prev = np.nanmean([aid.get((c, t - k), np.nan) for k in (1, 2, 3)])
        if not np.isfinite(a_now) or not np.isfinite(a_prev):
            continue
        r = base_row(i, c, t)
        r["aid_level"] = np.log1p(max(a_prev, 0))
        r["aid_drop"] = float(a_now - a_prev < -2)
        r["aid_change"] = float(np.clip(a_now - a_prev, -20, 20)) / 10
        rows.append(r)
d = pd.DataFrame(rows)
CT = ["lg", "dem", "g_prev", "aid_level"]
out["aid"] = {y: {"drop>2%GNI": logit(d, y, ["aid_drop"], CT + (["conflict"] if y != "onset2" else [])),
                  "change": logit(d, y, ["aid_change"], CT + (["conflict"] if y != "onset2" else []))}
              for y in ("onset2", "coup2", "collapse2")}
out["aid"]["growth"] = ols(d, "g1", ["aid_change"], CT + ["conflict"])
out["aid"]["drops"] = int(d.aid_drop.sum())
print("B. aid:", json.dumps(out["aid"], indent=1), flush=True)

# ---------- C. commodity-price shocks
cm = pd.read_excel(RAW / "wb_cmo_annual.xlsx", sheet_name="Annual Indices (Real)", header=None, skiprows=9)
cm = cm[pd.to_numeric(cm[0], errors="coerce").notna()]
idx = pd.DataFrame({"year": cm[0].astype(int), "fuel": cm[2], "metal": cm[14], "food": cm[6], "agri": cm[10]}).set_index("year").astype(float)
dl = np.log(idx).diff()
ce = pd.read_csv(IV / "commodity_exports_cy.csv").set_index(["iso3", "year"])
SH = {"fuel": "fuel_pct_merch", "metal": "ores_metals_pct_merch", "food": "food_pct_merch", "agri": "agri_raw_pct_merch"}
rows = []
for i, c in enumerate(p.codes):
    for t in range(1966, 2023):
        if not np.isfinite(lg[i, Y.index(t)]) or t + 1 not in dl.index:
            continue
        def share(col):                                      # last reported share 2-6 years before (predetermined)
            for k in range(2, 7):
                v = ce[col].get((c, t - k), np.nan) if (c, t - k) in ce.index else np.nan
                if np.isfinite(v):
                    return v / 100
            return np.nan
        sh = {k: share(v) for k, v in SH.items()}
        ex = np.nan
        for k in range(2, 7):
            v = ce.exports_gnfs_pct_gdp.get((c, t - k), np.nan) if (c, t - k) in ce.index else np.nan
            if np.isfinite(v):
                ex = min(v / 100, 1.5)
                break
        if not all(np.isfinite(list(sh.values()))) or not np.isfinite(ex):
            continue
        r = base_row(i, c, t)
        # price change from t to t+1 (the outcome year), as the simulator would see a drawn price path
        r["tot"] = ex * sum(sh[k] * dl[k].get(t + 1, np.nan) for k in SH)
        r["tot_neg"] = min(r["tot"], 0)
        r["dep"] = sum(sh.values())
        rows.append(r)
d = pd.DataFrame(rows)
CT = ["lg", "dem", "g_prev", "conflict", "dep"]
# (outcomes over t+1 only for growth; 2-year windows for events)
out["commodity"] = {"growth": ols(d, "g1", ["tot"], CT), "growth_neg_only": ols(d, "g1", ["tot_neg"], CT),
                    **{y: logit(d, y, ["tot"], CT if y != "onset2" else [c_ for c_ in CT if c_ != "conflict"])
                       for y in ("collapse2", "onset2", "coup2")}}
print("C. commodity:", json.dumps(out["commodity"], indent=1))
json.dump(out, open(ROOT / "outputs" / "intervention_test.json", "w"), indent=1)
