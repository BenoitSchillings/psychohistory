"""Innovation: what history says about catch-up, the frontier, and fast technology spread.

A. Catch-up (country level). Does innovation effort (R&D % GDP, researchers per million, resident patents and
   articles per person; WDI 1996-) predict 5-year growth of GDP per person beyond the growth controls (as in
   workforce_test.py: gap to the frontier, past growth, conflict, democracy, population growth, working-age change)?
   Year effects, clustered by country; out-of-sample fit to 2007, score 2008-2018 windows.
B. The frontier (US). Bloom, Jones, Van Reenen & Webb (2020): research productivity = TFP growth / effective
   researchers (R&D spending / nominal GDP per person) falls steadily. Fit the semi-endogenous law
   g_A = a * S^lam * A^-phi in decade averages (log g_A on log S and log A), and project frontier TFP growth to 2065
   under researcher growth (i) at its 2000-2024 pace, (ii) halving by 2045 (ageing), and (iii) AI adding
   research effort: S x (1 + m * adoption), m the effective-researcher multiplier at full adoption (assumption).
C. Fast spread and unrest. Does a fast 5-year rise in internet or mobile use predict protest waves, coup attempts,
   civil-conflict onset or autocratization (next 2 years), with year effects and controls?
python -P scripts/innovation_test.py  -> outputs/innovation_test.json
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

IN = RAW / "innovation"
p = get_panel()
b = base_arrays(p)
Y = list(p.years)
lg = b["lg"]
front = np.nanmax(np.where(np.isfinite(lg), lg, -np.inf), 0)
lw = working_age_share(p)
out = {}


def ols(d, y, cols, ctrl, split):
    sub = d.dropna(subset=[y] + cols + ctrl)
    fe = pd.get_dummies(sub.year, prefix="y", drop_first=True).astype(float)
    m = sm.OLS(sub[y], pd.concat([sub[cols + ctrl], fe], axis=1).assign(const=1.0)).fit(
        cov_type="cluster", cov_kwds={"groups": sub.c.astype("category").cat.codes})
    r = {k: [round(float(m.params[k]), 4), round(float(m.bse[k]), 4)] for k in cols}
    tr, te = sub.year <= split, sub.year > split + 4
    for tag, cc in (("without", ctrl), ("with", cols + ctrl)):
        Z = sm.add_constant(sub[cc])
        mo = sm.OLS(sub[y][tr], Z[tr]).fit()
        r[f"oos_rmse_{tag}"] = round(float(np.sqrt(np.mean((mo.predict(Z[te]) - sub[y][te]) ** 2))), 5)
    r["n"] = int(len(sub))
    return r


# ---------- A. catch-up
w = pd.read_csv(IN / "wdi_innovation_cy.csv").set_index(["iso3", "year"])
rows = []
for i, c in enumerate(p.codes):
    for t in range(1996, 2020):
        ti = Y.index(t)
        if not all(np.isfinite(lg[i, ti - k]) for k in range(6)) or not np.isfinite(lg[i, ti + 5]):
            continue
        g = lambda col: w[col].get((c, t), np.nan) if (c, t) in w.index else np.nan
        # effort averaged over the 3 years to t (sparse reporting)
        avg = lambda col: np.nanmean([w[col].get((c, t - k), np.nan) if (c, t - k) in w.index else np.nan for k in range(3)])
        pop = np.exp(b["lpop"][i, ti])
        rows.append({"c": c, "year": t, "g5": (lg[i, ti + 5] - lg[i, ti]) / 5,
                     "gap": front[ti] - lg[i, ti], "g_prev": np.clip(lg[i, ti] - lg[i, ti - 1], -.3, .3),
                     "g_prev5": (lg[i, ti] - lg[i, ti - 5]) / 5, "conflict": float(b["conflict"][i, ti] > 0),
                     "dem": b["dem"][i, ti], "popg": (b["lpop"][i, ti] - b["lpop"][i, ti - 5]) / 5,
                     "dw5": (lw[i, ti + 5] - lw[i, ti]) / 5,
                     "rd": np.log1p(avg("rd_exp_pct_gdp")), "res": np.log1p(avg("researchers_per_million")),
                     "pat": np.log1p(avg("patent_apps_resident") / pop * 1e6), "art": np.log1p(avg("sci_tech_articles") / pop * 1e6)})
d = pd.DataFrame(rows)
CTRL = ["gap", "g_prev", "g_prev5", "conflict", "dem", "popg", "dw5"]
A = {}
for name, cols in (("rd", ["rd"]), ("researchers", ["res"]), ("patents", ["pat"]), ("articles", ["art"]),
                   ("rd x gap", ["rd", "rd_gap"])):
    d["rd_gap"] = d.rd * d.gap
    A[name] = ols(d, "g5", cols, CTRL, 2007)
print("A. catch-up:", json.dumps(A, indent=1), flush=True)
out["catch_up"] = A

# ---------- B. frontier
u = pd.read_csv(IN / "us_frontier_annual.csv")
u = u[["year", "bls_tfp_private_nonfarm_2017eq100", "eff_researchers_rd_mn"]].dropna()
u["lA"] = np.log(u.bls_tfp_private_nonfarm_2017eq100)
u["lS"] = np.log(u.eff_researchers_rd_mn)
u["dec"] = (u.year // 10) * 10
dec = u.groupby("dec").agg(lA0=("lA", "first"), lA1=("lA", "last"), y0=("year", "first"), y1=("year", "last"), lS=("lS", "mean"), lAm=("lA", "mean"))
dec = dec[dec.y1 - dec.y0 >= 4]
dec["gA"] = (dec.lA1 - dec.lA0) / (dec.y1 - dec.y0)
dec = dec[dec.gA > 0]
rp = u.assign(gA=u.lA.diff().rolling(10).mean())
rp["log_rp"] = np.log(rp.gA.clip(lower=1e-4) / np.exp(rp.lS))
tr = sm.OLS(rp.log_rp.dropna(), sm.add_constant(rp.loc[rp.log_rp.notna(), "year"])).fit()
fit = sm.OLS(np.log(dec.gA), sm.add_constant(dec[["lS", "lAm"]])).fit()
lam, phi = float(fit.params["lS"]), -float(fit.params["lAm"])
B = {"decades": {int(k): {"gA_pct": round(float(v.gA * 100), 2), "S_mn": round(float(np.exp(v.lS)), 2)} for k, v in dec.iterrows()},
     "research_productivity_trend_pct_per_year": round(float(tr.params["year"]) * 100, 2),
     "lambda": round(lam, 3), "phi": round(phi, 3), "se": {k: round(float(v), 3) for k, v in fit.bse.items()},
     "steady_state_gA_per_gS": round(lam / phi, 3) if phi > 0 else None}
# projections 2025-2065: integrate dlnA = a S^lam A^-phi with a matched to 2010-2024
last = u[u.year >= 2010]
gS_recent = float(np.polyfit(last.year, last.lS, 1)[0])
gA_recent = float(np.polyfit(last.year, last.lA, 1)[0])
A0, S0 = float(u.lA.iloc[-1]), float(u.lS.iloc[-1])
lam_u, phi_u = max(lam, 0.05), max(phi, 0.0)
a = gA_recent / np.exp(lam_u * S0 - phi_u * A0)


def project(gS_path, mult=lambda t: 1.0):
    lA, lS, path = A0, S0, {}
    for t in range(2025, 2066):
        lS += gS_path(t)
        g = a * np.exp(lam_u * (lS + np.log(mult(t))) - phi_u * lA)
        lA += g
        if t in (2030, 2040, 2050, 2065):
            path[t] = round(g * 100, 2)
    return path


ai_adopt = lambda t: 1 / (1 + np.exp(-(t - 2035) / 3))          # illustrative adoption curve (model's AI layer samples it)
B["gS_recent_pct"] = round(gS_recent * 100, 2)
B["gA_recent_pct"] = round(gA_recent * 100, 2)
B["projection_tfp_growth_pct"] = {
    "researchers_at_recent_pace": project(lambda t: gS_recent),
    "researcher_growth_halving_by_2045": project(lambda t: gS_recent * (1 - 0.5 * min(1, (t - 2025) / 20))),
    "ai_x3_researchers": project(lambda t: gS_recent, lambda t: 1 + 2 * ai_adopt(t)),
    "ai_x10_researchers": project(lambda t: gS_recent, lambda t: 1 + 9 * ai_adopt(t)),
    "ai_x100_researchers": project(lambda t: gS_recent, lambda t: 1 + 99 * ai_adopt(t))}
print("B. frontier:", json.dumps(B, indent=1), flush=True)
out["frontier"] = B

# ---------- C. fast spread and unrest
def owid(slug):
    x = pd.read_csv(RAW / f"{slug}.csv")
    return x.rename(columns={x.columns[3]: "v"}).dropna(subset=["code"]).set_index(["code", "year"]).v
inet, mob = owid("share-of-individuals-using-the-internet"), owid("mobile-cellular-subscriptions-per-100-people")
rows = []
for i, c in enumerate(p.codes):
    for t in range(1995, 2022):
        ti = Y.index(t)
        if not np.isfinite(b["dem"][i, ti]) or not np.isfinite(lg[i, ti]) or not np.isfinite(b["dem"][i, ti + 2]):
            continue
        conf = b["conflict"][i, ti]
        mm = b["mass_mobilization"]
        rows.append({"c": c, "year": t,
                     "d_inet5": (inet.get((c, t), np.nan) - inet.get((c, t - 5), np.nan)) / 100,
                     "d_mob5": (mob.get((c, t), np.nan) - mob.get((c, t - 5), np.nan)) / 100,
                     "lg": lg[i, ti], "dem": b["dem"][i, ti], "conflict": float(conf > 0),
                     "g_prev": np.clip(lg[i, ti] - lg[i, ti - 1], -.3, .3) if np.isfinite(lg[i, ti - 1]) else np.nan,
                     "mob_up": float(np.nanmax(mm[i, ti + 1:ti + 3]) - mm[i, ti] > 1.0) if np.isfinite(mm[i, ti]) else np.nan,
                     "coup2": float(np.nansum(b["coup_attempts"][i, ti + 1:ti + 3]) > 0),
                     "onset2": float(np.nansum(b["conflict"][i, ti + 1:ti + 3]) > 0) if conf == 0 else np.nan,
                     "autoc2": float(b["dem"][i, ti + 2] - b["dem"][i, ti] < -0.1)})
d = pd.DataFrame(rows)
C_ = {}
for x in ("d_inet5", "d_mob5"):
    C_[x] = {}
    for yv in ("mob_up", "coup2", "onset2", "autoc2"):
        sub = d.dropna(subset=[x, yv, "lg", "dem", "g_prev"])
        sub = sub[sub.groupby("year")[yv].transform("max") > 0]
        fe = pd.get_dummies(sub.year, prefix="y", drop_first=True).astype(float)
        X = pd.concat([sub[[x, "lg", "dem", "conflict", "g_prev"]], fe], axis=1).assign(const=1.0)
        X = X.loc[:, X.std() > 0].assign(const=1.0)
        try:
            m = sm.Logit(sub[yv], X).fit(disp=0, maxiter=300, cov_type="cluster",
                                         cov_kwds={"groups": sub.c.astype("category").cat.codes})
            C_[x][yv] = [round(float(m.params[x]), 3), round(float(m.bse[x]), 3), int(sub[yv].sum()), int(len(sub))]
        except Exception as e:
            C_[x][yv] = repr(e)[:80]
print("C. spread and unrest [coef, se, events, n]:", json.dumps(C_, indent=1))
out["spread_unrest"] = C_
json.dump(out, open(ROOT / "outputs" / "innovation_test.json", "w"), indent=1)
