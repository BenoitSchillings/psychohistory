"""Generic test of a new country-year (or static per-country) data family for the risk models.

python -P scripts/new_data_test.py NAME   (NAME in SOURCES below) -> outputs/new_data/<NAME>.json
1. logit for each task with a small control set (log GDP, log population, conflict history, democracy), errors
   clustered by country: does each new column carry a coefficient?
2. tuned GBM rolling-origin backtest with and without the columns: development origins 1975-2005, then the
   2010/2015 lockbox. Kept only if the development score improves beyond noise (~0.005) and the lockbox does not get
   worse."""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

from psychohistory.pipeline import prepare
from psychohistory.features import FEATURES, TASKS
from psychohistory.data import RAW, ROOT
from psychohistory.backtest import flatten
from psychohistory import models as M
from psychohistory.evaluate import evaluate, DEV_ORIGINS, LOCK_ORIGINS
sys.path.insert(0, os.path.dirname(__file__))
from exp import Wrap, TUNED_GBM

ND = RAW / "newdata"


def _static(p, path, cols, key="iso3"):
    d = pd.read_csv(path).set_index(key)
    return {c: np.repeat(d[c].reindex(list(p.codes)).to_numpy(float)[:, None], len(p.years), 1) for c in cols}


def _yearly(p, path, cols, key="iso3"):
    d = pd.read_csv(path)
    return {c: d.pivot_table(index=key, columns="year", values=c).reindex(index=list(p.codes), columns=list(p.years)).to_numpy(float)
            for c in cols}


def structural(p):
    s = _static(p, ND / "structural_static.csv", ["lmtnest", "ef", "numlang", "plural", "ncontig", "elevdiff", "first_state_year"])
    age = np.asarray(p.years)[None, :] - s["first_state_year"]
    age = np.where(age >= 0, age, np.nan)
    return {"lmtnest": s["lmtnest"], "ethnic_frac": s["ef"], "log_numlang": np.log1p(s["numlang"]), "plural_share": s["plural"],
            "noncontiguous": s["ncontig"], "log_elevdiff": np.log1p(s["elevdiff"]), "log_state_age": np.log1p(age),
            "new_state": (age <= 2).astype(float) * np.isfinite(age)}


def heat(p):
    sys.path.insert(0, os.path.dirname(__file__))
    from heat_test import heat_arrays
    a = heat_arrays(p)
    return {n: a[..., i] for i, n in enumerate(["tw_level", "tw_shock", "tw_hot24", "tw_ta_gap"])}


def imf(p):
    """IMF Programs Dataset 1952-2025 (Stubbs, Harvard Dataverse doi:10.7910/DVN/V4V1BA, CC0)"""
    y = _yearly(p, ND / "imf" / "imf_programs_cy.csv", ["IMFm", "IMFnn", "nonIMFm"], key="country_code")
    on = y["IMFnn"]
    prev = np.concatenate([np.full((on.shape[0], 1), np.nan), on[:, :-1]], 1)
    share10 = np.full(on.shape, np.nan)
    for t in range(10, on.shape[1]):
        share10[:, t] = np.nanmean(on[:, t - 9:t + 1], 1)
    return {"imf_months": y["IMFm"] / 12, "imf_share_10y": share10, "imf_new": ((on == 1) & (prev == 0)).astype(float) * np.isfinite(prev),
            "imf_nonconcessional": y["nonIMFm"] / 12}


def polity(p):
    """Polity5 (Center for Systemic Peace, p5v2018, data to 2020): factionalism (parcomp = 3), the PITF 'partial
    democracy with factionalism' (factional and polity2 1-7), anocracy (polity2 -5..5), regime durability; carried
    forward up to 5 years (to 2025)"""
    from psychohistory import interstate as I
    d = pd.read_csv(ND / "polity" / "p5v2018.csv")
    c2i = I._ccode_to_iso(set(p.codes))
    d["iso3"] = d.ccode.map(c2i)
    d = d.dropna(subset=["iso3"]).drop_duplicates(["iso3", "year"], keep="last")
    d["factional"] = (d.parcomp == 3).astype(float).where(d.parcomp >= 0)
    d["fact_partdem"] = ((d.parcomp == 3) & d.polity2.between(1, 7)).astype(float).where(d.parcomp >= 0)
    d["anocracy"] = d.polity2.between(-5, 5).astype(float).where(d.polity2.notna())
    d["log_durable"] = np.log1p(d.durable)
    tmp = ND / "polity" / "_features.csv"
    d[["iso3", "year", "factional", "fact_partdem", "anocracy", "log_durable"]].to_csv(tmp, index=False)
    y = _yearly(p, tmp, ["factional", "fact_partdem", "anocracy", "log_durable"])
    out = {}
    for k, a in y.items():
        a = a.copy()
        for t in range(1, a.shape[1]):            # carry forward up to 5 years
            fill = np.isnan(a[:, t])
            src = a[:, t - 1]
            a[fill, t] = src[fill]
        lastobs = np.where(np.isfinite(y[k]), np.asarray(p.years)[None, :], np.nan)
        lastobs = pd.DataFrame(lastobs).ffill(axis=1).to_numpy()
        a[(np.asarray(p.years)[None, :] - lastobs) > 5] = np.nan
        out[k] = a
    return out


def macro(p):
    """WDI macro-vulnerability indicators (data/raw/newdata/macro/fetch_wdi_macro.py), aimed at economic collapse"""
    y = _yearly(p, ND / "macro" / "wdi_macro_cy.csv", ["FI.RES.TOTL.MO", "BN.CAB.XOKA.GD.ZS", "FS.AST.PRVT.GD.ZS",
                                                        "PA.NUS.FCRF", "DT.DOD.DSTC.IR.ZS", "TT.PRI.MRCH.XD.WD"])
    lag = lambda a, k: np.concatenate([np.full((a.shape[0], k), np.nan), a[:, :-k]], 1)
    fx = np.log(np.where(y["PA.NUS.FCRF"] > 0, y["PA.NUS.FCRF"], np.nan))
    cr, tt = y["FS.AST.PRVT.GD.ZS"], np.log(np.where(y["TT.PRI.MRCH.XD.WD"] > 0, y["TT.PRI.MRCH.XD.WD"], np.nan))
    return {"log_reserves_months": np.log1p(np.clip(y["FI.RES.TOTL.MO"], 0, None)), "current_account": np.clip(y["BN.CAB.XOKA.GD.ZS"], -50, 50),
            "credit_boom_5y": np.clip(cr - lag(cr, 5), -100, 100), "depreciation_1y": np.clip(fx - lag(fx, 1), -3, 3),
            "depreciation_3y": np.clip(fx - lag(fx, 3), -3, 3), "log_stdebt_reserves": np.log1p(np.clip(y["DT.DOD.DSTC.IR.ZS"], 0, None)),
            "tot_change_3y": np.clip(tt - lag(tt, 3), -2, 2)}


def warstructure(p):
    """structure of ongoing civil wars, aimed at war endings: UCDP rebel groups, intensity, territorial aims and
    internationalization (1946-2024), UCDP external support (1975-2017, carried 8 years), UN peacekeepers (1990+).
    Country-years absent from a file within its coverage are 0 (no conflict / no mission)."""
    S = RAW / "structure"
    def fill0(a, lo, hi, carry=0):
        yrs = np.asarray(p.years)
        a = np.where(np.isnan(a) & (yrs >= lo)[None] & (yrs <= hi)[None], 0.0, a)
        for k in range(carry):
            t = np.flatnonzero(yrs == hi + 1 + k)
            if len(t):
                a[:, t[0]] = a[:, t[0] - 1]
        return a
    fr = _yearly(p, S / "ucdp_rebel_fragmentation_country_year.csv", ["n_rebel_groups", "max_intensity", "n_territorial_conflicts", "internationalized"])
    es = _yearly(p, S / "ucdp_external_support_country_year.csv", ["rebel_ext_support_state", "govt_foreign_troops"])
    pk = _yearly(p, RAW / "intervention" / "un_pko_personnel_cy.csv", ["total"])
    return {"log_rebel_groups": np.log1p(fill0(fr["n_rebel_groups"], 1946, 2024, 1)),
            "war_intensity": fill0(fr["max_intensity"], 1946, 2024, 1),
            "territorial_conflict": (fill0(fr["n_territorial_conflicts"], 1946, 2024, 1) > 0).astype(float),
            "internationalized": fill0(fr["internationalized"], 1946, 2024, 1),
            "rebels_state_support": fill0(es["rebel_ext_support_state"], 1975, 2017, 8),
            "govt_foreign_troops": fill0(es["govt_foreign_troops"], 1975, 2017, 8),
            "log_peacekeepers": np.log1p(fill0(pk["total"], 1990, 2026))}


def milex(p):
    """SIPRI military expenditure (data/raw/intervention/sipri_milex_cy.csv): budget cuts as coup triggers (Powell 2012)"""
    d = pd.read_csv(RAW / "intervention" / "sipri_milex_cy.csv")
    col = [c for c in d.columns if "const" in c.lower() or "usd" in c.lower()]
    col = col[0] if col else d.columns[2]
    y = _yearly(p, RAW / "intervention" / "sipri_milex_cy.csv", [col])[col]
    l = np.log(np.where(y > 0, y, np.nan))
    lag = lambda a, k: np.concatenate([np.full((a.shape[0], k), np.nan), a[:, :-k]], 1)
    g1 = np.clip(l - lag(l, 1), -2, 2)
    return {"milex_change_1y": g1, "milex_cut_10pct": (g1 < -0.105).astype(float) * np.isfinite(g1),
            "milex_change_3y": np.clip(l - lag(l, 3), -3, 3)}


def rivalry(p):
    """great-power rivalry spilling into other countries' wars (proxy wars). UN General Assembly ideal points (Bailey,
    Strezhnev & Voeten; data/raw/interstate/unga_dy.csv, 1946-2025):
      gp_rivalry           distance between the US and Soviet/Russian ideal points (one value per year)
      gp_rivalry_change_5y its 5-year change
      contested            1 at the midpoint between the US and the opposing pole, 0 at either pole or beyond (swing states)
      rivalry_x_contested  the product: where rivalry should matter most"""
    u = pd.read_csv(RAW / "interstate" / "unga_dy.csv")
    ip = pd.concat([u[["iso3_a", "year", "ideal_a"]].set_axis(["c", "year", "ip"], axis=1),
                    u[["iso3_b", "year", "ideal_b"]].set_axis(["c", "year", "ip"], axis=1)]).groupby(["c", "year"]).ip.mean()
    ip = ip.unstack().reindex(index=list(p.codes), columns=list(p.years))
    ip = ip.T.ffill(limit=2).T.to_numpy(float)
    codes = list(p.codes)
    us, ru, cn = ip[codes.index("USA")], ip[codes.index("RUS")], ip[codes.index("CHN")]
    riv = np.abs(us - ru)
    # the opposing pole: Russia or China, whichever votes further from the US (China after the 1990s: Russia's votes
    # moved toward the US's, leaving most of the global South beyond it)
    pole = np.where(np.nan_to_num(np.abs(cn - us)) > np.abs(ru - us), cn, ru)
    pos = (ip - pole[None]) / np.where(np.abs(us - pole) > 0.1, us - pole, np.nan)[None]   # 0 at the pole, 1 at the US
    contested = np.clip(1 - np.abs(2 * pos - 1), 0, 1)
    lag5 = np.concatenate([np.full(5, np.nan), riv[:-5]])
    R = np.repeat(riv[None], len(codes), 0)
    return {"gp_rivalry": R, "gp_rivalry_change_5y": np.repeat((riv - lag5)[None], len(codes), 0),
            "contested": contested, "rivalry_x_contested": R * contested}


def personal(p):
    """personalist rule beyond what the model has (it has neopatrimonialism, clientelism, regime types, legitimation by
    ideology and performance): V-Dem v15 leader-cult legitimation (v2exl_legitlead) and its 5-year change,
    presidentialism (v2xnp_pres), regime corruption (v2xnp_regcorr), the regime's support coalition (military,
    foreign governments, an ethnic group: v2regsupgroups 5, 13, 6) and a head of state who took power by force
    (v2expathhs = 0). data/raw/newdata/personal/vdem_personal_cy.csv"""
    return _yearly(p, ND / "personal" / "vdem_personal_cy.csv",
                   ["legit_personalist", "personalization_5y", "presidentialism", "regime_corruption", "support_military",
                    "support_foreign", "support_ethnic", "path_force"])


def personal_support(p):
    """ablation of personal: the regime's support coalition and power taken by force only"""
    a = personal(p)
    return {k: a[k] for k in ("support_military", "support_foreign", "support_ethnic", "path_force")}


def personal_core(p):
    """ablation of personal: leader-cult legitimation, its change, presidentialism and regime corruption only"""
    a = personal(p)
    return {k: a[k] for k in ("legit_personalist", "personalization_5y", "presidentialism", "regime_corruption")}


def _placebo(seed, family="personal"):
    def f(p):
        """placebo for personal: the same columns with whole country rows shuffled (realistic values and time
        patterns, wrong country), to measure the gain that eight extra columns give by chance"""
        a = SOURCES[family](p)
        perm = np.random.default_rng(seed).permutation(len(p.codes))
        # (renamed: once a family is adopted its real names are in FEATURES and would be picked instead of the copies)
        return {f"placebo{seed}_{k}": v[perm] for k, v in a.items()}
    return f


def memory(p):
    """generational memory (scripts/build_memory.py): share of today's population old enough (>= 8 at the time) to
    remember the country's most recent interstate war, civil conflict, hyperinflation and economic collapse"""
    return _yearly(p, ND / "memory" / "memory_cy.csv",
                   ["mem_interstate_war", "mem_civil_conflict", "mem_hyperinflation", "mem_collapse"])


def _base(p):
    from psychohistory.features import base_arrays
    return base_arrays(p)


def _lag(a, k):
    return np.concatenate([np.full((a.shape[0], k), np.nan), a[:, :-k]], 1)


def _ffill_t(a, limit):
    return pd.DataFrame(a).T.ffill(limit=limit).T.to_numpy(float)


def _xs_resid(y, xs, min_n=30):
    """per-year cross-country OLS residual of y on xs (what a country has beyond what its covariates predict)"""
    out = np.full(y.shape, np.nan)
    for t in range(y.shape[1]):
        Z = np.column_stack([np.ones(y.shape[0])] + [x[:, t] for x in xs])
        ok = np.isfinite(Z).all(1) & np.isfinite(y[:, t])
        if ok.sum() >= min_n:
            beta = np.linalg.lstsq(Z[ok], y[ok, t], rcond=None)[0]
            out[ok, t] = y[ok, t] - Z[ok] @ beta
    return out


def deprivation(p):
    """expectations against reality (relative deprivation, Gurr): growth against the country's own previous decade and
    against its neighbours, schooling ahead of income, educated and young unemployment"""
    import torch
    from psychohistory.features import _neighbours
    b = _base(p)
    lg = b["lg"]
    g5 = (lg - _lag(lg, 5)) / 5
    prev = (_lag(lg, 5) - _lag(lg, 15)) / 10
    nb = _neighbours(torch.as_tensor(g5.T, dtype=torch.float32)).numpy().T
    sch, ter, un = _ffill_t(b["schooling"], 10), _ffill_t(b["tertiary_share"], 10), _ffill_t(b["unemployment"], 3)
    return {"growth_vs_own_trend": np.clip(g5 - prev, -0.3, 0.3), "growth_vs_neighbours": np.clip(g5 - nb, -0.3, 0.3),
            "schooling_ahead_of_income": _xs_resid(sch, [lg]), "educated_unemployment": ter * un / 100,
            "youth_unemployment": b["youth"] * un}


def brittleness(p):
    """hidden preferences and cascades (Kuran, Granovetter): a democracy deficit against what income, schooling and
    cities predict (modernisation), held down by repression, with the crowd capacity to cascade"""
    b = _base(p)
    lg, dem = b["lg"], b["dem"]
    sch, urb = _ffill_t(b["schooling"], 10), b["urban"]
    deficit = -_xs_resid(dem, [lg, sch, urb])                       # > 0: less democratic than predicted
    repress = np.clip(-b["cso_repression"], 0, None)                # V-Dem: low = repressive
    mobpot = urb / 100 * b["youth"] * sch / 12
    return {"dem_deficit": deficit, "deficit_x_repression": np.clip(deficit, 0, None) * repress,
            "mobilisation_potential": mobpot, "repression_x_mobilisation": repress * mobpot}


def elite(p):
    """elite overproduction and the wealth pump (Turchin): graduates beyond what income predicts and its 10-year
    change, graduate growth against income growth, the top 1% share's 10-year change"""
    b = _base(p)
    lg = b["lg"]
    ter = _ffill_t(b["tertiary_share"], 10)
    excess = _xs_resid(ter, [lg])
    top1 = _ffill_t(b["top1"], 10)
    return {"graduates_excess": excess, "graduates_excess_change_10y": excess - _lag(excess, 10),
            "graduates_vs_growth_10y": (ter - _lag(ter, 10)) / 10 - (lg - _lag(lg, 10)) * 10,
            "top1_change_10y": top1 - _lag(top1, 10)}


def coalition(p):
    """the rest of the regime's support coalition and its opposition (V-Dem v15 v2regsupgroups / v2regoppgroups:
    aristocracy, agrarian elites, party elites, business elites, civil servants, religious groups, local elites, urban
    and rural workers and middle classes; opposition also military and ethnic groups),
    data/raw/newdata/personal/vdem_coalition_cy.csv"""
    d = pd.read_csv(ND / "personal" / "vdem_coalition_cy.csv")
    cols = [c for c in d.columns if c not in ("iso3", "year")]
    return _yearly(p, ND / "personal" / "vdem_coalition_cy.csv", cols)


def crisis_memory(p):
    """long crisis memory since 1800 (scripts/build_crisis_memory.py: Reinhart-Rogoff, BoC-BoE, GMD): years in external
    default, inflation-crisis and banking-crisis years since 1800, and years since the last default and inflation crisis"""
    return _yearly(p, ND / "crisis_memory_cy.csv", ["default_years_since1800", "log_years_since_default",
                   "inflation_crises_since1800", "log_years_since_inflation_crisis", "banking_crises_since1800"])


def coalition_core(p):
    """the coalition variables selected on development years only (logit p < 0.001 for some task with controls, label
    years to 2005): military and business elites in opposition; religious groups, civil servants, rural workers and
    agrarian elites among the regime's supporters"""
    a = coalition(p)
    return {k: a[k] for k in ("opp_military", "opp_business_elites", "sup_religious", "sup_civil_servants",
                              "sup_rural_workers", "sup_agrarian_elites")}


def sovereignty(p):
    """confirmed by the V-Dem indicator screen of the risk models' errors (scripts/residual_screen.py indicators, dev
    1975-2005 then lockbox): the state's international and domestic autonomy (v2svinlaut, v2svdomaut), equality of
    civil liberties across social groups (v2clsocgrp) and the head of state dying in office (v2exdeathos)"""
    return _yearly(p, ND / "personal" / "vdem_sovereignty_cy.csv", ["intl_autonomy", "domestic_autonomy", "social_group_equality", "hos_died"])


def combo(p):
    """structural + IMF + war structure together (each positive on development and lockbox alone, below noise)"""
    return {**structural(p), **imf(p), **warstructure(p)}


SOURCES = {"combo": combo, "milex": milex, "structural": structural, "heat": heat, "imf": imf, "polity": polity, "macro": macro, "warstructure": warstructure, "rivalry": rivalry, "personal": personal, "personal_support": personal_support, "personal_core": personal_core, "placebo1": _placebo(1), "placebo2": _placebo(2), "memory": memory, "deprivation": deprivation, "brittleness": brittleness, "elite": elite, "coalition": coalition, "coalition_placebo1": _placebo(1, "coalition"), "coalition_placebo2": _placebo(2, "coalition"), "crisis_memory": crisis_memory, "coalition_core": coalition_core, "coalition_core_placebo": _placebo(3, "coalition_core"), "sovereignty": sovereignty}


def logit_test(p, Xf, Yf, ci, year, names, new):
    import statsmodels.api as sm
    ctrl = ["log_gdp", "log_pop", "conflict_past_10y", "log_years_since_conflict", "democracy"]
    out = {}
    for task in ("onset_1y", "civil_conflict_5y", "coup_1y", "econ_collapse_5y", "autocratization_5y"):
        k = TASKS.index(task)
        for col in new:
            cols = ctrl + [col]
            Z = Xf[:, [names.index(c) for c in cols]].astype(float)
            y = Yf[:, k]
            ok = np.isfinite(Z).all(1) & np.isfinite(y) & (year >= 1950)
            if Z[ok, -1].std() == 0:
                continue
            Zs = (Z[ok] - Z[ok].mean(0)) / (Z[ok].std(0) + 1e-9)
            try:
                m = sm.Logit(y[ok], sm.add_constant(Zs)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": ci[ok]})
            except Exception as e:
                out[f"{task}:{col}"] = f"failed: {e}"
                continue
            out[f"{task}:{col}"] = {"coef_sd": round(float(m.params[-1]), 3), "se": round(float(m.bse[-1]), 3),
                                    "p": round(float(m.pvalues[-1]), 3), "n": int(ok.sum()), "events": int(y[ok].sum())}
    return out


def main():
    name = sys.argv[1]
    p, base, X, Yl = prepare()
    arrs = SOURCES[name](p)
    new = list(arrs)
    Xa = np.concatenate([X, np.stack([arrs[c] for c in new], -1).astype(np.float32)], -1)
    Xf, Yf, ci, year = flatten(p, Xa, Yl)
    res = {"coverage": {c: round(float(np.isfinite(arrs[c]).mean()), 3) for c in new},
           "logit": logit_test(p, Xf, Yf, ci, year, FEATURES + new, new)}
    sig = {k: v for k, v in res["logit"].items() if isinstance(v, dict) and v["p"] < 0.05}
    print(json.dumps(sig, indent=1), flush=True)
    fac = lambda: Wrap(M.GBMModel(TUNED_GBM))
    bt = {}
    for nm, XX in (("base", X), (name, Xa)):
        for split, origins in (("dev", DEV_ORIGINS), ("lock", LOCK_ORIGINS)):
            s, out, _ = evaluate(p, XX, Yl, fac, origins)
            bt[f"{nm}:{split}"] = {t: {m: round(v, 4) for m, v in out[t].items() if m in ("auc", "bss")} for t in out} | {"score": round(s, 4)}
            print(nm, split, bt[f"{nm}:{split}"]["score"], flush=True)
    res["gbm_backtest"] = bt
    os.makedirs(ROOT / "outputs" / "new_data", exist_ok=True)
    json.dump(res, open(ROOT / "outputs" / "new_data" / f"{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
