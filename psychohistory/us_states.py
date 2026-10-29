"""US states: a state-level psychohistory module.

Turchin applied structural-demographic theory to the United States itself (Ages of Discord).
Here the unit is the state-year, 1929-2024. Five crises are forecast five years ahead:

  income_shock       real personal income per person falls >= 3% below its starting level
  unemployment_surge the unemployment rate rises >= 2.5 points above its starting level
  housing_bust       real house prices fall >= 15% below their starting level
  overdose_surge     the drug-overdose death rate rises >= 50% (from >= 2 per 100k)
  mass_shooting      a public mass shooting (Mother Jones database)

Data: BEA (income, population 1929+), BLS (unemployment 1976+, CPI), FHFA (house prices 1975+),
CDC (overdose 1999+, suicide 1999-2017), Mother Jones (mass shootings 1982+), MIT Election Lab
(presidential returns 1976-2024), Shor & McCarty (legislative polarisation 1993-2020),
NOAA nClimDiv (state temperature and precipitation 1895+), V-Dem (national polarisation).
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .data import RAW, ROOT, get_panel

US = RAW / "us"
FIRST, LAST = 1929, 2025
YEARS = np.arange(FIRST, LAST + 1)
ORIGIN = 2024

STATES = [  # postal, fips, name, census region, NOAA nClimDiv state code
    ("AL", 1, "Alabama", "South", 1), ("AK", 2, "Alaska", "West", 50), ("AZ", 4, "Arizona", "West", 2),
    ("AR", 5, "Arkansas", "South", 3), ("CA", 6, "California", "West", 4), ("CO", 8, "Colorado", "West", 5),
    ("CT", 9, "Connecticut", "Northeast", 6), ("DE", 10, "Delaware", "South", 7),
    ("DC", 11, "District of Columbia", "South", 18), ("FL", 12, "Florida", "South", 8),
    ("GA", 13, "Georgia", "South", 9), ("HI", 15, "Hawaii", "West", None), ("ID", 16, "Idaho", "West", 10),
    ("IL", 17, "Illinois", "Midwest", 11), ("IN", 18, "Indiana", "Midwest", 12), ("IA", 19, "Iowa", "Midwest", 13),
    ("KS", 20, "Kansas", "Midwest", 14), ("KY", 21, "Kentucky", "South", 15), ("LA", 22, "Louisiana", "South", 16),
    ("ME", 23, "Maine", "Northeast", 17), ("MD", 24, "Maryland", "South", 18),
    ("MA", 25, "Massachusetts", "Northeast", 19), ("MI", 26, "Michigan", "Midwest", 20),
    ("MN", 27, "Minnesota", "Midwest", 21), ("MS", 28, "Mississippi", "South", 22),
    ("MO", 29, "Missouri", "Midwest", 23), ("MT", 30, "Montana", "West", 24), ("NE", 31, "Nebraska", "Midwest", 25),
    ("NV", 32, "Nevada", "West", 26), ("NH", 33, "New Hampshire", "Northeast", 27),
    ("NJ", 34, "New Jersey", "Northeast", 28), ("NM", 35, "New Mexico", "West", 29),
    ("NY", 36, "New York", "Northeast", 30), ("NC", 37, "North Carolina", "South", 31),
    ("ND", 38, "North Dakota", "Midwest", 32), ("OH", 39, "Ohio", "Midwest", 33), ("OK", 40, "Oklahoma", "South", 34),
    ("OR", 41, "Oregon", "West", 35), ("PA", 42, "Pennsylvania", "Northeast", 36),
    ("RI", 44, "Rhode Island", "Northeast", 37), ("SC", 45, "South Carolina", "South", 38),
    ("SD", 46, "South Dakota", "Midwest", 39), ("TN", 47, "Tennessee", "South", 40), ("TX", 48, "Texas", "South", 41),
    ("UT", 49, "Utah", "West", 42), ("VT", 50, "Vermont", "Northeast", 43), ("VA", 51, "Virginia", "South", 44),
    ("WA", 53, "Washington", "West", 45), ("WV", 54, "West Virginia", "South", 46),
    ("WI", 55, "Wisconsin", "Midwest", 47), ("WY", 56, "Wyoming", "West", 48),
]
CODES = [s[0] for s in STATES]
NAMES = [s[2] for s in STATES]
REGIONS = [s[3] for s in STATES]
BY_NAME = {s[2].lower(): s[0] for s in STATES}
BY_FIPS = {s[1]: s[0] for s in STATES}

TASKS = ["income_shock", "unemployment_surge", "housing_bust", "overdose_surge", "mass_shooting"]
TASK_NAMES = {"income_shock": "Real income shock (≥3%)", "unemployment_surge": "Unemployment surge (+2.5 pts)",
              "housing_bust": "Housing bust (≥15%)", "overdose_surge": "Overdose surge (+50%)",
              "mass_shooting": "Public mass shooting"}
H = 5


def _blank():
    return np.full((len(CODES), len(YEARS)), np.nan)


def _put(arr, df, col):
    """df with columns code, year, <col> -> fill arr."""
    ci = {c: i for i, c in enumerate(CODES)}
    for c, y, v in zip(df.code, df.year, df[col]):
        if c in ci and FIRST <= y <= LAST and pd.notna(v):
            arr[ci[c], int(y) - FIRST] = v
    return arr


def _cpi():
    d = pd.read_csv(US / "cu.data.1.AllItems", sep=r"\s+", dtype=str)
    d = d[d.series_id.str.strip() == "CUUR0000SA0"]
    d = d[d.period.str.startswith("M") & (d.period != "M13")]
    d["year"] = d.year.astype(int)
    d["value"] = pd.to_numeric(d.value, errors="coerce")
    cpi = d.groupby("year").value.mean()
    return cpi / cpi.get(2024)


def load():
    x = {}
    # --- BEA: personal income and population, 1929+
    b = pd.read_csv(US / "SAINC1__ALL_AREAS_1929_2025.csv", encoding="latin-1", dtype=str)
    b = b[b.LineCode.str.strip().isin(["2", "3"])]
    b["fips"] = b.GeoFIPS.str.strip().str.strip('"').str[:2].astype(int)
    b = b[b.GeoFIPS.str.strip().str.strip('"').str[2:] == "000"]
    yrs = [c for c in b.columns if c.isdigit()]
    long = b.melt(id_vars=["fips", "LineCode"], value_vars=yrs, var_name="year", value_name="v")
    long["year"] = long.year.astype(int)
    long["v"] = pd.to_numeric(long.v, errors="coerce")
    long["code"] = long.fips.map(BY_FIPS)
    cpi = _cpi()
    pop = long[long.LineCode.str.strip() == "2"]
    inc = long[long.LineCode.str.strip() == "3"].copy()
    inc["real"] = inc.v / inc.year.map(cpi)
    x["pop"] = _put(_blank(), pop, "v")
    x["income"] = _put(_blank(), inc, "real")              # real 2024 dollars per person
    # --- BLS LAUS unemployment rate (seasonally adjusted, annual mean), 1976+
    u = pd.read_csv(US / "la.data.3.AllStatesS", sep=r"\s+", dtype=str)
    u = u[u.series_id.str.endswith("03") & u.period.str.startswith("M") & (u.period != "M13")]
    u["fips"] = u.series_id.str[5:7].astype(int)
    u["code"] = u.fips.map(BY_FIPS)
    u["year"] = u.year.astype(int)
    u["value"] = pd.to_numeric(u.value, errors="coerce")
    ua = u.groupby(["code", "year"]).agg(value=("value", "mean"), n=("value", "size")).reset_index()
    ua = ua[ua.n >= 12]   # complete years only
    x["unemp"] = _put(_blank(), ua, "value")
    # --- FHFA all-transactions house price index, real, 1975+
    f = pd.read_csv(US / "fhfa_hpi_state.csv", header=None, names=["code", "year", "q", "hpi"])
    f["hpi"] = pd.to_numeric(f.hpi, errors="coerce")
    fa = f.groupby(["code", "year"]).agg(hpi=("hpi", "mean"), n=("hpi", "size")).reset_index()
    fa = fa[fa.n == 4]
    fa["real"] = fa.hpi / fa.year.map(cpi)
    x["hpi"] = _put(_blank(), fa, "real")
    # --- CDC overdose death rate per 100k (1999-2015 NCHS; 2016+ VSRR 12-month-ending December counts)
    d = pd.read_csv(US / "cdc_drug_poisoning_1999_2015.csv")
    d = d[(d.Sex == "Both Sexes") & (d.Age == "All Ages") & (d["Race and Hispanic Origin"] == "All Races-All Origins")]
    d["code"] = d.State.str.lower().map(BY_NAME)
    od = _put(_blank(), d.rename(columns={"Year": "year"}), "Crude Death Rate")
    v = pd.read_csv(US / "cdc_vsrr_overdose.csv")
    v = v[(v.Indicator == "Number of Drug Overdose Deaths") & (v.Month == "December") & (v.Year >= 2016)]
    v["deaths"] = pd.to_numeric(v["Data Value"].astype(str).str.replace(",", ""), errors="coerce")
    v = v.rename(columns={"State": "code", "Year": "year"})
    cnt = _put(_blank(), v, "deaths")
    with np.errstate(invalid="ignore"):
        od = np.where(np.isnan(od), cnt / x["pop"] * 1e5, od)
    x["overdose"] = od
    # --- CDC suicide, age-adjusted, 1999-2017
    l = pd.read_csv(US / "cdc_leading_causes.csv")
    l = l[l["Cause Name"] == "Suicide"]
    l["code"] = l.State.str.lower().map(BY_NAME)
    x["suicide"] = _put(_blank(), l.rename(columns={"Year": "year"}), "Age-adjusted Death Rate")
    # --- Mother Jones public mass shootings
    m = pd.read_csv(US / "motherjones_mass_shootings.csv")
    m["year"] = pd.to_datetime(m.date, format="mixed", errors="coerce").dt.year
    st = m.location.str.split(",").str[-1].str.strip()
    abbr = {s: s for s in CODES}
    m["code"] = st.map(lambda s: abbr.get(s.upper(), BY_NAME.get(s.lower())) if isinstance(s, str) else None)
    ms = m.dropna(subset=["code", "year"]).groupby(["code", "year"]).size().rename("n").reset_index()
    a = _blank()
    a[:, (YEARS >= 1982) & (YEARS <= 2026)] = 0
    x["shootings"] = _put(a, ms, "n")
    # --- presidential two-party Democratic margin (ffill between elections)
    pr = pd.read_csv(US / "president_1976_2024.csv")
    pr = pr[pr.party_simplified.isin(["DEMOCRAT", "REPUBLICAN"])]
    g = pr.pivot_table(index=["state_po", "year"], columns="party_simplified", values="candidatevotes", aggfunc="sum").reset_index()
    g["margin"] = (g.DEMOCRAT - g.REPUBLICAN) / (g.DEMOCRAT + g.REPUBLICAN)
    x["margin"] = _put(_blank(), g.rename(columns={"state_po": "code"}), "margin")
    # --- Shor-McCarty: distance between party medians in the legislature
    sm = pd.read_csv(US / "shor_mccarty_legislators.tab", sep="\t", low_memory=False)
    rows = []
    for y in range(1993, 2021):
        cols = [c for c in (f"house{y}", f"senate{y}") if c in sm.columns]
        serving = sm[sm[cols].notna().any(axis=1) & (sm[cols].fillna(0) != 0).any(axis=1)]
        med = serving.groupby(["st", "party"]).np_score.median().unstack()
        if "R" in med and "D" in med:
            for stc, val in (med["R"] - med["D"]).items():
                rows.append((stc, y, val))
    x["leg_polar"] = _put(_blank(), pd.DataFrame(rows, columns=["code", "year", "v"]), "v")
    # --- NOAA nClimDiv statewide annual temperature (degC), max temperature, precipitation (mm)
    noaa = {s[4]: s[0] for s in STATES if s[4] is not None and s[0] != "DC"}
    for v, fn, conv in (("temp", "tmpcst", lambda f: (f - 32) / 1.8), ("tmax", "tmaxst", lambda f: (f - 32) / 1.8),
                        ("precip", "pcpnst", lambda i: i * 25.4)):
        rows = []
        for line in open(US / f"climdiv-{fn}.txt"):
            code, vals = line[:10], line[10:].split()
            sc, year = int(code[:3]), int(code[6:10])
            if sc in noaa and len(vals) == 12:
                arr = np.array(vals, float)
                if (arr > -99).all():
                    rows.append((noaa[sc], year, conv(arr.sum() if v == "precip" else arr.mean())))
        x[v] = _put(_blank(), pd.DataFrame(rows, columns=["code", "year", "v"]), "v")
        dc = CODES.index("DC")
        x[v][dc] = x[v][CODES.index("MD")]                    # DC: Maryland's climate division data
    # --- national context: V-Dem polarisation of the United States
    p = get_panel()
    vp = p.x["polarization"][p.codes.index("USA")]
    nat = np.full(len(YEARS), np.nan)
    for i, y in enumerate(YEARS):
        if y in p.years:
            nat[i] = vp[p.yi(y)]
    x["us_polarization"] = np.tile(nat, (len(CODES), 1))
    return x


# ---------------------------------------------------------------------------

def _ffill(a, limit):
    out = a.copy()
    for j in range(1, a.shape[1]):
        miss = np.isnan(out[:, j])
        out[miss, j] = out[miss, j - 1]
    # undo fills longer than `limit`
    age = np.zeros(a.shape[0])
    for j in range(a.shape[1]):
        age = np.where(np.isfinite(a[:, j]), 0, age + 1)
        out[age > limit, j] = np.nan
    return out


def _lag(a, k):
    return np.concatenate([np.full((a.shape[0], k), np.nan), a[:, :-k]], 1)


def _roll(a, k, fn):
    out = np.full(a.shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(a.shape[1]):
            out[:, j] = fn(a[:, max(0, j - k + 1):j + 1], 1)
    return out


GROUPS = {
    "Economy": ["log_income", "income_growth_1y", "income_growth_5y", "income_vol_10y", "income_drawdown",
                "income_vs_us", "unemployment", "unemp_change_1y", "unemp_change_3y", "us_unemployment",
                "us_unemp_change_1y"],
    "Housing": ["hpi_growth_1y", "hpi_growth_5y", "hpi_vs_10y", "hpi_to_income"],
    "Demography": ["log_pop", "pop_growth_5y"],
    "Despair": ["overdose", "overdose_change_3y", "suicide"],
    "Politics": ["margin_abs", "margin_swing", "leg_polarization", "us_polarization"],
    "Violence": ["shootings_10y_pc", "us_shootings_5y"],
    "Climate shocks": ["temp_shock", "tmax_shock", "precip_shock"],
}
FEATURES = [f for g in GROUPS.values() for f in g]


def features(x):
    li = np.log(x["income"])
    g = np.diff(li, axis=1, prepend=np.nan)
    pop = x["pop"]
    w = pop / np.nansum(pop, 0, keepdims=True)
    us_u = np.nansum(np.nan_to_num(x["unemp"]) * w * np.isfinite(x["unemp"]), 0) / np.nansum(w * np.isfinite(x["unemp"]), 0)
    us_u = np.where(np.isfinite(x["unemp"]).any(0), us_u, np.nan)
    us_li = np.log(np.nansum(np.nan_to_num(x["income"]) * pop, 0) / np.nansum(pop * np.isfinite(x["income"]), 0))
    lh = np.log(x["hpi"])
    od = _ffill(x["overdose"], 2)
    margin = _ffill(x["margin"], 3)
    sh = x["shootings"]
    us_sh = np.nansum(sh, 0)
    us_sh = np.where(np.isfinite(sh).any(0), us_sh, np.nan)
    f = {
        "log_income": li, "income_growth_1y": g, "income_growth_5y": (li - _lag(li, 5)) / 5,
        "income_vol_10y": _roll(g, 10, np.nanstd), "income_drawdown": li - _roll(li, 10, np.nanmax),
        "income_vs_us": li - us_li[None, :],
        "unemployment": x["unemp"], "unemp_change_1y": x["unemp"] - _lag(x["unemp"], 1),
        "unemp_change_3y": x["unemp"] - _lag(x["unemp"], 3),
        "us_unemployment": np.tile(us_u, (len(CODES), 1)),
        "us_unemp_change_1y": np.tile(us_u - np.concatenate([[np.nan], us_u[:-1]]), (len(CODES), 1)),
        "hpi_growth_1y": lh - _lag(lh, 1), "hpi_growth_5y": (lh - _lag(lh, 5)) / 5,
        "hpi_vs_10y": lh - _roll(lh, 10, np.nanmean), "hpi_to_income": lh - li,
        "log_pop": np.log(pop), "pop_growth_5y": (np.log(pop) - _lag(np.log(pop), 5)) / 5,
        "overdose": od, "overdose_change_3y": np.log1p(od) - _lag(np.log1p(od), 3),
        "suicide": _ffill(x["suicide"], 7),
        "margin_abs": np.abs(margin), "margin_swing": margin - _ffill(_lag(x["margin"], 4), 3),
        "leg_polarization": _ffill(x["leg_polar"], 4), "us_polarization": _ffill(x["us_polarization"], 2),
        "shootings_10y_pc": _roll(sh, 10, np.nansum) / pop * 1e7,
        "us_shootings_5y": np.tile(np.convolve(np.nan_to_num(us_sh), np.ones(5), "full")[:len(YEARS)], (len(CODES), 1))
        * np.where(np.isfinite(us_sh), 1, np.nan)[None, :],
    }
    for v, name in (("temp", "temp_shock"), ("tmax", "tmax_shock")):
        f[name] = x[v] - _roll(_lag(x[v], 1), 10, np.nanmean)
    pm, ps = _roll(_lag(x["precip"], 1), 10, np.nanmean), _roll(_lag(x["precip"], 1), 10, np.nanstd)
    f["precip_shock"] = (x["precip"] - pm) / ps
    return np.stack([f[k] for k in FEATURES], -1)          # [S, Y, F]


def labels(x):
    S, Yn = x["income"].shape
    out = np.full((S, Yn, len(TASKS)), np.nan)
    last = {"income_shock": 2024, "unemployment_surge": 2024, "housing_bust": 2024, "overdose_surge": 2024,
            "mass_shooting": 2025}
    series = {"income_shock": x["income"], "unemployment_surge": x["unemp"], "housing_bust": x["hpi"],
              "overdose_surge": x["overdose"], "mass_shooting": x["shootings"]}
    for t in range(Yn):
        fut = slice(t + 1, t + H + 1)
        for k, task in enumerate(TASKS):
            if YEARS[t] + H > last[task]:
                continue
            a = series[task]
            now, f = a[:, t], a[:, fut]
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if task == "income_shock":
                    ev, risk = (f <= now[:, None] * 0.97).any(1), np.isfinite(now)
                elif task == "unemployment_surge":
                    ev, risk = (f >= now[:, None] + 2.5).any(1), np.isfinite(now)
                elif task == "housing_bust":
                    ev, risk = (f <= now[:, None] * 0.85).any(1), np.isfinite(now)
                elif task == "overdose_surge":
                    ev, risk = (f >= now[:, None] * 1.5).any(1), now >= 2
                else:
                    ev, risk = (f > 0).any(1), np.isfinite(now)
            obs = np.isfinite(f).all(1)
            y = np.where(ev, 1.0, np.where(obs, 0.0, np.nan))
            out[risk, t, k] = y[risk]
    return out


# ---------------------------------------------------------------------------

class StateModel:
    """logistic regression + gradient boosting per task, averaged."""

    def fit(self, X, Y):
        self.mu, self.sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
        Z = self._z(X)
        self.m = {}
        for k in range(Y.shape[1]):
            ok = np.isfinite(Y[:, k])
            if ok.sum() < 40 or Y[ok, k].min() == Y[ok, k].max():
                continue
            lr = LogisticRegression(C=0.1, max_iter=3000).fit(Z[ok], Y[ok, k])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gb = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=25,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3,
                                        verbose=-1).fit(X[ok], Y[ok, k])
            self.m[k] = (lr, gb)
        return self

    def _z(self, X):
        z = np.clip((X - self.mu) / self.sd, -6, 6)
        m = np.isnan(z)
        return np.concatenate([np.where(m, 0, z), m], 1)

    def predict_parts(self, X):
        Z = self._z(X)
        lr = np.full((len(X), len(TASKS)), np.nan)
        gb = lr.copy()
        for k, (a, b) in self.m.items():
            lr[:, k] = a.predict_proba(Z)[:, 1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gb[:, k] = b.predict_proba(X)[:, 1]
        return lr, gb

    def predict(self, X):
        lr, gb = self.predict_parts(X)
        return (lr + gb) / 2

    def drivers(self, X, k):
        lr = self.m[k][0]
        z = np.nan_to_num(np.clip((X - self.mu) / self.sd, -6, 6))
        c = z * lr.coef_[0][:len(FEATURES)]
        return {g: c[:, [FEATURES.index(f) for f in fs]].sum(1) for g, fs in GROUPS.items()}


def _flat(X, Y):
    S, Yn, F = X.shape
    si, ti = np.meshgrid(np.arange(S), np.arange(Yn), indexing="ij")
    return X.reshape(-1, F), Y.reshape(-1, Y.shape[-1]), si.ravel(), YEARS[ti.ravel()]


def _mask(Yf, year, origin, first=1960):
    Y = Yf.copy()
    Y[year + H > origin] = np.nan
    return Y, (year >= first) & np.isfinite(Y).any(1)


def backtest(X, Y, origins=range(1990, 2020, 5)):
    Xf, Yf, si, year = _flat(X, Y)
    rows = []
    for T in origins:
        Ytr, tr = _mask(Yf, year, T)
        m = StateModel().fit(Xf[tr], Ytr[tr])
        te = (year >= T) & (year < T + 5)
        lr, gb = m.predict_parts(Xf[te])
        base = np.nanmean(Ytr[tr], 0)
        for k, task in enumerate(TASKS):
            ok = np.isfinite(Yf[te, k])
            for name, pr in (("Logit", lr), ("GBM", gb), ("Average", (lr + gb) / 2)):
                rows.append(pd.DataFrame({"origin": T, "task": task, "model": name, "p": pr[ok, k],
                                          "y": Yf[te, k][ok], "base": base[k]}))
    bt = pd.concat(rows)
    summ = []
    for (task, model), g in bt.groupby(["task", "model"]):
        g = g.dropna()
        if g.y.nunique() < 2:
            continue
        b = ((g.base - g.y) ** 2).mean()
        summ.append({"task": task, "model": model, "n": len(g), "events": int(g.y.sum()),
                     "auc": roc_auc_score(g.y, g.p), "bss": 1 - ((g.p - g.y) ** 2).mean() / b})
    return bt, pd.DataFrame(summ)


def recalibrate(bt):
    """Platt map per task on the averaged backtest predictions; kept only where it improves
    leave-one-origin-out Brier. Also a skill verdict per task."""
    out = {}
    for task, g in bt[bt.model == "Average"].dropna().groupby("task"):
        lp = np.log(np.clip(g.p, 1e-4, 1 - 1e-4) / (1 - np.clip(g.p, 1e-4, 1 - 1e-4))).to_numpy()[:, None]
        y, o = g.y.to_numpy(), g.origin.to_numpy()
        raw = cal = 0.0
        for origin in np.unique(o):
            tr, te = o != origin, o == origin
            m = LogisticRegression(C=1e4).fit(lp[tr], y[tr])
            cal += ((m.predict_proba(lp[te])[:, 1] - y[te]) ** 2).sum()
            raw += ((g.p.to_numpy()[te] - y[te]) ** 2).sum()
        m = LogisticRegression(C=1e4).fit(lp, y)
        auc = roc_auc_score(y, g.p)
        brier_ref = ((g.base - g.y) ** 2).mean()
        best = min(raw, cal) / len(y)
        out[task] = {"a": float(m.intercept_[0]), "b": float(m.coef_[0, 0]), "use": bool(cal < raw),
                     "auc": float(auc), "bss": float(1 - best / brier_ref),
                     "skill": "none" if auc < 0.55 else ("low" if auc < 0.65 or best >= brier_ref else "useful")}
    return out


def climate_projection(x):
    """state warming by scenario: national CMIP6 delta (World Bank CCKP, delta-corrected) scaled by
    each state's observed 1970-2024 warming relative to the national trend (pattern scaling)."""
    from . import climate
    p = get_panel()
    obs = {v: p.x[f"c_{v}"] for v in climate.VARS}
    usa = p.codes.index("USA")
    yrs = np.arange(1970, ORIGIN + 1)
    idx = [int(y - FIRST) for y in yrs]
    t = x["temp"][:, idx]
    slope = np.array([np.polyfit(yrs[np.isfinite(r)], r[np.isfinite(r)], 1)[0] if np.isfinite(r).sum() > 20 else np.nan
                      for r in t])
    w = x["pop"][:, ORIGIN - FIRST]
    nat = np.nansum(slope * w) / np.nansum(w * np.isfinite(slope))
    scale = np.clip(np.nan_to_num(slope / nat, nan=1.0), 0.5, 2.0)
    base = np.nanmean(x["temp"][:, ORIGIN - FIRST - 9:ORIGIN - FIRST + 1], 1)
    out = {}
    for s in climate.SCENARIOS:
        out[s] = {}
        for st in ("median", "p10", "p90"):
            path = climate.scenario_paths(p.codes, p.years, s, obs, st)["tas"][usa]
            ref = np.nanmean(path[p.yi(2015):p.yi(ORIGIN) + 1])
            out[s][st] = {y: (base + scale * (np.mean(path[p.yi(y) - 2:p.yi(y) + 3]) - ref)).round(2) for y in (2035, 2050, 2065)}
    return {"base": base.round(2), "scale": scale.round(2), "scenarios": out}


def stress_index(X):
    """a Turchin-style composite: mean z-score (over states and years since 1999) of the
    pressures available for every state: unemployment, lost income momentum, overdoses,
    mass shootings, house prices vs income, electoral polarisation."""
    comp = {"unemployment": 1, "income_growth_5y": -1, "overdose": 1, "shootings_10y_pc": 1,
            "hpi_to_income": 1, "margin_abs": 1}
    sel = YEARS >= 1999
    zs = []
    for f, sign in comp.items():
        a = X[:, :, FEATURES.index(f)]
        mu, sd = np.nanmean(a[:, sel]), np.nanstd(a[:, sel])
        zs.append(sign * (a - mu) / sd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(np.stack(zs), 0), list(comp)


def run(verbose=True):
    x = load()
    X, Y = features(x), labels(x)
    if verbose:
        for k, t in enumerate(TASKS):
            v = Y[:, :, k]
            print(f"  {t:20s} rows {int(np.isfinite(v).sum()):5d} base rate {np.nanmean(v):.3f}")
    bt, summ = backtest(X, Y)
    Xf, Yf, si, year = _flat(X, Y)
    Ytr, tr = _mask(Yf, year, ORIGIN)
    m = StateModel().fit(Xf[tr], Ytr[tr])
    Xn = X[:, ORIGIN - FIRST]
    lr, gb = m.predict_parts(Xn)
    risk = (lr + gb) / 2
    cal = recalibrate(bt)
    for k, t in enumerate(TASKS):
        c = cal.get(t)
        if c and c["use"]:
            f = lambda p_: 1 / (1 + np.exp(-(c["a"] + c["b"] * np.log(np.clip(p_, 1e-4, 1 - 1e-4) / (1 - np.clip(p_, 1e-4, 1 - 1e-4))))))
            risk[:, k], lr[:, k], gb[:, k] = f(risk[:, k]), f(lr[:, k]), f(gb[:, k])
        if c and c["skill"] == "none":
            risk[:, k] = np.nan                       # no out-of-sample skill: do not forecast
    drv = {t: m.drivers(Xn, k) for k, t in enumerate(TASKS) if k in m.m}
    clim = climate_projection(x)
    psi, comps = stress_index(X)
    states = []
    for i, c in enumerate(CODES):
        r = {"code": c, "name": NAMES[i], "region": REGIONS[i], "fips": STATES[i][1],
             "pop_m": float(x["pop"][i, ORIGIN - FIRST] / 1e6),
             "income": float(x["income"][i, ORIGIN - FIRST]), "unemployment": float(x["unemp"][i, ORIGIN - FIRST]),
             "overdose": float(np.nan_to_num(X[i, ORIGIN - FIRST, FEATURES.index("overdose")], nan=-1)),
             "margin": float(np.nan_to_num(_ffill(x["margin"], 3)[i, ORIGIN - FIRST])),
             "temp_base": float(clim["base"][i]), "warming_scale": float(clim["scale"][i]),
             "climate": {s: {st: {str(y): float(v[i]) for y, v in d.items()} for st, d in sc.items()}
                         for s, sc in clim["scenarios"].items()},
             "stress": [None if not np.isfinite(v) else round(float(v), 3) for v in psi[i, 1990 - FIRST:ORIGIN - FIRST + 1]]}
        for k, t in enumerate(TASKS):
            r[t] = None if not np.isfinite(risk[i, k]) else float(risk[i, k])
            r[t + "_lo"] = float(np.nanmin([lr[i, k], gb[i, k]]))
            r[t + "_hi"] = float(np.nanmax([lr[i, k], gb[i, k]]))
            if t in drv:
                r[t + "_groups"] = {g: round(float(v[i]), 3) for g, v in drv[t].items()}
        states.append(r)
    w = x["pop"][:, 1990 - FIRST:ORIGIN - FIRST + 1]
    us_stress = np.nansum(np.nan_to_num(psi[:, 1990 - FIRST:ORIGIN - FIRST + 1]) * w, 0) / np.nansum(w, 0)
    base_rates = {t: float(np.nanmean(Ytr[tr][:, k])) for k, t in enumerate(TASKS)}
    return {"origin": ORIGIN, "tasks": TASK_NAMES, "base_rates": base_rates, "states": states,
            "stress_years": list(range(1990, ORIGIN + 1)), "us_stress": us_stress.round(3).tolist(),
            "stress_components": comps, "backtest": summ.to_dict("records"), "calibration": cal,
            "n_features": len(FEATURES), "groups": list(GROUPS)}
