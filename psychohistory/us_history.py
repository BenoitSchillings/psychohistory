"""The United States over 230 years: a structural-demographic test (Turchin, Ages of Discord).

Annual series 1790-2025 (see data/raw/us_history/README.md):
  violence      Turchin's US Political Violence Database (riots, lynchings, terrorism, assassinations,
                insurrections), events per 10 million people, 1782-2025
  polarization  distance between the two main parties' medians on DW-NOMINATE dimension 1 (Voteview),
                House and Senate averaged, 1789-2026
  immiseration  inverse of the relative wage: production-worker pay / GDP per person (MeasuringWorth),
                10-year moving average because pre-1900 wages are published to the cent
  elites        lawyers per 100,000 people (ABA, 1878-2024, interpolated between benchmark years)
  fiscal        public debt / GDP (Jorda-Schularick-Taylor, 1870-2020; extended with the last value)
  democracy     V-Dem electoral democracy index for the US, 1789-2024

Questions:
  1. Does a structural stress index lead political violence by a decade, out of sample?
  2. Does it lead declines in democracy?
  3. Where is the US in 2024 relative to the 1850s, the 1910s and the late 1960s?
There are only four or five independent cycles in this history, so every test here is weak; the
module reports that alongside the estimates.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import RAW, ROOT

US = RAW / "us_history"
OUT = ROOT / "outputs"
HORIZON = 10


def load() -> pd.DataFrame:
    yrs = pd.Index(range(1790, 2026), name="year")
    v = pd.read_csv(US / "uspvd_annual.csv").set_index("year")
    e = pd.read_csv(US / "elite_annual.csv").set_index("year")
    vv = pd.read_csv(US / "voteview_annual.csv").groupby("year").mean(numeric_only=True)
    pop = e["population"].reindex(yrs).interpolate()
    df = pd.DataFrame(index=yrs)
    df["violence"] = (v["events_total"].reindex(yrs).fillna(0) / pop * 1e7)
    df["violence_deaths"] = (v["fatalities_total"].reindex(yrs).fillna(0) / pop * 1e7)
    df["polarization"] = vv[["house_party_median_gap", "senate_party_median_gap"]].mean(1).reindex(yrs)
    rw = e["relwage_production_comp"].reindex(yrs)
    df["immiseration"] = 1 / rw.rolling(10, min_periods=5).mean()
    df["elites"] = e["lawyers_per_100k"].reindex(yrs).interpolate(limit_area="inside")
    j = pd.read_csv(RAW / "energy" / "jst_cy.csv")
    debt = j[j.iso3 == "USA"].set_index("year")["debt_gdp"].reindex(yrs)
    df["fiscal"] = debt.ffill(limit=5)
    try:
        import pyreadr  # noqa: F401
        from .data import _vdem_cols
        d = _vdem_cols(["v2x_polyarchy"])
        df["democracy"] = d[d.code == "USA"].set_index("year")["v2x_polyarchy"].reindex(yrs)
    except Exception:
        df["democracy"] = np.nan
    df["violence_next"] = df["violence"][::-1].rolling(HORIZON, min_periods=HORIZON).mean()[::-1].shift(-1)
    df["dem_change_next"] = df["democracy"].shift(-HORIZON) - df["democracy"]
    return df


def stress_index(df, parts=("polarization", "immiseration", "elites", "fiscal")):
    """mean of z-scores over the parts available in each year (z over 1870-2024)."""
    ref = df.loc[1870:2024]
    z = pd.DataFrame({k: (df[k] - ref[k].mean()) / ref[k].std() for k in parts})
    return z.mean(1, skipna=True).where(z.notna().sum(1) >= 2), z


def rolling_test(df, x_cols, y_col="violence_next", origins=range(1880, 2016, 10), log=True):
    """fit y ~ x on years whose outcome window ends before the origin; predict the next decade."""
    rows = []
    for T in origins:
        tr = df.loc[:T - HORIZON - 1].dropna(subset=list(x_cols) + [y_col])
        te = df.loc[[T]].dropna(subset=list(x_cols) + [y_col])
        if len(tr) < 25 or te.empty:
            continue
        f = (lambda s: np.log1p(s)) if log else (lambda s: s)
        X = np.c_[np.ones(len(tr)), tr[list(x_cols)].to_numpy()]
        beta = np.linalg.lstsq(X, f(tr[y_col]).to_numpy(), rcond=None)[0]
        pred = float(np.r_[1, te[list(x_cols)].to_numpy()[0]] @ beta)
        naive = float(f(df.loc[T - HORIZON + 1:T, y_col.replace("_next", "")].mean())) if y_col == "violence_next" else 0.0
        rows.append({"origin": T, "actual": float(f(te[y_col]).iloc[0]), "pred": pred, "naive": naive})
    r = pd.DataFrame(rows)
    if r.empty:
        return {}
    mae, mae_naive = (r.pred - r.actual).abs().mean(), (r.naive - r.actual).abs().mean()
    corr = float(np.corrcoef(r.pred, r.actual)[0, 1]) if len(r) > 2 else np.nan
    return {"n": len(r), "mae": float(mae), "mae_naive": float(mae_naive), "skill": float(1 - mae / mae_naive) if mae_naive else None,
            "corr": corr, "rows": r.round(3).to_dict("records")}


def run(verbose=True):
    df = load()
    psi, z = stress_index(df)
    df["stress"] = psi
    out = {"years": df.index.tolist()}
    # 1. lead-lag correlations (in-sample, descriptive)
    lead = {}
    for k in ["stress", "polarization", "immiseration", "elites", "fiscal"]:
        ok = df[[k, "violence_next"]].dropna().loc[1870:]
        lead[k] = float(np.corrcoef(ok[k], np.log1p(ok["violence_next"]))[0, 1]) if len(ok) > 20 else None
    out["lead_corr_violence"] = lead
    # 2. out-of-sample decade forecasts of violence
    tests = {"stress": rolling_test(df, ["stress"]), "polarization": rolling_test(df, ["polarization"]),
             "stress+recent violence": rolling_test(df.assign(recent=np.log1p(df.violence.rolling(10).mean())), ["stress", "recent"])}
    out["violence_tests"] = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in tests.items()}
    out["violence_rows"] = tests["stress"].get("rows", [])
    # 3. democracy: does stress precede declines?
    dm = df[["stress", "polarization", "dem_change_next"]].dropna().loc[1870:]
    out["lead_corr_democracy"] = {k: float(np.corrcoef(dm[k], dm["dem_change_next"])[0, 1]) for k in ["stress", "polarization"]}
    out["democracy_tests"] = {k: {kk: vv for kk, vv in rolling_test(df, [k], "dem_change_next", log=False).items() if kk != "rows"}
                              for k in ["stress", "polarization"]}
    # 4. cycles: autocorrelation of decade-smoothed violence
    sm = np.log1p(df.violence.rolling(10, center=True, min_periods=5).mean()).loc[1800:2020].dropna()
    sm = sm - sm.rolling(60, center=True, min_periods=20).mean()
    ac = {lag: float(sm.autocorr(lag)) for lag in range(20, 81, 5)}
    out["violence_autocorr"] = ac
    peaks = df.violence.rolling(10, center=True).mean().loc[1800:2025]
    out["violence_decades"] = df.violence.groupby(df.index // 10 * 10).mean().round(2).to_dict()
    # 5. where are we now: 2024 vs past peaks
    now = {k: float(df.loc[2024, k]) if pd.notna(df.loc[2024, k]) else float(df[k].dropna().iloc[-1]) for k in
           ["stress", "polarization", "immiseration", "elites", "fiscal", "violence"]}
    past = {str(y): {k: (float(df.loc[y, k]) if pd.notna(df.loc[y, k]) else None) for k in now} for y in (1858, 1913, 1968)}
    out["now"], out["past"] = now, past
    # 6. forecast for 2025-2034 from the stress model fitted on all history
    tr = df.dropna(subset=["stress", "violence_next"])
    X = np.c_[np.ones(len(tr)), tr["stress"]]
    beta, res, *_ = np.linalg.lstsq(X, np.log1p(tr["violence_next"]), rcond=None)
    resid_sd = float(np.std(np.log1p(tr["violence_next"]) - X @ beta))
    mu = float(beta[0] + beta[1] * now["stress"])
    out["forecast_2025_2034"] = {"violence_per_10m_median": float(np.expm1(mu)),
                                 "band_80": [float(np.expm1(mu - 1.2816 * resid_sd)), float(np.expm1(mu + 1.2816 * resid_sd))],
                                 "historical_median": float(df.violence.loc[1870:2024].median()),
                                 "percentile_vs_history": float((tr["violence_next"] < np.expm1(mu)).mean())}
    out["series"] = {k: [None if not np.isfinite(v) else round(float(v), 4) for v in df[k]] for k in
                     ["violence", "polarization", "immiseration", "elites", "fiscal", "stress", "democracy"]}
    OUT.mkdir(exist_ok=True)
    (OUT / "us_history.json").write_text(json.dumps(out, indent=1, default=lambda o: None))
    if verbose:
        print(json.dumps({k: out[k] for k in ["lead_corr_violence", "violence_tests", "lead_corr_democracy", "democracy_tests",
                                              "now", "past", "forecast_2025_2034"]}, indent=1))
        print("autocorr", {k: round(v, 2) for k, v in ac.items()})
    return out


if __name__ == "__main__":
    run()
