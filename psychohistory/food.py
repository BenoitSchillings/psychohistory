"""Food balance in simulation: harvests against mouths, and a world grain price (data/raw/food/README.md).

Tested first as predictors (2026-10-25): world real food-price changes x cereal import dependency, import dependency,
cereal production per person, food inflation and relative food prices added nothing to forecasts of civil conflict,
coups, autocratization or economic collapse (dev score -0.001, 90% interval -0.005 to +0.004; the price-shock terms
alone +0.001, -0.003 to +0.004). The literature's link from food prices runs mainly to riots and protests, which the
model does not forecast. So food is a mechanism, not a predictor:

- per-person cereal use grows with income (elasticity ~0.13 for poor countries, less for rich ones, fit_demand);
- each country's cereal production = yield (simulated: Theil-Sen trend shrunk to the cross-country mean, warming,
  AI) x its cereal area, which grows at its 15-year trend (0-3%/yr, damped) up to 0.9 x arable land; its
  self-sufficiency moves with production growth minus population growth, so population pressure on limited land
  raises import dependency;
- the world real grain price (World Bank grains index) follows estimated equations (1962-origin): passes through
  oil-price changes, rises when the previous year's world harvest per person falls short of its trend, and otherwise
  replays the scenario year's historical residual; a weak pull toward its 20-year average;
- calories per person follow income (as before) and, estimated on FAOSTAT food balances, own production per person
  in proportion to self-sufficiency (a 10% drop costs ~0.3%), and world prices in proportion to import dependency
  (no measurable effect: +0.003 +/- 0.005 per unit dep x d log price). The previous assumption (calories x yield
  change^0.3) was about ten times the estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RAW

F = RAW / "food"
ANCHOR_START = 1975                  # grain prices: estimation start and the start of the long-run average


def _world(origin):
    """world grain-price changes, oil-price changes, last year's world cereal harvest per person (deviation from its
    pre-origin mean) and last year's gap of the price to its 20-year average. Harvests are FAOSTAT QCL production
    (no 2010 food-balance splice), chain-linked over the countries observed in both years so that countries entering
    or leaving the panel do not move the total."""
    w = pd.read_csv(F / "world_food_prices_y.csv").set_index("year")
    fb = pd.read_csv(F / "food_balance_cy.csv")
    oil = pd.read_csv(RAW / "energy" / "ei_oil_annual.csv").set_index("year").oil_real2024_usd_bbl
    un = pd.read_csv(RAW / "population-with-un-projections.csv")
    un["pop_un"] = un.population__sex_all__age_all__variant_estimates.fillna(un.population__sex_all__age_all__variant_medium__projected)
    un = un.rename(columns={"code": "iso3"})[["iso3", "year", "pop_un"]]
    fb = fb.merge(un.dropna(), on=["iso3", "year"], how="left")
    # (UN population: the food-balance population ends in 2023)
    fb["population_fbs"] = fb.pop_un.fillna(fb.population_fbs)
    q = fb[["iso3", "year", "cereal_production_qcl_t", "population_fbs"]].dropna()
    q = q[(q.cereal_production_qcl_t > 0) & (q.population_fbs > 0)]
    m = q.merge(q.assign(year=q.year + 1), on=["iso3", "year"], suffixes=("", "_l"))
    g = m.groupby("year")[["cereal_production_qcl_t", "population_fbs", "cereal_production_qcl_t_l", "population_fbs_l"]].sum()
    sup = (np.log(g.cereal_production_qcl_t / g.population_fbs)
           - np.log(g.cereal_production_qcl_t_l / g.population_fbs_l)).loc[:origin]
    sup_mean = float(sup.mean())
    lg = np.log(w.wb_grains_index_real)
    anchor = float(lg.loc[ANCHOR_START:origin].mean())
    d = pd.DataFrame({"dg": lg.diff(), "doil": np.log(oil).diff(), "dsup_l": (sup - sup_mean).shift(1),
                      "gap_l": (lg - anchor).shift(1)})
    return d.loc[ANCHOR_START + 1:origin].dropna(), sup_mean, w, anchor


def fit_world_price(origin):
    """grain price equation and the residual of each historical year (for replay), estimated from 1976: real grain
    prices revert toward their 1975..origin average (2026-10-25: 0.18 a year, se 0.06; intercept -0.016, se 0.016,
    kept, so the long-run level is ~8% below that average). A version pulling toward a moving 20-year average with
    the 1962-2025 intercept (-1.7% a year) implied an endless decline (5% of 2065 paths below a quarter of today's
    price); the post-1975 record is better described as stationary around a fixed level."""
    d, sup_mean, w, anchor = _world(origin)
    X = np.c_[np.ones(len(d)), d.doil, d.dsup_l, d.gap_l]
    b = np.linalg.lstsq(X, d.dg, rcond=None)[0]
    res = d.dg - X @ b
    lg = np.log(w.wb_grains_index_real)
    return {"const": float(b[0]), "oil": float(b[1]), "supply": float(b[2]), "revert": float(-b[3]),
            "resid": res.to_dict(), "resid_sd": float(res.std()), "supply_mean": sup_mean,
            "p0": float(lg.get(origin, lg.dropna().iloc[-1])), "anchor": anchor}


def _income(fb):
    """growth of GDP per person and its lagged log level for food-balance rows (iso3, year)"""
    from .data import get_panel
    from .features import base_arrays
    p = get_panel()
    b = base_arrays(p)
    yi = {y: i for i, y in enumerate(p.years)}
    ci = {c: i for i, c in enumerate(p.codes)}
    ix = [(ci.get(c, -1), yi.get(y, 0)) for c, y in zip(fb.iso3, fb.year)]
    g = np.array([b["lg"][i, t] - b["lg"][i, t - 1] if i >= 0 and t > 0 else np.nan for i, t in ix])
    lgl = np.array([b["lg"][i, t - 1] if i >= 0 and t > 0 else np.nan for i, t in ix])
    return g, lgl


def fit_calories(origin):
    """d log kcal = b_g * g + b_prod * (1 - dep+) * d log production per person + b_price * dep+ * d log world price
    (income growth estimated jointly, so the harvest term does not also carry the income effect of good harvests)"""
    fb = pd.read_csv(F / "food_balance_cy.csv").sort_values(["iso3", "year"])
    w = pd.read_csv(F / "world_food_prices_y.csv").set_index("year")
    lp = np.log(w.wb_grains_index_real)          # the index the simulator moves
    fb = fb[(fb.year <= origin) & (fb.levels_imputed == 0)].copy()     # both ends of each change observed
    fb["dk"] = np.log(fb.kcal_total_pc_day_adj).groupby(fb.iso3).diff()
    dep = fb.cereal_import_dependency.clip(0, 1).groupby(fb.iso3).shift(1)
    kg = (fb.cereal_production_t / fb.population_fbs * 1000).where(lambda v: v > 0)
    fb["own"] = (1 - dep) * np.log(kg).groupby(fb.iso3).diff()
    fb["px"] = dep * fb.year.map(lp.diff())
    fb["g"], _ = _income(fb)
    q = fb[(fb.year != 2010) & (fb.year.groupby(fb.iso3).diff() == 1)][["dk", "g", "own", "px"]].dropna()
    q = q[(q.dk.abs() < 0.2) & (q.g.abs() < 0.3)]
    X = np.c_[np.ones(len(q)), q.g, q.own, q.px]
    b = np.linalg.lstsq(X, q.dk, rcond=None)[0]
    return {"g": float(b[1]), "own": float(b[2]), "price": float(b[3]), "n": int(len(q))}


def fit_demand(origin):
    """income elasticity of cereal use per person (food, feed and other uses: FAOSTAT domestic supply), falling with
    income: d log use = e0 * g + e1 * g * (log GDP per person - 9). Estimated 1962..origin (2026-10-25: e0 0.127
    (se 0.016), e1 -0.021 (0.013); the world price has the wrong sign (stock building) and is not used)"""
    fb = pd.read_csv(F / "food_balance_cy.csv").sort_values(["iso3", "year"])
    fb = fb[(fb.year <= origin) & (fb.levels_imputed == 0)].copy()     # both ends of each change observed
    # one-year changes only (differenced before dropping the change across the 2009-2010 splice)
    fb["dds"] = np.log(fb.cereal_domestic_supply_t / fb.population_fbs).groupby(fb.iso3).diff()
    fb = fb[(fb.year != 2010) & (fb.year.groupby(fb.iso3).diff() == 1)].copy()
    fb["g"], fb["lgl"] = _income(fb)
    q = fb[["dds", "g", "lgl"]].dropna()
    q = q[q.dds.abs() < 0.3]
    X = np.c_[np.ones(len(q)), q.g, q.g * (q.lgl - 9)]
    c = np.linalg.lstsq(X, q.dds, rcond=None)[0]
    return {"e0": float(c[1]), "e1": float(c[2]), "n": int(len(q))}


def observed_food(origin):
    """change in the real FAO cereal price index in the (partly observed) year after the origin, or None"""
    w = pd.read_csv(F / "world_food_prices_y.csv").set_index("year")
    c = w.ffpi_cereals_real
    # only a year still in progress when the forecast is made (not a hindcast's known future)
    if origin + 1 in w.index and w.partial_year.get(origin + 1, 0) == 1 \
            and np.isfinite(c.get(origin + 1, np.nan)) and np.isfinite(c.get(origin, np.nan)):
        return float(np.log(c[origin + 1] / c[origin]))
    return None


def agri_share(p, origin):
    """agriculture, forestry and fishing as a share of GDP (World Bank via OWID): the latest value within ten years
    of the origin, else the median of the country's region"""
    a = pd.read_csv(RAW / "agriculture-share-gdp.csv")
    piv = a.pivot_table(index="code", columns="year", values=a.columns[-1]).reindex(index=list(p.codes))
    piv = piv.loc[:, [c for c in piv.columns if origin - 10 <= c <= origin]]
    last = piv.T.ffill().T.iloc[:, -1].to_numpy(float) / 100 if piv.shape[1] else np.full(len(p.codes), np.nan)
    reg = np.asarray(p.regions)
    for r in np.unique(reg):
        m = (reg == r) & ~np.isfinite(last)
        if m.any():
            last[m] = np.nanmedian(last[reg == r]) if np.isfinite(last[reg == r]).any() else np.nanmedian(last)
    return np.clip(np.nan_to_num(last, nan=0.1), 0, 0.7)


def dryland(p):
    """1 where 1961-1990 mean rainfall (ERA5) is below 600 mm a year: the 42 countries in which cereal area moved with
    decade-mean rainfall (elasticity 0.71, s.e. 0.29, 1961-2022; none in wetter countries)"""
    pr = np.asarray(p.x["c_pr"], float)
    yrs = np.asarray(p.years)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = np.nanmean(pr[:, (yrs >= 1961) & (yrs <= 1990)], 1)
    return (np.nan_to_num(b, nan=1000) < 600).astype(float)


def yield_residuals(p, origin):
    """log cereal yield minus its local trend (centred 11-year median, FAOSTAT via OWID), per country and year,
    using data up to the origin only. The simulator replays these by historical year, so a world reliving 1988 or
    2012 gets that year's harvests everywhere at once (bad years hit several breadbaskets together, as they did)."""
    d = pd.read_csv(RAW / "cereal-yield.csv")
    piv = d.pivot_table(index="code", columns="year", values=d.columns[-1]).reindex(index=list(p.codes), columns=list(p.years))
    piv = piv.loc[:, [c for c in piv.columns if c <= origin]]
    lg = np.log(piv.where(piv > 0))
    tr = lg.T.rolling(11, center=True, min_periods=6).median().T
    r = (lg - tr).reindex(columns=list(p.years))
    return r.to_numpy(float).clip(-0.8, 0.5)
