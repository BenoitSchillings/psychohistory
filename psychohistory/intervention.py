"""Peacekeeping, the 2025 US aid cut and commodity-price shocks (scripts/intervention_test.py, outputs/intervention_test.json).

Peacekeeping (UN Peace and Security Data Hub, uniformed personnel by host, 1990-2025): in civil wars, more
peacekeepers predict an earlier end, +0.32 log-odds per log(1 + thousands) (s.e. 0.15), with year effects, income,
democracy, war duration and battle deaths held fixed (missions go to hard cases, so this leans conservative); relapse
after a war: -0.27 (s.e. 0.24), not significant, not used. The simulator keeps each host's 2025 deployment, cut by
a sampled share (the UN's 2025 liquidity crisis cut peacekeepers by about a quarter) and winding down at a sampled
rate; it sends no new missions to new wars (so it understates what peacekeeping could do).
Aid shocks (ODA % of GNI, 1975-2021): falls of more than 2% of GNI did not significantly predict onset, coups or
collapse (rejected as an estimate). The 2025 USAID cut (CGD estimate, % of GNI) enters only as a one-time loss of
demand in 2026, x a sampled multiplier 0-1 (assumption).
Commodity prices (WDI export shares x Pink Sheet real indices, 1966-2022): a fall in a country's export prices,
weighted by exports / GDP, raised coup attempts over the next two years (-2.8 log-odds per unit, s.e. 1.0); growth
+0.046 (s.e. 0.023), which the simulator already carries for oil (oil_beta). The coup channel uses the simulated food
price with each country's food export share x exports / GDP (oil exporters' coup risk already moves with oil prices
through the country models' oil-exposure features).
US tariffs 2025 (Census import data by partner; Yale Budget Lab effective rates): each partner loses
tariff_pass x (2024 exports to the US / GDP) x (rise in its effective tariff, last 2025 month vs 2024) of output in
2026, and the US loses tariff_us; both sampled (assumption: no historical episode of this size to estimate from;
the 2018-19 trade war cost China and the US a few tenths of a percent).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RAW

IV = RAW / "intervention"
PKO_BETA = 0.315
COMMOD_COUP = -2.8


def _nominal_gdp_usd(codes, last_year=2024):
    """GDP in current US dollars (External Wealth of Nations dataset, millions), each country's latest year"""
    try:
        e = pd.read_csv(RAW / "structure" / "ewn_country_year.csv", usecols=["iso3", "year", "gdp_usd"])
    except (FileNotFoundError, ValueError):
        return np.full(len(codes), np.nan)
    e = e[(e.year <= last_year) & e.gdp_usd.notna()].sort_values("year").groupby("iso3").gdp_usd.last() * 1e6
    return np.array([float(e.get(c, np.nan)) for c in codes])


def inputs(codes, origin, gdp=None):
    pk = pd.read_csv(IV / "un_pko_personnel_cy.csv")
    pk = pk[pk.year <= origin].sort_values("year").groupby("iso3").last()
    pk_n = np.array([float(pk.pko_total.get(c, 0.0)) if c in pk.index and pk.year.get(c, 0) >= origin - 1 else 0.0
                     for c in codes])
    cut = pd.read_csv(IV / "us_aid_cut_2025.csv").set_index("iso3").cgd_cut_pct_gni
    aid_cut = np.array([float(np.nan_to_num(cut.get(c, 0.0))) / 100 for c in codes])
    ce = pd.read_csv(IV / "commodity_exports_cy.csv")
    ce = ce[ce.year <= origin]
    def last(col):
        x = ce.dropna(subset=[col]).sort_values("year").groupby("iso3")[col].last()
        return np.array([float(x.get(c, np.nan)) for c in codes])
    ex = np.clip(np.nan_to_num(last("exports_gnfs_pct_gdp") / 100, nan=0.3), 0, 1.5)
    du = pd.read_csv(RAW / "political_economy" / "us_import_duties_by_partner.csv")
    du = du[du.is_group == 0].set_index(["iso3", "year"])
    tariff = np.zeros(len(codes))
    # exports to the US over GDP in current dollars (External Wealth of Nations, latest year to 2024). The simulator's
    # GDP is in 2011 PPP dollars, about twice nominal for developing exporters, which halved their exposure; it is the
    # fallback only where the nominal figure is missing (debate 2026-11-05; open since the 2026-10-28 review)
    nom = _nominal_gdp_usd(codes)
    for i, c in enumerate(codes):
        g_ = nom[i] if np.isfinite(nom[i]) and nom[i] > 0 else (gdp[i] if gdp is not None else np.nan)
        if (c, 2024) in du.index and (c, 2025) in du.index and np.isfinite(g_) and g_ > 0:
            imp = float(du.imports_cons_usd[(c, 2024)])
            rise = float(np.nan_to_num(du.last_month_eff_tariff_pct[(c, 2025)] - du.eff_tariff_pct[(c, 2024)])) / 100
            tariff[i] = min(imp / g_, 0.5) * max(rise, 0.0)
    return {"pko": pk_n, "tariff": tariff, "aid_cut": np.clip(aid_cut, 0, 0.2),
            "fuel": np.nan_to_num(last("fuel_pct_merch") / 100) * ex, "food": np.nan_to_num(last("food_pct_merch") / 100) * ex}
