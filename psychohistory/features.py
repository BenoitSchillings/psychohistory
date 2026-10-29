"""Structural features and outcome labels.

`compute_features` is written in torch over a batch of "worlds" so the exact same
code produces training features from history and features inside the Monte Carlo
simulator (where the batch dimension is the scenario index).

Feature groups loosely follow structural-demographic theory (Goldstone, Turchin):
mass mobilisation potential (immiseration, youth bulge, urbanisation), elite
mobilisation potential (elite overproduction, inequality), state fiscal stress,
plus regime type, ideology and religion, conflict history / contagion and climate shocks.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import Panel, LAST_OBS_YEAR

W = 11  # history window: t-10 .. t

# variables that need a history window (the simulator evolves these)
WINDOW_VARS = ["lg", "dem", "conflict", "urban", "lpop", "tertiary_enrol", "tas", "pr", "cdd", "txx", "oil_price"]
CLIMATE_VARS = ["tas", "pr", "cdd", "txx"]   # ERA5 observed; CMIP6 scenario paths in simulation
# variables only needed at time t
CURRENT_VARS = ["median_age", "youth", "fertility", "log_cm", "life_exp", "ysc",
                "inflation", "unemployment", "extreme_poverty", "calories", "cereal_yield", "trade",
                "oil_pc", "tertiary_share", "schooling", "top1", "gini", "gov_exp", "tax_rev",
                "military", "corruption", "refugees_pc", "homicide", "pre1946", "war", "interstate_recent",
                "oil_price_20y", "coups_total", "conflict_years", "obs_years", "collapses_total",
                "battle_deaths_pc", "nonstate_3y_pc", "onesided_3y_pc",
                "landlocked", "dist_coast", "near_coast", "rugged", "desert", "tropical", "area_km2",
                "arable_pc", "water_stress", "pop_density", "energy_pc", "fossil_self_suff", "low_carbon_share"]
# slow-moving structural variables used as-is (feature name == variable name); persisted in simulation
PASSTHROUGH = {
    "Ideology & religion": ["legit_ideology", "legit_nationalist", "legit_socialist", "legit_restorative",
                            "legit_separatist", "legit_religious", "legit_performance", "polarization",
                            "mass_mobilization", "religious_freedom", "religious_repression",
                            "rel_fractionalization", "rel_polarization", "share_muslim", "share_nonreligious",
                            "ethnic_excluded", "ethnic_discriminated"],
    "State capacity & coercion": ["fiscal_capacity", "political_killings", "physical_integrity",
                                  "neopatrimonialism", "rule_of_law", "power_by_wealth"],
    "Fiscal stress & coups": ["public_debt", "debt_change_5y", "fin_crisis_recent", "coup_recent"],
    "Institutions": ["jud_constraints", "leg_constraints", "horiz_acc", "diag_acc", "free_expression",
                     "media_censorship", "cso_repression", "mob_democracy", "mob_autocracy", "regime_military",
                     "regime_party", "regime_hereditary", "regime_confidence", "party_inst", "free_fair_elections",
                     "elected_regime", "exec_respects_const", "support_group_size", "opposition_group_size",
                     "log_regime_duration", "liberal_dem", "civil_liberties", "cso_participation", "clientelism",
                     "torture", "political_violence", "admin_impartial", "territorial_control",
                     "election_year", "years_since_election",
                     "jucon_change_5y", "legcon_change_5y", "freexp_change_5y", "libdem_change_5y", "csrepress_change_5y"],
    "Leaders": ["leader_log_tenure", "leader_age", "leader_military", "leader_elected", "election_anticipated",
                "elections_delayed", "months_since_irregular", "months_since_loss"],
    # add-on families (data.py @extra loaders deliver finished arrays as p.x["feat__<name>"])
    "Parties": ["vp_gov_antiplural", "vp_gov_populism", "vp_all_antiplural_votew", "vp_max_antiplural_major",
                "vp_gov_reject_violence", "vp_coalition_n_parties"],
    # (tested and rejected, 2026-10-24: cabinet turnover and composition (WhoGov), irregular leader exits (Archigos),
    #  regime type and personalism (Geddes-Wright-Frantz), election disputes (NELDA))
    # (tested and rejected, 2026-10-24: annual within-country surges in machine-coded events (ICEWS 1995-2022,
    #  POLECAT 2023-2024) -- conflict, protest, repression, violence, threats, coverage; the signal lives at the
    #  monthly scale, see scripts/monthly_nowcast.py)
    # (tested and rejected, 2026-10-25: migration -- net migration rate, migrant share and its 5-year change,
    #  refugees hosted per person, remittances (UN WPP, UN DESA, UNHCR, World Bank): dev score +0.004, 90% interval
    #  -0.001 to +0.009. Migration enters the simulator as a mechanism instead, see migration.py)
    # (tested and rejected, 2026-10-25: food -- world real food-price changes x cereal import dependency (1 and 3
    #  years), import dependency, cereal production per person, food inflation and relative food prices: dev score
    #  -0.001 (90% interval -0.005 to +0.004); the two price-shock terms alone +0.001 (-0.003 to +0.004). Food enters
    #  the simulator as a mechanism instead, see food.py)
    # personal rule and the regime's support coalition (V-Dem; data.load_personal), adopted 2026-10-28
    "Personal rule": ["legit_personalist", "personalization_5y", "presidentialism", "regime_corruption",
                      "support_military", "support_foreign", "support_ethnic", "path_force"],
    # the rest of the regime coalition (data.load_coalition), 2026-10-28
    "Regime coalition": ["opp_military", "opp_business_elites", "sup_religious", "sup_civil_servants",
                         "sup_rural_workers", "sup_agrarian_elites"],
    "Protest": ["protest_recent", "protest_violent_recent", "protest_repressed_recent", "navco_campaigns_ongoing",
                "navco_nonviolent_ongoing", "navco_violent_ongoing"],
    # (tested and rejected, 2026-10-24: fuel exports/imports as % of merchandise trade (WDI); international
    #  crises (ICB), UN voting distance from the US and its 5-year change, press geopolitical risk (GPR))
    # (tested and rejected: Global Macro Database exchange rates, real exchange rate, current account,
    #  deficit, interest rates, money growth, investment, exports and house prices)
}
# features that only some tasks may use ({feature: tasks}); every model family blanks them for the other tasks
# (models.task_blocked). The regime-coalition family sharpened coup and autocratisation forecasts but blurred the
# civil-conflict ones in the simulator hindcasts (2026-10-28), so it feeds those two only
TASK_ONLY = {f: {"coup_1y", "coup_5y", "autocratization_5y"} for f in PASSTHROUGH["Regime coalition"]}
# event memories: exponentially decaying counts, updated (or just decayed) in simulation
ADDON_GROUPS = ["Parties", "Protest", "Personal rule", "Regime coalition"]
DECAY = {"coup_recent": 0.7, "fin_crisis_recent": 0.7, "debt_change_5y": 0.8, "interstate_recent": 0.7,
         "nonstate_3y_pc": 0.6, "onesided_3y_pc": 0.6,
         # recent changes fade toward zero in simulation (the level they produced persists)
         "jucon_change_5y": 0.8, "personalization_5y": 0.8, "legcon_change_5y": 0.8, "freexp_change_5y": 0.8, "libdem_change_5y": 0.8,
         "csrepress_change_5y": 0.8}
CURRENT_VARS += [v for g in PASSTHROUGH.values() for v in g]

# Variables that trend world-wide over time (world conflict share, internet and mobile
# adoption, temperature level) are deliberately excluded: they act as calendar dummies
# and extrapolate era effects into the future. Climate enters only as a detrended shock.
GROUPS = {
    "Economy / immiseration": ["log_gdp", "gdp_gap_frontier", "growth_1y", "growth_5y", "growth_vol_10y",
                               "gdp_drawdown", "inflation_log", "unemployment", "extreme_poverty",
                               "log_calories", "cereal_yield", "trade", "log_oil_pc"],
    "Demography": ["median_age", "youth_share", "log_pop", "pop_growth_5y", "fertility", "urban",
                   "urban_change_10y", "log_child_mort", "life_exp"],
    "Elites & inequality": ["tertiary_enrol", "tertiary_change_10y", "tertiary_share", "schooling",
                            "top1_share", "gini"],
    "State & regime": ["gov_exp", "tax_rev", "military", "democracy", "democracy_sq", "democracy_change_5y",
                       "corruption"],
    "Violence & contagion": ["conflict_now", "conflict_past_10y", "log_years_since_conflict",
                             "log_refugees_pc", "log_homicide", "region_conflict", "region_dem_gap",
                             "war_now", "interstate_recent",
                             # distance-weighted neighbours (exp(-d/800 km)), see set_spatial()
                             "nb_conflict", "nb_coups", "nb_dem_gap", "nb_dem_change"],
    "Economic contagion": ["nb_growth_1y", "nb_growth_5y", "nb_drawdown", "nb_inflation"],
    # (tested and rejected: UCDP battle deaths, non-state and one-sided violence, 1989+; loader kept in data.py)
    # (tested and rejected: long-memory counts of past coups / war years / collapses since 1946)
    # weather shocks relative to the country's previous decade (detrended by construction)
    "Climate shocks": ["temp_shock", "heat_shock", "rain_shock", "drought_shock"],
    # (tested and rejected: the global oil-price cycle and its interaction with oil exposure)
    # (tested and rejected for crisis prediction: geography — landlocked, distance to coast, ruggedness,
    #  desert, tropics, area — and resources — arable land, water stress, density, energy per person,
    #  fossil self-sufficiency. They enter the world simulator as mechanisms instead; see world.py)
    # 1 before 1946: lets models absorb the higher base rates of the war-and-depression era
    # when learning from long-run history (always 0 in forecasts)
    "Era": ["pre1946"],
    **PASSTHROUGH,
}
FEATURES = [f for g in GROUPS.values() for f in g]
F = len(FEATURES)


def _nanmean(x, dim):
    m = torch.isfinite(x)
    s = torch.where(m, x, torch.zeros_like(x)).sum(dim)
    n = m.sum(dim)
    return torch.where(n > 0, s / n.clamp(min=1), torch.full_like(s, float("nan")))


def _nanstd(x, dim):
    mu = _nanmean(x, dim).unsqueeze(dim)
    n = torch.isfinite(x).sum(dim)
    sd = torch.sqrt(_nanmean((x - mu) ** 2, dim))
    return torch.where(n >= 2, sd, torch.full_like(sd, float("nan")))


def _group_mean_excl(v, regions, n_regions):
    """mean of v over other members of the same region; v [B, C] (nan-aware)."""
    m = torch.isfinite(v)
    vz = torch.where(m, v, torch.zeros_like(v))
    onehot = torch.nn.functional.one_hot(regions, n_regions).to(v.dtype)  # [C, R]
    s = vz @ onehot                      # [B, R]
    n = m.to(v.dtype) @ onehot
    s_ex = s[:, regions] - vz            # [B, C]
    n_ex = n[:, regions] - m.to(v.dtype)
    return torch.where(n_ex > 0, s_ex / n_ex.clamp(min=1), torch.full_like(v, float("nan")))


SPATIAL = {"W": None}


def set_spatial(codes, scale_km=800.0):
    """distance-decay weights between countries (centroids), used for neighbour features."""
    import pandas as pd
    from .data import RAW
    # keep_default_na: Namibia's code "NA" is not missing. Keep each country's own row, not overseas
    # territories that share its code (the Canary Islands would otherwise overwrite Spain).
    c = pd.read_csv(RAW / "country_centroids.csv", keep_default_na=False)
    c = c[c.COUNTRY == c.COUNTRYAFF].drop_duplicates("ISO", keep="first")
    iso = pd.read_csv(RAW / "iso3166.csv", dtype=str, keep_default_na=False)
    c["iso3"] = c.ISO.map(dict(zip(iso["alpha-2"], iso["alpha-3"])))
    pos = {r.iso3: (r.latitude, r.longitude) for r in c.itertuples() if isinstance(r.iso3, str)}
    pos.update({"HKG": (22.3, 114.2), "TWN": (23.7, 121.0)})
    lat = np.radians([pos.get(k, (np.nan, np.nan))[0] for k in codes])
    lon = np.radians([pos.get(k, (np.nan, np.nan))[1] for k in codes])
    dlat, dlon = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    d = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    w = np.exp(-np.nan_to_num(d, nan=1e9) / scale_km)
    np.fill_diagonal(w, 0.0)
    SPATIAL["W"] = torch.as_tensor(w, dtype=torch.float32)


def _neighbours(v):
    """distance-weighted mean of v over other countries; v [B, C] (nan-aware)."""
    W = SPATIAL["W"]
    if W is None:
        return torch.full_like(v, float("nan"))
    if W.device != v.device:
        W = SPATIAL["W"] = W.to(v.device)
    m = torch.isfinite(v).to(v.dtype)
    s = torch.where(m > 0, v, torch.zeros_like(v)) @ W.T
    n = m @ W.T
    return torch.where(n > 1e-3, s / n.clamp(min=1e-6), torch.full_like(v, float("nan")))


def compute_features(h: dict, cur: dict, regions: torch.Tensor, n_regions: int) -> torch.Tensor:
    """h[name]: [B, C, W] window ending at t; cur[name]: [B, C]. Returns [B, C, F] in FEATURES order."""
    lg, dem, conf = h["lg"], h["dem"], h["conflict"]
    d = torch.diff(lg, dim=-1)
    now = lambda a: a[..., -1]
    lg_now, dem_now, conf_now, lpop = now(lg), now(dem), now(conf), h["lpop"]
    frontier = torch.nanquantile(lg_now, 0.9, dim=-1, keepdim=True)  # ~ rich-world frontier
    f = {
        "log_gdp": lg_now, "gdp_gap_frontier": lg_now - frontier, "growth_1y": d[..., -1],
        "growth_5y": (lg_now - lg[..., -6]) / 5, "growth_vol_10y": _nanstd(d, -1),
        "gdp_drawdown": lg_now - torch.nan_to_num(lg, nan=-1e9).amax(-1),
        "inflation_log": torch.sign(cur["inflation"]) * torch.log1p(cur["inflation"].abs().clamp(max=1e4)),
        "unemployment": cur["unemployment"], "extreme_poverty": cur["extreme_poverty"],
        "log_calories": torch.log(cur["calories"]), "cereal_yield": cur["cereal_yield"], "trade": cur["trade"],
        "log_oil_pc": torch.log1p(cur["oil_pc"]),
        "median_age": cur["median_age"], "youth_share": cur["youth"], "log_pop": now(lpop),
        "pop_growth_5y": (now(lpop) - lpop[..., -6]) / 5, "fertility": cur["fertility"],
        "urban": now(h["urban"]) / 100, "urban_change_10y": (now(h["urban"]) - h["urban"][..., 0]) / 100,
        "log_child_mort": cur["log_cm"], "life_exp": cur["life_exp"],
        "tertiary_enrol": now(h["tertiary_enrol"]),
        "tertiary_change_10y": now(h["tertiary_enrol"]) - h["tertiary_enrol"][..., 0],
        "tertiary_share": cur["tertiary_share"], "schooling": cur["schooling"], "top1_share": cur["top1"],
        "gini": cur["gini"], "gov_exp": cur["gov_exp"], "tax_rev": cur["tax_rev"], "military": cur["military"],
        "democracy": dem_now, "democracy_sq": dem_now ** 2, "democracy_change_5y": dem_now - dem[..., -6],
        "corruption": cur["corruption"],
        "conflict_now": conf_now, "conflict_past_10y": _nanmean(conf[..., :-1], -1),
        "log_years_since_conflict": torch.log1p(cur["ysc"].clamp(max=60)),
        "log_refugees_pc": torch.log1p(cur["refugees_pc"] * 1e3), "log_homicide": torch.log1p(cur["homicide"]),
        "region_conflict": _group_mean_excl(conf_now, regions, n_regions),
        "region_dem_gap": _group_mean_excl(dem_now, regions, n_regions) - dem_now,
        "pre1946": cur["pre1946"],
        "war_now": cur["war"], "interstate_recent": cur["interstate_recent"],
        "landlocked": cur["landlocked"], "log_dist_coast": torch.log1p(cur["dist_coast"]),
        "near_coast": cur["near_coast"], "rugged": cur["rugged"], "desert": cur["desert"], "tropical": cur["tropical"],
        "log_area": torch.log(cur["area_km2"]), "log_arable_pc": torch.log(cur["arable_pc"] + 1e-3),
        "log_water_stress": torch.log1p(cur["water_stress"]), "log_pop_density": torch.log1p(cur["pop_density"]),
        "log_energy_pc": torch.log1p(cur["energy_pc"]), "log_fossil_self_suff": torch.log1p(cur["fossil_self_suff"].clamp(0, 50)),
        "low_carbon_share": cur["low_carbon_share"],
        "log_battle_deaths": torch.log1p(cur["battle_deaths_pc"]),
        "log_nonstate_3y": torch.log1p(cur["nonstate_3y_pc"]), "log_onesided_3y": torch.log1p(cur["onesided_3y_pc"]),
        "log_coups_total": torch.log1p(cur["coups_total"]),
        "war_share_history": cur["conflict_years"] / cur["obs_years"].clamp(min=1),
        "log_collapses_total": torch.log1p(cur["collapses_total"]),
        "oil_price_change_3y": now(h["oil_price"]) - h["oil_price"][..., -4],
        "oil_price_vs_20y": now(h["oil_price"]) - cur["oil_price_20y"],
        "oil_exposure_x_change": torch.log1p(cur["oil_pc"]) / 10 * (now(h["oil_price"]) - h["oil_price"][..., -4]),
        "oil_exposure_x_level": torch.log1p(cur["oil_pc"]) / 10 * (now(h["oil_price"]) - cur["oil_price_20y"]),
        "nb_conflict": _neighbours(conf_now), "nb_coups": _neighbours(cur["coup_recent"]),
        "nb_dem_gap": _neighbours(dem_now) - dem_now, "nb_dem_change": _neighbours(dem_now - dem[..., -6]),
        "nb_growth_1y": _neighbours(d[..., -1]), "nb_growth_5y": _neighbours((lg_now - lg[..., -6]) / 5),
        "nb_drawdown": _neighbours(lg_now - torch.nan_to_num(lg, nan=-1e9).amax(-1)),
        "nb_inflation": _neighbours(torch.sign(cur["inflation"]) * torch.log1p(cur["inflation"].abs().clamp(max=1e4))),
        "temp_shock": now(h["tas"]) - _nanmean(h["tas"][..., :-1], -1),
        "heat_shock": now(h["txx"]) - _nanmean(h["txx"][..., :-1], -1),
        "rain_shock": (now(h["pr"]) - _nanmean(h["pr"][..., :-1], -1)) / (_nanstd(h["pr"][..., :-1], -1) + 1e-6),
        "drought_shock": (now(h["cdd"]) - _nanmean(h["cdd"][..., :-1], -1)) / (_nanstd(h["cdd"][..., :-1], -1) + 1e-6),
    }
    for g in PASSTHROUGH.values():
        for k in g:
            f[k] = cur[k]
    return torch.stack([f[k] for k in FEATURES], -1)


# ---------------------------------------------------------------------------
# Panel -> base arrays

def _ffill(a: np.ndarray, limit: int) -> np.ndarray:
    """forward fill along years, at most `limit` steps (sparse survey data)."""
    out = a.copy()
    C, Y = a.shape
    last_val = np.full(C, np.nan)
    age = np.full(C, 10 ** 6)
    for y in range(Y):
        fin = np.isfinite(a[:, y])
        last_val[fin] = a[fin, y]
        age[fin] = 0
        fill = ~fin & (age <= limit)
        out[fill, y] = last_val[fill]
        age += 1
        age[fin] = 1
    return out


def _decayed(events: np.ndarray, rate: float) -> np.ndarray:
    """r_t = rate * r_{t-1} + e_t; nan until the country's series starts, nan events count as 0 after."""
    out = np.full(events.shape, np.nan)
    r = np.full(events.shape[0], np.nan)
    for y in range(events.shape[1]):
        e = events[:, y]
        started = np.isfinite(r) | np.isfinite(e)
        r = np.where(started, rate * np.nan_to_num(r) + np.nan_to_num(e), np.nan)
        out[:, y] = r
    return out


def base_arrays(p: Panel) -> dict[str, np.ndarray]:
    x = p.x
    pop = x["pop"]
    # state-system entry: the Powell & Thyne coup series starts at independence (or 1950).
    # Before entry a territory's wars were coded to its predecessor state, so its own
    # conflict record is unknown, not peaceful (e.g. South Sudan before 2011).
    ca = x["coup_attempts"]
    # (Powell & Thyne years only: the V-Dem pre-1950 coup patch, data.load_history_coups, would otherwise move entry
    # back for 19 countries, e.g. the Baltic states to 1918-20 and Namibia to 1926; code review 2026-10-28)
    has = np.isfinite(ca) & (p.years[None, :] >= 1950)
    entry = np.where(has.any(1), p.years[np.argmax(has, 1)], p.years[0])
    pre_entry = (p.years[None, :] < entry[:, None]) & (entry[:, None] > 1950)
    conflict = np.where(pre_entry, np.nan, x["conflict"])
    b = {
        "lg": np.log(x["gdp_pc"]),
        "dem": x["democracy"],
        "conflict": conflict,
        "urban": x["urban"],
        "lpop": np.log(pop),
        # (carried up to 3 years, as the simulator does, so an origin year after the last ERA5 year does not lose its
        # weather shocks; ERA5 runs to 2025 since 2026-10-28)
        **{k: _ffill(x[f"c_{k}"], 3) for k in CLIMATE_VARS},
        "tertiary_enrol": _ffill(x["tertiary_enrol"], 5),
        "median_age": x["median_age"],
        "youth": x["pop_0_14"] / pop,
        "fertility": x["fertility"],
        "log_cm": np.log(_ffill(x["child_mort"], 5)),
        "life_exp": _ffill(x["life_exp"], 5),
        "oil_pc": _ffill(x["oil_twh"], 3) * 1e9 / pop,  # kWh per person
        "refugees_pc": _ffill(x["refugees"], 3) / pop,
    }
    for k in ["inflation", "unemployment", "calories", "cereal_yield", "trade", "gov_exp", "tax_rev",
              "military", "corruption", "homicide"]:
        b[k] = _ffill(x[k], 3)
    for k in ["extreme_poverty", "tertiary_share", "schooling", "top1", "gini"]:
        b[k] = _ffill(x[k], 10)
    # ideology, religion, state capacity (V-Dem ends 2023 for legitimation items)
    for k in PASSTHROUGH["Ideology & religion"][:11] + PASSTHROUGH["State capacity & coercion"]:
        b[k] = _ffill(x[k], 3)
    for k in ["rel_fractionalization", "rel_polarization", "share_muslim", "share_nonreligious"]:
        b[k] = _ffill(x[k], 15)   # WRP ends 2010; composition is slow-moving
    b["ethnic_excluded"] = _ffill(x["excluded"], 5)
    b["ethnic_discriminated"] = _ffill(x["discriminated"], 5)
    debt = _ffill(x["public_debt"], 3)
    b["public_debt"] = debt
    b["debt_change_5y"] = debt - np.concatenate([np.full((debt.shape[0], 5), np.nan), debt[:, :-5]], 1)
    # institutions (V-Dem): levels held up to 3 years, 5-year changes as erosion signals
    for k in PASSTHROUGH["Institutions"]:
        if k in x:
            b[k] = _ffill(x[k], 3)
    b["log_regime_duration"] = np.log1p(_ffill(x["regime_duration"], 3))
    for g in ADDON_GROUPS:
        for k in PASSTHROUGH[g]:
            b[k] = x["feat__" + k] if "feat__" + k in x else np.full(x["pop"].shape, np.nan)
    # V-Party's coding ends with the 2019 elections: after that, if the head of government changed since the
    # last coded election, the governing party's scores are unknown rather than carried forward
    if "feat__vp_vparty_election_year" in x and "leader_change" in x:
        ey, lc = x["feat__vp_vparty_election_year"], np.nan_to_num(x["leader_change"])
        stale = np.zeros(ey.shape, bool)
        for t in range(ey.shape[1]):
            if p.years[t] < 2020:
                continue
            since = (p.years[None, :] > ey[:, t, None] + 1) & (p.years[None, :] <= p.years[t])  # +1: inauguration year
            stale[:, t] = (lc * since).sum(1) > 0
        for k in ("vp_gov_antiplural", "vp_gov_populism", "vp_gov_reject_violence"):
            b[k] = np.where(stale, np.nan, b[k])
    d5 = lambda a: a - np.concatenate([np.full((a.shape[0], 5), np.nan), a[:, :-5]], 1)
    d3 = lambda a: a - np.concatenate([np.full((a.shape[0], 3), np.nan), a[:, :-3]], 1)
    for name, src in [("jucon_change_5y", "jud_constraints"), ("legcon_change_5y", "leg_constraints"),
                      ("freexp_change_5y", "free_expression"), ("libdem_change_5y", "liberal_dem"),
                      ("csrepress_change_5y", "cso_repression")]:
        b[name] = d5(b[src])
    el = x["election"]
    b["election_year"] = np.where(np.isfinite(x["democracy"]), np.nan_to_num(el), np.nan)
    ys = np.full(el.shape, np.nan)
    run = np.full(el.shape[0], np.nan)
    for y in range(el.shape[1]):
        e = el[:, y]
        run = np.where(e == 1, 0.0, np.where(np.isfinite(run), run + 1, np.where(np.isfinite(e), 10.0, np.nan)))
        ys[:, y] = run
    b["years_since_election"] = np.minimum(ys, 30)
    # macro-finance (Global Macro Database)
    lfx = np.log(_ffill(x["gmd_USDfx"], 2))
    b["fx_depreciation_1y"] = lfx - np.concatenate([np.full((lfx.shape[0], 1), np.nan), lfx[:, :-1]], 1)
    b["fx_depreciation_3y"] = d3(lfx)
    b["reer_change_3y"] = d3(np.log(_ffill(x["gmd_REER"], 2)))
    b["current_account"] = _ffill(x["gmd_CA_GDP"], 2)
    b["gov_deficit"] = _ffill(x["gov_deficit"], 2)
    b["long_rate"] = _ffill(x["gmd_ltrate"], 2)
    b["short_rate"] = _ffill(x["gmd_strate"], 2)
    infl = _ffill(x["inflation"], 3)
    b["real_rate"] = b["short_rate"] - np.clip(infl, -50, 100)
    b["money_growth_3y"] = d3(np.log(_ffill(x["gmd_M2"], 2))) - d3(np.log(_ffill(x["gmd_nGDP"], 2)))
    b["investment_share"] = _ffill(x["gmd_inv_GDP"], 2)
    b["export_share_change_3y"] = d3(_ffill(x["gmd_exports_GDP"], 2))
    b["house_price_change_3y"] = d3(np.log(_ffill(x["gmd_rHPI"], 2)))
    b["oil_price"] = x["oil_price"]
    b["oil_price_20y"] = np.stack([np.nanmean(x["oil_price"][:, max(0, j - 19):j + 1], 1) if np.isfinite(x["oil_price"][:, max(0, j - 19):j + 1]).any() else np.full(x["oil_price"].shape[0], np.nan)
                                   for j in range(x["oil_price"].shape[1])], 1)
    b["war"] = np.where(np.isfinite(conflict), np.nan_to_num(x["war"]), np.nan)
    b["interstate_recent"] = _decayed(x["interstate"], DECAY["interstate_recent"])
    # violence intensity per million people (UCDP 1989+)
    if "battle_deaths" in x:
        pm = pop / 1e6
        b["battle_deaths_pc"] = x["battle_deaths"] / pm
        s3 = lambda a: a + np.concatenate([np.full((a.shape[0], 1), np.nan), a[:, :-1]], 1) \
            + np.concatenate([np.full((a.shape[0], 2), np.nan), a[:, :-2]], 1)
        b["nonstate_3y_pc"] = s3(x["nonstate_deaths"]) / pm
        b["onesided_3y_pc"] = s3(x["onesided_deaths"]) / pm
        # UCDP death counts end a year before the origin: carry the last year forward (2026-10-26; the 2025 origin
        # had these three features missing for every country)
        for k in ("battle_deaths_pc", "nonstate_3y_pc", "onesided_3y_pc"):
            b[k] = _ffill(b[k], 1)
    else:
        for k in ("battle_deaths_pc", "nonstate_3y_pc", "onesided_3y_pc"):
            b[k] = np.full(pop.shape, np.nan)
    for k in ("landlocked", "dist_coast", "near_coast", "rugged", "desert", "tropical", "area_km2"):
        b[k] = x[k]
    b["arable_pc"] = _ffill(x["arable_pc"], 5)
    b["water_stress"] = _ffill(x["water_stress"], 10)
    b["pop_density"] = _ffill(x["pop_density"], 3)
    for k in ("energy_pc", "fossil_self_suff", "low_carbon_share"):
        b[k] = _ffill(x[k], 3)
    ca = x["coup_attempts"]
    b["coups_total"] = np.where(np.isfinite(ca).cumsum(1) > 0, np.cumsum(np.nan_to_num(ca) > 0, 1), np.nan)
    cf = conflict
    b["conflict_years"] = np.where(np.isfinite(cf).cumsum(1) > 0, np.cumsum(np.nan_to_num(cf), 1), np.nan)
    b["obs_years"] = np.where(np.isfinite(cf).cumsum(1) > 0, np.cumsum(np.isfinite(cf), 1), np.nan)
    # completed collapses: GDP per person >= 10% below its previous 5-year peak, counted once per episode
    lgx = np.log(x["gdp_pc"])
    peak5 = np.stack([np.nanmax(lgx[:, max(0, j - 5):j], 1) if j > 0 else np.full(lgx.shape[0], np.nan)
                      for j in range(lgx.shape[1])], 1) if True else None
    with np.errstate(invalid="ignore"):
        inc = (lgx - peak5 < np.log(0.9)).astype(float)
    start = np.concatenate([inc[:, :1], np.clip(inc[:, 1:] - inc[:, :-1], 0, 1)], 1)
    started = (np.isfinite(lgx).cumsum(1) > 0) & (p.years[None, :] >= 1950)
    b["collapses_total"] = np.where(started, np.cumsum(np.where(p.years[None, :] >= 1950, start, 0), 1), np.nan)
    b["pre1946"] = np.broadcast_to((p.years < 1946).astype(float), x["pop"].shape).copy()
    # leaders: tenure from V-Dem's year-end leader names, which records every turnover (1789-2025).
    # REIGN's attributes (age, military career, elected, months since an irregular exit or an
    # electoral loss) end in August 2021. After that they are carried forward only while V-Dem shows
    # the same leader in office; a successful coup marks an irregular, military takeover; any other
    # new leader's attributes are unknown (NaN).
    la, lm, le = x["leader_age"].copy(), x["leader_military"].copy(), x["leader_elected"].copy()
    irr, loss = x["months_since_irregular"].copy(), x["months_since_loss"].copy()
    lc = x["leader_change"]
    for t in range(1, la.shape[1]):
        if not (2021 < p.years[t] <= LAST_OBS_YEAR):
            continue
        same, changed = lc[:, t] == 0, lc[:, t] == 1
        coup = np.nan_to_num(x["coups"][:, t]) > 0
        la[:, t] = np.where(same, la[:, t - 1] + 1, np.nan)
        lm[:, t] = np.where(same, lm[:, t - 1], np.where(changed & coup, 1.0, np.nan))
        le[:, t] = np.where(same, le[:, t - 1], np.where(changed & coup, 0.0, np.nan))
        irr[:, t] = np.where(coup, np.log(6), np.log(np.exp(irr[:, t - 1]) + 12))
        electoral = changed & (np.nan_to_num(x["election"][:, t]) == 1)
        loss[:, t] = np.where(electoral, np.log(6), np.log(np.exp(loss[:, t - 1]) + 12))
    b["leader_log_tenure"] = np.log1p(12 * x["leader_years"])
    b["leader_change"] = x["leader_change"]
    b["leader_age"], b["leader_military"], b["leader_elected"] = la, lm, le
    b["months_since_irregular"], b["months_since_loss"] = irr, loss
    # REIGN ends in August 2021. After that, an election is anticipated when the country holds elections (one in
    # the past 15 years) and the next falls due at its usual interval (median of its last gaps), as in the simulator,
    # at the rate REIGN flags such years
    ea = _ffill(x["election_anticipated"], 0)
    last_reign = int(np.max(np.where(np.isfinite(x["election_anticipated"]).any(0))[0])) \
        if np.isfinite(x["election_anticipated"]).any() else ea.shape[1] - 1
    ey = np.nan_to_num(b["election_year"])
    for t in range(last_reign + 1, min(ea.shape[1], p.yi(LAST_OBS_YEAR) + 1)):
        for i in range(ea.shape[0]):
            yrs = np.where(ey[i, :t + 1] == 1)[0]
            if not np.isfinite(b["years_since_election"][i, t]):
                continue
            gap = np.clip(np.median(np.diff(yrs[-8:])) if len(yrs) >= 3 else 5.0, 2, 8)
            holds = len(yrs) > 0 and yrs[-1] >= t - 15
            # (REIGN flags an election within ~6 months: 30% of years before an election, 2% of others, 1990-2020)
            ea[i, t] = 0.3 if (holds and b["years_since_election"][i, t] + 1 >= gap) else 0.0
    b["election_anticipated"] = ea
    b["elections_delayed"] = _ffill(x["elections_delayed"], 4)       # (REIGN to 2021, carried to the 2025 origin)
    b["coup_attempts"] = x["coup_attempts"]
    b["coups_success"] = x["coups"]
    b["interstate"] = x["interstate"]
    b["fin_crisis"] = x["fin_crisis"]
    b["coup_recent"] = _decayed(x["coup_attempts"], DECAY["coup_recent"])
    b["fin_crisis_recent"] = _decayed(x["fin_crisis"], DECAY["fin_crisis_recent"])
    # years since last intrastate conflict; left-censored at UCDP start (1946) or state entry,
    # where the counter starts at 20
    ysc = np.full(conflict.shape, np.nan)
    run = np.full(conflict.shape[0], np.nan)
    for y in range(conflict.shape[1]):
        c = conflict[:, y]
        run = np.where(np.isfinite(c), np.where(c == 1, 0.0, np.where(np.isfinite(run), run + 1, 20.0)), np.nan)
        ysc[:, y] = run
    b["ysc"] = ysc
    return b


def panel_features(p: Panel, base: dict, device="cpu") -> np.ndarray:
    """X[C, Y, F] for every country-year (NaN before enough history)."""
    if SPATIAL["W"] is None or SPATIAL["W"].shape[0] != len(p.codes):
        set_spatial(p.codes)
    C, Y = len(p.codes), len(p.years)
    regions = torch.as_tensor(p.regions, device=device)
    X = np.full((C, Y, F), np.nan, dtype=np.float32)
    tb = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in base.items()}
    for t in range(W - 1, Y):
        h = {k: tb[k][None, :, t - W + 1:t + 1] for k in WINDOW_VARS}
        cur = {k: tb[k][None, :, t] for k in CURRENT_VARS}
        X[:, t] = compute_features(h, cur, regions, len(p.region_names))[0].cpu().numpy()
    return X


# ---------------------------------------------------------------------------
# Outcome labels ("Seldon crises")

TASKS = ["civil_conflict_5y", "econ_collapse_5y", "autocratization_5y", "coup_5y", "onset_1y", "termination_1y",
         "coup_1y"]


def labels(p: Panel, base: dict, horizon: int = 5):
    """Returns Y[C, Yrs, T] in {0,1,nan}. nan = not at risk or unobserved."""
    conf, lg, dem, coup = base["conflict"], base["lg"], base["dem"], base["coup_attempts"]
    C, Yn = conf.shape
    last = p.yi(LAST_OBS_YEAR)
    out = np.full((C, Yn, len(TASKS)), np.nan, dtype=np.float32)
    onset = np.zeros_like(conf)
    onset[:, 1:] = (conf[:, 1:] == 1) & (conf[:, :-1] == 0)
    for t in range(Yn):
        if t + horizon <= last:
            fut = slice(t + 1, t + horizon + 1)
            # an event seen in any observed future year counts even if later years are missing
            # (state collapse often ends reporting); a non-event needs the full window observed
            def put(k, at_risk, event, observed):
                y = np.where(event, 1.0, np.where(observed, 0.0, np.nan))
                out[at_risk, t, k] = y[at_risk]
            with np.errstate(invalid="ignore"):
                fin = lambda a: np.isfinite(a[:, fut]).all(1)
                put(0, conf[:, t] == 0, (onset[:, fut] == 1).any(1), fin(conf))
                put(1, np.isfinite(lg[:, t]), (lg[:, fut] - lg[:, [t]] < np.log(0.9)).any(1), fin(lg))  # >=10% contraction
                # autocratization needs room to fall: a 0.10 drop is impossible below 0.10
                put(2, dem[:, t] >= 0.10, (dem[:, fut] - dem[:, [t]] <= -0.10).any(1), fin(dem))
                put(3, np.ones(C, bool), (coup[:, fut] > 0).any(1), fin(coup))
        if t + 1 <= last:
            ok = (conf[:, t] == 0) & np.isfinite(conf[:, t + 1])
            out[ok, t, 4] = conf[ok, t + 1]
            ok = (conf[:, t] == 1) & np.isfinite(conf[:, t + 1])
            out[ok, t, 5] = 1 - conf[ok, t + 1]
            ok = np.isfinite(coup[:, t + 1])
            out[ok, t, 6] = (coup[ok, t + 1] > 0).astype(float)
    return out
