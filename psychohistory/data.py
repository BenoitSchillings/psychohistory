"""Data acquisition: downloads long-run country-year series and builds a wide panel.

Every series ends up as an array [n_countries, n_years] over YEARS, aligned on
OWID ISO3 codes. Projections (UN WPP medium variant) are merged into the same
array for years after the last estimate.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OWID = "https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=true"
UCDP_URL = "https://ucdp.uu.se/downloads/ucdpprio/ucdp-prio-acd-261-csv.zip"
UCDP_FILE = "ucdp-26.1.zip"
VDEM_URL = "https://github.com/vdeminstitute/vdemdata/raw/master/data/vdem.RData"
GMD_URL = "https://www.globalmacrodata.com/GMD.csv"
EPR_URL = "https://icr.ethz.ch/data/epr/core/EPR-2021.csv"
WRP_URL = "https://correlatesofwar.org/wp-content/uploads/WRP_national.csv"
REIGN_URL = "https://raw.githubusercontent.com/OEFDataScience/REIGN.github.io/gh-pages/data_sets/REIGN_2021_8.csv"

# V-Dem indicators beyond the democracy index: ideology, religion, coercion, state capacity, coups
VDEM_VARS = {
    "legit_ideology": "v2exl_legitideol",          # regime legitimated by an ideology
    "legit_nationalist": "v2exl_legitideolcr_0",
    "legit_socialist": "v2exl_legitideolcr_1",
    "legit_restorative": "v2exl_legitideolcr_2",   # conservative / restorative
    "legit_separatist": "v2exl_legitideolcr_3",
    "legit_religious": "v2exl_legitideolcr_4",
    "legit_performance": "v2exl_legitperf",
    "polarization": "v2cacamps",                   # political polarization
    "mass_mobilization": "v2cagenmob",
    "religious_freedom": "v2clrelig",
    "religious_repression": "v2csrlgrep",          # repression of religious organisations
    "fiscal_capacity": "v2stfisccap",
    "political_killings": "v2clkill",
    "physical_integrity": "v2x_clphy",
    "neopatrimonialism": "v2x_neopat",
    "rule_of_law": "v2x_rule",
    "power_by_wealth": "v2pepwrses",               # political power distributed by socio-economic position
    "coups": "e_pt_coup",                          # Powell & Thyne successful coups
    "coup_attempts": "e_pt_coup_attempts",
    # institutions: constraints, accountability, information, coercion, regime composition
    "jud_constraints": "v2x_jucon", "leg_constraints": "v2xlg_legcon", "horiz_acc": "v2x_horacc",
    "diag_acc": "v2x_diagacc", "free_expression": "v2x_freexp_altinf", "media_censorship": "v2mecenefm",
    "cso_repression": "v2csreprss", "mob_democracy": "v2cademmob", "mob_autocracy": "v2caautmob",
    "regime_military": "v2x_ex_military", "regime_party": "v2x_ex_party", "regime_hereditary": "v2x_ex_hereditary",
    "regime_confidence": "v2x_ex_confidence", "party_inst": "v2xps_party", "free_fair_elections": "v2xel_frefair",
    "elected_regime": "v2x_elecreg", "exec_respects_const": "v2exrescon", "support_group_size": "v2regsupgroupssize",
    "opposition_group_size": "v2regoppgroupssize", "regime_duration": "v2regdur", "liberal_dem": "v2x_libdem",
    "civil_liberties": "v2x_civlib", "cso_participation": "v2x_cspart", "clientelism": "v2xnp_client",
    "torture": "v2cltort", "political_violence": "v2caviol", "admin_impartial": "v2stcritrecadm",
    "territorial_control": "v2svstterr",
}
VDEM_ELECTIONS = [f"v2eltype_{i}" for i in range(10)]

FIRST_YEAR, LAST_OBS_YEAR, LAST_YEAR = 1850, 2025, 2100
YEARS = np.arange(FIRST_YEAR, LAST_YEAR + 1)

# name -> (OWID slug, column-name substring for estimates, optional projection substring)
SERIES: dict[str, tuple[str, str, str | None]] = {
    # --- economy / immiseration
    "gdp_pc": ("gdp-per-capita-maddison-project-database", "gdp_per_capita", None),
    "gdp_pc_wb": ("gdp-per-capita-worldbank", "ny_gdp_pcap_pp_kd", None),
    "inflation": ("inflation-of-consumer-prices", "fp_cpi_totl_zg", None),
    "unemployment": ("unemployment-rate", "sl_uem_totl_zs", None),
    "extreme_poverty": ("share-of-population-in-extreme-poverty", "headcount_ratio", None),
    "calories": ("daily-per-capita-caloric-supply", "daily_calories", None),
    "cereal_yield": ("cereal-yield", "yield", None),
    "trade": ("trade-as-share-of-gdp", "ne_trd_gnfs_zs", None),
    "oil_twh": ("oil-production-by-country", "oil_production_twh", None),
    # --- demography
    "pop": ("population-with-un-projections", "variant_estimates", "variant_medium__projected"),
    "pop_hist": ("population", "population_historical", None),   # long-run (HYDE/Gapminder/UN) for pre-1950
    "median_age": ("median-age", "variant_estimates", "variant_medium__projected"),
    "fertility": ("fertility-rate-with-projections", "variant_estimates", "variant_medium__projected"),
    "pop_0_14": ("population-young-working-elderly-with-projections", "age_0_14__variant_estimates", "age_0_14__variant_medium__projected"),
    "urban": ("share-of-population-urban", "share__area_type_urban", None),
    "child_mort": ("child-mortality", "child_mortality_rate", None),
    "life_exp": ("life-expectancy", "life_expectancy_0", None),
    # --- elites / inequality (Turchin: elite overproduction, wealth pump)
    "tertiary_enrol": ("gross-enrollment-ratio-in-tertiary-education", "gross_enrolment_ratio", None),
    "tertiary_share": ("share-of-the-population-with-completed-tertiary-education", "percentage_of_tertiary", None),
    "schooling": ("mean-years-of-schooling-long-run", "average_years_of_education", None),
    "top1": ("income-share-top-1-before-tax-wid", "richest_1pct", None),
    "gini": ("economic-inequality-gini-index", "gini", None),
    # --- state, fiscal, regime
    "gov_exp": ("total-gov-expenditure-gdp-wdi", "gc_xpn_totl_gd_zs", None),
    "tax_rev": ("tax-revenues-as-a-share-of-gdp-unu-wider", "tax_inc_sc", None),
    "military": ("military-expenditure-share-gdp", "share_gdp", None),
    "democracy": ("electoral-democracy-index", "electdem_vdem__estimate_best", None),
    "civ_libs": ("human-rights-index-vdem", "civ_libs_vdem__estimate_best", None),
    "corruption": ("political-corruption-index", "corruption_vdem__estimate_best", None),
    "regime": ("political-regime", "regime_row_owid", None),
    # --- violence, displacement, climate, technology
    "homicide": ("homicide-rate-unodc", "rate_per_100_000", None),
    "refugees": ("refugee-population-by-country-or-territory-of-origin", "refugees", None),
    "temp_anom": ("annual-temperature-anomalies", "temperature_anomaly", None),
    # resources
    "arable_pc": ("arable-land-use-per-person", "ag_lnd_arbl_ha_pc", None),
    "water_stress": ("freshwater-withdrawals-as-a-share-of-internal-resources", "er_h2o_stress", None),
    "pop_density": ("population-density", "population_density", None),
    "warheads": ("nuclear-warhead-stockpiles", "number_of_warheads", None),
    "internet": ("share-of-individuals-using-the-internet", "it_net_user_zs", None),
    "mobile": ("mobile-cellular-subscriptions-per-100-people", "it_cel_sets_p2", None),
}

EPR_NAME_FIX = {
    "Belarus (Byelorussia)": "Belarus", "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Burkina Faso (Upper Volta)": "Burkina Faso", "Cambodia (Kampuchea)": "Cambodia",
    "Congo, Democratic Republic of (Zaire)": "Democratic Republic of Congo", "Cote D'Ivoire": "Cote d'Ivoire",
    "Czech Republic": "Czechia", "German Federal Republic": "Germany", "Iran (Persia)": "Iran",
    "Italy/Sardinia": "Italy", "Korea, People's Republic of": "North Korea", "Korea, Republic of": "South Korea",
    "Kyrgyz Republic": "Kyrgyzstan", "Macedonia (FYROM/North Macedonia)": "North Macedonia",
    "Madagascar (Malagasy)": "Madagascar", "Myanmar (Burma)": "Myanmar", "Rumania": "Romania",
    "Russia (Soviet Union)": "Russia", "Sri Lanka (Ceylon)": "Sri Lanka", "Surinam": "Suriname",
    "Swaziland (Eswatini)": "Eswatini", "Tanzania (Tanganyika)": "Tanzania", "Turkey (Ottoman Empire)": "Turkey",
    "United States of America": "United States", "Vietnam, Democratic Republic of": "Vietnam",
    "Yemen (Arab Republic of Yemen)": "Yemen", "Yugoslavia": "Serbia", "Zimbabwe (Rhodesia)": "Zimbabwe",
}
REIGN_NAME_FIX = {
    "Bosnia": "Bosnia and Herzegovina", "Cen African Rep": "Central African Republic", "Congo-Brz": "Congo",
    "Congo/Zaire": "Democratic Republic of Congo", "Czech Rep": "Czechia", "Dominican Rep": "Dominican Republic",
    "Guinea Bissau": "Guinea-Bissau", "Ivory Coast": "Cote d'Ivoire", "Korea North": "North Korea",
    "Korea South": "South Korea", "Macedonia": "North Macedonia", "Soviet Union": "Russia", "Swaziland": "Eswatini",
    "UKG": "United Kingdom", "USA": "United States", "Yugoslavia": "Serbia", "St Lucia": "Saint Lucia",
}
UCDP_NAME_FIX = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina", "Cambodia (Kampuchea)": "Cambodia",
    "DR Congo (Zaire)": "Democratic Republic of Congo", "Hyderabad": "India",
    "Ivory Coast": "Cote d'Ivoire", "Madagascar (Malagasy)": "Madagascar",
    "Myanmar (Burma)": "Myanmar", "Russia (Soviet Union)": "Russia",
    "Serbia (Yugoslavia)": "Serbia", "South Vietnam": "Vietnam", "South Yemen": "Yemen",
    "United States of America": "United States", "Vietnam (North Vietnam)": "Vietnam",
    "Yemen (North Yemen)": "Yemen", "Zimbabwe (Rhodesia)": "Zimbabwe",
}


def _fetch(url: str, dest: Path, refresh: bool = False) -> Path:
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def _pick(cols, sub):
    matches = [c for c in cols if sub in c and not c.endswith("__annotations")]
    if not matches:
        raise KeyError(f"no column containing {sub!r} in {list(cols)}")
    return matches[0]


def load_series(name: str, refresh: bool = False) -> pd.DataFrame:
    slug, est, proj = SERIES[name]
    df = pd.read_csv(_fetch(OWID.format(slug=slug), RAW / f"{slug}.csv", refresh))
    df = df[df.code.notna() & ~df.code.astype(str).str.startswith("OWID")]
    val = pd.to_numeric(df[_pick(df.columns, est)], errors="coerce")
    if proj:
        val = val.fillna(pd.to_numeric(df[_pick(df.columns, proj)], errors="coerce"))
    return pd.DataFrame({"code": df.code, "year": df.year, "value": val}).dropna()


def load_ucdp(refresh: bool = False) -> pd.DataFrame:
    path = _fetch(UCDP_URL, RAW / UCDP_FILE, refresh)
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        return pd.read_csv(io.BytesIO(z.read(name)))


@dataclass
class Panel:
    codes: list[str]
    names: list[str]
    regions: np.ndarray            # [C] int region index
    region_names: list[str]
    years: np.ndarray
    x: dict[str, np.ndarray] = field(default_factory=dict)   # name -> [C, Y]

    def yi(self, year: int) -> int:
        return int(year - self.years[0])

    def save(self, path: Path):
        np.savez_compressed(path, codes=self.codes, names=self.names, regions=self.regions,
                            region_names=self.region_names, years=self.years,
                            **{f"x_{k}": v for k, v in self.x.items()})

    @staticmethod
    def load(path: Path) -> "Panel":
        z = np.load(path, allow_pickle=False)
        x = {k[2:]: z[k] for k in z.files if k.startswith("x_")}
        return Panel(list(z["codes"]), list(z["names"]), z["regions"], list(z["region_names"]), z["years"], x)


def _to_wide(df: pd.DataFrame, codes: list[str]) -> np.ndarray:
    df = df[df.year.between(FIRST_YEAR, LAST_YEAR) & df.code.isin(codes)]
    piv = df.pivot_table(index="code", columns="year", values="value", aggfunc="mean")
    piv = piv.reindex(index=codes, columns=YEARS)
    return piv.to_numpy(dtype=np.float64)


# How a country-year counts as in civil conflict. "location" (to February 2027): every country in UCDP's location field.
# "deaths" (third debate, 9 November 2026; under test): from 1989, where UCDP GED records the conflict's deaths, a listed
# country keeps it only if at least 25 of that conflict-year's deaths, or half of them, fall on its soil (the United
# States was coded at war 2001-2019 for the conflict with al-Qaida, and troop contributors for wars abroad).
# Conflict-years without GED events keep the location coding; before 1989 nothing changes.
CONFLICT_CODING = "deaths"     # (version 55, 9 November 2026: scripts/conflict_coding_test.py)


def _ged_located():
    """{(conflict_id, year): {iso3: deaths}} from UCDP GED (state-based violence), country of the event"""
    from .monthly import gw_map
    gw2iso, _ = gw_map()
    f = RAW / "outcomes" / "ucdp" / "ged261-csv" / "GEDEvent_v26_1.csv"
    e = pd.read_csv(f, usecols=["type_of_violence", "conflict_new_id", "year", "country_id", "best"])
    e = e[e.type_of_violence == 1]
    e["iso3"] = e.country_id.map(gw2iso)
    g = e.dropna(subset=["iso3"]).groupby(["conflict_new_id", "year", "iso3"]).best.sum()
    out = {}
    for (cid, y, iso), d in g.items():
        out.setdefault((int(cid), int(y)), {})[iso] = float(d)
    return out


def _conflict_arrays(ucdp: pd.DataFrame, name_to_code: dict[str, str], codes: list[str], coding=None):
    """Intrastate conflict (UCDP types 3,4) active / war-level, and interstate involvement."""
    coding = coding or CONFLICT_CODING
    ged = _ged_located() if coding == "deaths" else None
    idx = {c: i for i, c in enumerate(codes)}
    C, Y = len(codes), len(YEARS)
    intra, war, inter = (np.zeros((C, Y)) for _ in range(3))
    for row in ucdp.itertuples():
        yi = row.year - FIRST_YEAR
        if not 0 <= yi < Y:
            continue
        for loc in str(row.location).split(","):
            loc = UCDP_NAME_FIX.get(loc.strip(), loc.strip())
            ci = idx.get(name_to_code.get(loc, ""), None)
            if ci is None:
                continue
            if row.type_of_conflict in (3, 4):
                if ged is not None and row.year >= 1989:
                    dd = ged.get((int(row.conflict_id), int(row.year)))
                    if dd:
                        mine = dd.get(codes[ci], 0.0)
                        if mine < 25 and mine < 0.5 * sum(dd.values()):
                            continue
                intra[ci, yi] = 1
                if row.intensity_level == 2:
                    war[ci, yi] = 1
            elif row.type_of_conflict == 2:
                inter[ci, yi] = 1
    # outside UCDP coverage the state is unknown
    for a in (intra, war, inter):
        a[:, YEARS < 1946] = np.nan
        a[:, YEARS > ucdp.year.max()] = np.nan
    return intra, war, inter


def _yearly(df, codes, value):
    """long (code, year, value) -> wide [C, Y]."""
    return _to_wide(df.rename(columns={value: "value"})[["code", "year", "value"]].dropna(), codes)


def load_vdem_extra(codes, refresh=False):
    import pyreadr
    path = _fetch(VDEM_URL, RAW / "vdem.RData", refresh)
    pq = RAW / "vdem_extra_v3.parquet"
    leader_cols = ["v2exnamhos", "v2exnamhog", "v2ex_hosw"]
    if not pq.exists() or refresh:
        d = pyreadr.read_r(path)["vdem"]
        d = d[["country_text_id", "year"] + list(VDEM_VARS.values()) + VDEM_ELECTIONS + leader_cols].rename(
            columns={"country_text_id": "code"})
        d.to_parquet(pq)
    d = pd.read_parquet(pq)
    d["year"] = d.year.astype(int)
    d["election"] = (d[VDEM_ELECTIONS] == 1).any(axis=1).astype(float)   # any national election that year
    out = {k: _yearly(d, codes, v) for k, v in VDEM_VARS.items()}
    out["election"] = _yearly(d, codes, "election")
    # effective leader: the head of state if that is the more powerful office (v2ex_hosw >= 0.5),
    # otherwise the head of government. Tenure = consecutive year-ends with the same name.
    d = d.sort_values(["code", "year"])
    use_hog = (d.v2ex_hosw < 0.5) & d.v2exnamhog.notna() & (d.v2exnamhog.astype(str).str.len() > 0)
    d["leader"] = np.where(use_hog, d.v2exnamhog, d.v2exnamhos)
    years_in, change = [], []
    prev_code, prev_name, run = None, None, 0
    for code, name in zip(d.code, d.leader):
        valid = isinstance(name, str) and len(name) > 0
        if code != prev_code or not valid:
            run, ch = (1 if valid else 0), np.nan
        elif name == prev_name:
            run, ch = run + 1, 0.0
        else:
            run, ch = 1, 1.0
        years_in.append(run if valid else np.nan)
        change.append(ch)
        prev_code, prev_name = code, (name if valid else None)
    d["leader_years"], d["leader_change"] = years_in, change
    out["leader_years"] = _yearly(d, codes, "leader_years")
    out["leader_change"] = _yearly(d, codes, "leader_change")
    return out


def load_gmd(codes, refresh=False):
    """Global Macro Database: public debt and financial crises."""
    macro = ["USDfx", "REER", "CA_GDP", "govdef_GDP", "gen_govdef_GDP", "ltrate", "strate", "M2", "nGDP", "inv_GDP",
             "exports_GDP", "rHPI", "infl"]
    g = pd.read_csv(_fetch(GMD_URL, RAW / "GMD.csv", refresh),
                    usecols=["ISO3", "year", "govdebt_GDP", "gen_govdebt_GDP", "BankingCrisis", "SovDebtCrisis", "CurrencyCrisis"] + macro)
    g = g.rename(columns={"ISO3": "code"})
    g["debt"] = g.gen_govdebt_GDP.fillna(g.govdebt_GDP)
    crisis = g[["BankingCrisis", "SovDebtCrisis", "CurrencyCrisis"]]
    g["fin_crisis"] = np.where(crisis.notna().any(axis=1), crisis.fillna(0).max(axis=1).clip(0, 1), np.nan)
    g = g[g.year <= LAST_OBS_YEAR]
    g["deficit"] = g.gen_govdef_GDP.fillna(g.govdef_GDP)
    out = {"public_debt": _yearly(g, codes, "debt"), "fin_crisis": _yearly(g, codes, "fin_crisis"),
           "gov_deficit": _yearly(g, codes, "deficit")}
    for k in ["USDfx", "REER", "CA_GDP", "ltrate", "strate", "M2", "nGDP", "inv_GDP", "exports_GDP", "rHPI", "infl"]:
        out[f"gmd_{k}"] = _yearly(g, codes, k)
    return out


def load_epr(codes, name_to_code, refresh=False):
    """Ethnic Power Relations: share of population in groups excluded from power."""
    e = pd.read_csv(_fetch(EPR_URL, RAW / "EPR-2021.csv", refresh))
    e["code"] = e.statename.map(lambda n: name_to_code.get(EPR_NAME_FIX.get(n, n)))
    e = e.dropna(subset=["code"])
    rows = []
    for code, y0, y1, size, status in zip(e.code, e["from"], e["to"], e["size"], e.status):
        for y in range(max(y0, FIRST_YEAR), min(y1, LAST_OBS_YEAR) + 1):
            rows.append((code, y, size, status))
    d = pd.DataFrame(rows, columns=["code", "year", "size", "status"])
    excl = d.status.isin(["POWERLESS", "DISCRIMINATED", "SELF-EXCLUSION"])
    d["excluded"] = d["size"] * excl
    d["discriminated"] = d["size"] * (d.status == "DISCRIMINATED")
    agg = d.groupby(["code", "year"]).agg(excluded=("excluded", "sum"), discriminated=("discriminated", "sum"),
                                          n_groups=("size", "size")).reset_index()
    return {k: _yearly(agg, codes, k) for k in ["excluded", "discriminated", "n_groups"]}


RELIGIONS = ["chrstprot", "chrstcat", "chrstorth", "chrstang", "chrstothr", "judgen", "islmsun", "islmshi",
             "budgen", "zorogen", "hindgen", "sikhgen", "shntgen", "bahgen", "taogen", "jaingen", "confgen",
             "syncgen", "anmgen", "nonrelig", "othrgen"]


def load_religion(codes, name_to_code, refresh=False):
    """COW World Religion Project: religious fractionalisation and polarisation (Reynal-Querol)."""
    w = pd.read_csv(_fetch(WRP_URL, RAW / "WRP_national.csv", refresh))
    rg = pd.read_csv(_fetch(REIGN_URL, RAW / "REIGN_2021_8.csv", refresh), usecols=["ccode", "country"]).drop_duplicates("ccode")
    cc = {r.ccode: name_to_code.get(REIGN_NAME_FIX.get(r.country, r.country)) for r in rg.itertuples()}
    cc[713] = "TWN"
    w["code"] = w.state.map(cc)
    w = w.dropna(subset=["code"])
    w["islmothr_all"] = w[["islmibd", "islmnat", "islmalw", "islmahm", "islmothr"]].sum(1)
    cols = RELIGIONS + ["islmothr_all"]
    s = w[cols].div(w[cols].sum(1).replace(0, np.nan), axis=0)
    w["rel_fractionalization"] = 1 - (s ** 2).sum(1)
    w["rel_polarization"] = 1 - (((0.5 - s) / 0.5) ** 2 * s).sum(1)
    w["share_muslim"] = s[["islmsun", "islmshi", "islmothr_all"]].sum(1)
    w["share_nonreligious"] = s["nonrelig"]
    out = {}
    for k in ["rel_fractionalization", "rel_polarization", "share_muslim", "share_nonreligious"]:
        out[k] = _yearly(w, codes, k)   # 5-yearly survey; held forward causally in features.base_arrays
    return out


def load_reign(codes, name_to_code, refresh=False):
    """REIGN (OEF Research) leader characteristics, 1950-2021, value at each year's last month."""
    r = pd.read_csv(_fetch(REIGN_URL, RAW / "REIGN_2021_8.csv", refresh), low_memory=False)
    r["code"] = r.country.map(lambda n: name_to_code.get(REIGN_NAME_FIX.get(n, n)))
    r = r.dropna(subset=["code"]).sort_values(["code", "year", "month"]).groupby(["code", "year"]).tail(1)
    r["year"] = r.year.astype(int)
    cols = {"leader_tenure": "tenure_months", "leader_age": "age", "leader_military": "militarycareer",
            "leader_elected": "elected", "election_anticipated": "anticipation", "elections_delayed": "delayed",
            "months_since_irregular": "irregular", "months_since_loss": "loss"}
    return {k: _yearly(r.rename(columns={v: "_v"}).assign(_v=pd.to_numeric(r[v], errors="coerce")), codes, "_v")
            for k, v in cols.items()}


def load_ucdp_violence(codes, name_to_code):
    """UCDP 25.1 (1989-2024): battle deaths in state-based conflicts, deaths in non-state conflicts
    and in one-sided violence against civilians, by location. Zero in covered years without events."""
    ci = {c: i for i, c in enumerate(codes)}
    out = {}
    specs = {"battle_deaths": ("BattleDeaths_v25_1_conf.csv", "location_inc", "bd_best"),
             "nonstate_deaths": ("NonState_v25_1.csv", "location", "best_fatality_estimate"),
             "onesided_deaths": ("OneSided_v25_1.csv", "location", "best_fatality_estimate")}
    for name, (fn, loc_col, val) in specs.items():
        df = pd.read_csv(RAW / "ucdp" / fn)
        a = np.full((len(codes), len(YEARS)), np.nan)
        a[:, (YEARS >= 1989) & (YEARS <= 2024)] = 0.0
        for loc, y, v in zip(df[loc_col], df.year, df[val]):
            locs = [l.strip() for l in str(loc).split(",")]
            for l in locs:
                c = name_to_code.get(UCDP_NAME_FIX.get(l, l))
                if c in ci and FIRST_YEAR <= y <= LAST_OBS_YEAR:
                    a[ci[c], y - FIRST_YEAR] += float(v) / len(locs)
        out[name] = a
    return out


SYS = RAW / "system"


def load_geography(codes):
    """static geography: CEPII (landlocked, area) and Nunn & Puga (2012) ruggedness data
    (distance to coast, share of land near the coast, ruggedness, desert and tropical shares)."""
    g = pd.read_excel(SYS / "geo_cepii.xls").drop_duplicates("iso3").set_index("iso3")
    n = pd.read_csv(SYS / "nunn" / "rugged_data.csv", keep_default_na=False, na_values=[""], encoding="latin-1").set_index("isocode")
    col = lambda s: np.array([s.get(c, np.nan) for c in codes], float)
    static = {"landlocked": col(g.landlocked), "area_km2": col(g.area), "dist_coast": col(n.dist_coast),
              "near_coast": col(n.near_coast), "rugged": col(n.rugged), "desert": col(n.desert),
              "tropical": col(n.tropical)}
    return {k: np.tile(v[:, None], (1, len(YEARS))) for k, v in static.items()}


def load_energy(codes):
    """OWID energy dataset: energy use per person, fossil self-sufficiency, low-carbon share."""
    e = pd.read_csv(SYS / "owid-energy-data.csv").rename(columns={"iso_code": "code"})
    e = e[e.code.isin(codes)]
    prod = e[["coal_production", "gas_production", "oil_production"]].sum(axis=1, min_count=1)
    e["fossil_self_suff"] = prod / e.fossil_fuel_consumption.replace(0, np.nan)
    return {"energy_pc": _yearly(e, codes, "energy_per_capita"),
            "fossil_self_suff": _yearly(e, codes, "fossil_self_suff"),
            "low_carbon_share": _yearly(e, codes, "low_carbon_share_energy")}


def build_panel(refresh: bool = False, verbose: bool = True) -> Panel:
    # country universe: V-Dem countries with UN population data
    dem_raw = pd.read_csv(_fetch(OWID.format(slug=SERIES["democracy"][0]), RAW / f"{SERIES['democracy'][0]}.csv", refresh))
    dem_raw = dem_raw[dem_raw.code.notna() & ~dem_raw.code.astype(str).str.startswith("OWID")]
    pop = load_series("pop", refresh)
    # microstates (< 100k people, e.g. the Vatican) behave nothing like countries in the training data
    pop_now = pop[pop.year == LAST_OBS_YEAR].set_index("code").value
    codes = sorted(c for c in set(dem_raw.code) & set(pop.code) if pop_now.get(c, 0) >= 1e5)
    names_df = pd.read_csv(RAW / f"{SERIES['pop'][0]}.csv")[["entity", "code"]].drop_duplicates()
    names_df = names_df[names_df.code.isin(codes)]
    code_to_name = dict(zip(names_df.code, names_df.entity))
    name_to_code = {v: k for k, v in code_to_name.items()}
    reg = dem_raw.dropna(subset=["owid_region"]).groupby("code").owid_region.last()
    region_names = sorted(reg.unique())
    regions = np.array([region_names.index(reg.get(c, "Asia")) for c in codes])

    panel = Panel(codes, [code_to_name.get(c, c) for c in codes], regions, region_names, YEARS)
    for name in SERIES:
        try:
            panel.x[name] = _to_wide(load_series(name, refresh), codes)
            if verbose:
                cov = np.isfinite(panel.x[name][:, panel.yi(1960):panel.yi(LAST_OBS_YEAR) + 1]).mean()
                print(f"  {name:16s} coverage 1960-{LAST_OBS_YEAR}: {cov:5.1%}")
        except Exception as e:  # keep going: one dead link should not kill the model
            print(f"  !! {name}: {e}")

    # real oil price (OWID/BP nominal, deflated by US CPI-U), a global series broadcast to all countries
    try:
        oil = pd.read_csv(_fetch(OWID.format(slug="crude-oil-prices"), RAW / "crude-oil-prices.csv", refresh))
        oil = oil[oil.entity == "World"].set_index("year").iloc[:, -1]
        cu = pd.read_csv(RAW / "us" / "cu.data.1.AllItems", sep=r"\s+", dtype=str)
        cu = cu[(cu.series_id.str.strip() == "CUUR0000SA0") & cu.period.str.startswith("M") & (cu.period != "M13")]
        cpi = cu.assign(year=cu.year.astype(int), v=pd.to_numeric(cu.value, errors="coerce")).groupby("year").v.mean()
        real = np.log(oil / cpi.reindex(oil.index)).dropna()
        arr = np.full(len(YEARS), np.nan)
        for y, v in real.items():
            if FIRST_YEAR <= y <= LAST_OBS_YEAR:
                arr[y - FIRST_YEAR] = v
        panel.x["oil_price"] = np.tile(arr, (len(codes), 1))
    except Exception as e:
        print(f"  !! oil price failed: {e!r}")

    # population before 1950 from the long-run series
    if "pop_hist" in panel.x:
        ph = panel.x.pop("pop_hist")
        early = panel.years < 1950
        panel.x["pop"][:, early] = np.where(np.isnan(panel.x["pop"][:, early]), ph[:, early], panel.x["pop"][:, early])
    # Maddison ends ~2022: chain-extend with World Bank PPP growth rates
    g, wb = panel.x["gdp_pc"], panel.x.pop("gdp_pc_wb")
    for c in range(len(codes)):
        last = np.where(np.isfinite(g[c]))[0]
        if len(last) == 0:
            continue
        for yi in range(last[-1] + 1, panel.yi(LAST_OBS_YEAR) + 1):
            if np.isfinite(wb[c, yi]) and np.isfinite(wb[c, yi - 1]) and wb[c, yi - 1] > 0:
                g[c, yi] = g[c, yi - 1] * wb[c, yi] / wb[c, yi - 1]
    # projections only exist for demography; everything else stops at LAST_OBS_YEAR
    for k, v in panel.x.items():
        if SERIES.get(k, (None, None, None))[2] is None:
            v[:, panel.yi(LAST_OBS_YEAR) + 1:] = np.nan

    for loader in (lambda: load_vdem_extra(codes, refresh), lambda: load_gmd(codes, refresh),
                   lambda: load_reign(codes, name_to_code, refresh),
                   lambda: load_ucdp_violence(codes, name_to_code),
                   lambda: load_geography(codes), lambda: load_energy(codes),
                   lambda: load_epr(codes, name_to_code, refresh), lambda: load_religion(codes, name_to_code, refresh)):
        try:
            for k, v in loader().items():
                panel.x[k] = v
                if verbose:
                    cov = np.isfinite(v[:, panel.yi(1960):panel.yi(LAST_OBS_YEAR) + 1]).mean()
                    print(f"  {k:22s} coverage 1960-{LAST_OBS_YEAR}: {cov:5.1%}")
        except Exception as e:
            print(f"  !! extra source failed: {e!r}")

    # observed climate (ERA5 via the World Bank CCKP); mean temperature is extended to
    # LAST_OBS_YEAR with the year-on-year change of the Copernicus anomaly series
    from . import climate
    try:
        obs = climate.observed(codes, YEARS, refresh)
        anom = panel.x["temp_anom"]
        tas = obs["tas"]
        for yi in range(panel.yi(2023), panel.yi(LAST_OBS_YEAR) + 1):
            tas[:, yi] = np.where(np.isnan(tas[:, yi]), tas[:, yi - 1] + anom[:, yi] - anom[:, yi - 1], tas[:, yi])
        for k, v in obs.items():
            panel.x[f"c_{k}"] = v
    except Exception as e:
        print(f"  !! climate failed: {e!r}")

    intra, war, inter = _conflict_arrays(load_ucdp(refresh), name_to_code, codes)
    panel.x["conflict"], panel.x["war"], panel.x["interstate"] = intra, war, inter
    _update_to_last_year(panel, codes)
    return panel


def _update_to_last_year(panel, codes):
    """Bring the V-Dem-based series, GDP and coups up to LAST_OBS_YEAR from the primary releases, which can be a
    year ahead of the Our World in Data mirrors: V-Dem v16 (covers 2025), World Bank / IMF GDP growth for 2025,
    and Powell & Thyne coups as carried by V-Dem (2025 rows exist only where a coup happened)."""
    d = _vdem_cols(["v2x_polyarchy", "v2x_civlib", "v2x_corr", "v2x_regime"])
    for name, col in [("democracy", "v2x_polyarchy"), ("civ_libs", "v2x_civlib"), ("corruption", "v2x_corr"),
                      ("regime", "v2x_regime")]:
        v = _yearly(d, codes, col)
        if name in panel.x:
            panel.x[name] = np.where(np.isfinite(v), v, panel.x[name])
    t = panel.yi(LAST_OBS_YEAR)
    g = panel.x["gdp_pc"]
    gr = RAW / "outcomes" / "gdp_2025_cy.csv"
    if gr.exists():
        q = pd.read_csv(gr)
        q = q[(q.year == LAST_OBS_YEAR) & (q.imf_is_projection == 0)].set_index("iso3").gdp_pc_growth_pct
        for i, c in enumerate(codes):
            if np.isnan(g[i, t]) and np.isfinite(g[i, t - 1]) and c in q.index and np.isfinite(q[c]):
                g[i, t] = g[i, t - 1] * (1 + q[c] / 100)
    ca = panel.x["coup_attempts"]
    cp = _vdem_cols(["e_pt_coup_attempts"])
    cp = cp[cp.year == LAST_OBS_YEAR].dropna()
    ca[:, t] = np.where(np.isfinite(ca[:, t - 1]) & np.isnan(ca[:, t]), 0.0, ca[:, t])
    cs = panel.x["coups"]
    cs[:, t] = np.where(np.isfinite(cs[:, t - 1]) & np.isnan(cs[:, t]), 0.0, cs[:, t])
    cp2 = _vdem_cols(["e_pt_coup_attempts", "e_pt_coup"])
    cp2 = cp2[cp2.year == LAST_OBS_YEAR].dropna()
    for c, v, w in zip(cp2.code, cp2.e_pt_coup_attempts, cp2.e_pt_coup):
        if c in codes:
            ca[codes.index(c), t] = v
            cs[codes.index(c), t] = w


def get_panel(refresh: bool = False) -> Panel:
    cache = ROOT / "data" / "panel.npz"
    if cache.exists() and not refresh:
        p = Panel.load(cache)
    else:
        p = build_panel(refresh)
        p.save(cache)
    if CONFLICT_CODING != "location":
        # (recoded from the raw UCDP files on top of the cached panel; the cache keeps the location coding)
        n2c = {n: c for c, n in zip(p.codes, p.names)}
        intra, war, inter = _conflict_arrays(load_ucdp(False), n2c, list(p.codes), CONFLICT_CODING)
        p.x["conflict"], p.x["war"] = intra, war
    return add_extras(p, refresh)


# ------------------------------------------------------------------ add-on data families
# Each add-on returns {name: [C, Y] array}; results are cached separately so that new families can
# be added (and switched off for ablations via PSYCHO_DISABLE=name,name) without rebuilding the panel.
EXTRAS = {}


def extra(name):
    def deco(fn):
        EXTRAS[name] = fn
        return fn
    return deco


def add_extras(p: Panel, refresh: bool = False) -> Panel:
    import os
    off = set(filter(None, os.environ.get("PSYCHO_DISABLE", "").split(",")))
    for name, fn in EXTRAS.items():
        if name in off:
            continue
        cache = ROOT / "data" / f"extra_{name}.npz"
        if cache.exists() and not refresh:
            arrs = dict(np.load(cache))
        else:
            try:
                arrs = fn(list(p.codes))
            except Exception as e:
                print(f"  !! add-on {name} failed: {e!r}")
                continue
            np.savez_compressed(cache, **arrs)
        for k, v in arrs.items():
            if k.startswith("patch__"):
                # fill gaps in an existing series (e.g. pre-1946 conflict history)
                tgt = k[len("patch__"):]
                p.x[tgt] = np.where(np.isnan(p.x[tgt]), v, p.x[tgt])
            else:
                p.x[k] = v
    return p


def _vdem_cols(cols):
    import pyreadr
    pq = RAW / ("vdem_" + "_".join(sorted(c.replace("e_", "") for c in cols))[:60] + ".parquet")
    if not pq.exists():
        d = pyreadr.read_r(RAW / "vdem.RData")["vdem"]
        d[["country_text_id", "year"] + cols].rename(columns={"country_text_id": "code"}).to_parquet(pq)
    d = pd.read_parquet(pq)
    d["year"] = d.year.astype(int)
    return d


@extra("history")
def load_history(codes):
    """Crisis history before the modern datasets begin, so that 1850-1945 can teach the models about
    civil war and coups (and so that 'years since the last war' reaches back before 1946).
    Civil war 1850-1945: Correlates of War intra-state wars (>= 1,000 battle deaths), via V-Dem
    e_civil_war; 89% of its war-years are also UCDP conflict-years after 1946, but it misses the
    smaller conflicts, so the pre-1946 era flag absorbs the lower base rate.
    Coups 1850-1949: Banks CNTS coups via V-Dem e_coups (matches successful Powell-Thyne coups)."""
    d = _vdem_cols(["e_civil_war", "e_coups"])
    cw = _yearly(d, codes, "e_civil_war")
    cw[:, YEARS >= 1946] = np.nan
    cw = np.where(np.isfinite(cw), (cw > 0).astype(float), np.nan)
    return {"patch__conflict": cw, "patch__war": cw}


@extra("history_coups")
def load_history_coups(codes):
    """Coups 1850-1949 from Banks CNTS (successful coups only; see load_history)."""
    d = _vdem_cols(["e_civil_war", "e_coups"])
    cp = _yearly(d, codes, "e_coups")
    cp[:, YEARS >= 1950] = np.nan
    return {"patch__coup_attempts": np.where(np.isfinite(cp), (cp > 0).astype(float), np.nan)}


def _cy(path, codes, cols, fill0_until=None):
    """tidy iso3-year csv -> {col: [C, Y]}; fill0_until: sparse event files where a missing row means 0
    for years up to that year (and NaN afterwards)."""
    d = pd.read_csv(path).rename(columns={"iso3": "code"})
    out = {}
    for c in cols:
        a = _yearly(d, codes, c)
        if fill0_until is not None:
            a = np.where(np.isnan(a), 0.0, a)
            a[:, YEARS > fill0_until] = np.nan
        out[c] = a
    return out


def _decay(a, rho):
    """exponentially decayed memory of an event series; NaN before the first observation."""
    out = np.full(a.shape, np.nan)
    m = np.full(a.shape[0], np.nan)
    for t in range(a.shape[1]):
        v = a[:, t]
        m = np.where(np.isfinite(v), np.where(np.isfinite(m), rho * m, 0.0) + v, np.where(np.isfinite(m), rho * m, np.nan))
        out[:, t] = m
    return out


def _hold(a, limit):
    out = a.copy()
    for t in range(1, a.shape[1]):
        prev = out[:, t - 1]
        out[:, t] = np.where(np.isnan(out[:, t]), prev, out[:, t])
    # only hold `limit` years past the last observation
    last = np.full(a.shape[0], -10 ** 6)
    for t in range(a.shape[1]):
        last = np.where(np.isfinite(a[:, t]), t, last)
        out[:, t] = np.where(t - last > limit, np.nan, out[:, t])
    return out


@extra("energy_trade")
def load_energy_trade(codes):
    """World Bank WDI: fuel exports and imports as % of merchandise trade (1962-2024)."""
    d = _cy(RAW / "energy" / "wdi_cy.csv", codes, ["fuel_exports_pct_merch", "fuel_imports_pct_merch"])
    return {"feat__fuel_exports": _hold(d["fuel_exports_pct_merch"], 3) / 100,
            "feat__fuel_imports": _hold(d["fuel_imports_pct_merch"], 3) / 100}


@extra("geopolitics")
def load_geopolitics(codes):
    """International crises (ICB, 1918-2022), UN General Assembly ideal points (Bailey, Strezhnev & Voeten,
    1946-2025) and country geopolitical risk in the press (Caldara & Iacoviello, 44 countries, 1900-)."""
    icb = _cy(RAW / "interstate" / "icb_cy.csv", codes, ["icb_crises_new", "icb_war_crises_active"], fill0_until=2022)
    for k in icb:
        icb[k][:, (YEARS >= 2023) & (YEARS <= LAST_OBS_YEAR)] = 0.0   # ICB ends 2022: decay through 2024
        icb[k][:, YEARS < 1918] = np.nan
    un = _cy(RAW / "interstate" / "unga_cy.csv", codes, ["ideal_point"])["ideal_point"]
    un = _hold(un, 2)
    usa = un[codes.index("USA")] if "USA" in codes else np.full(un.shape[1], np.nan)
    dist = np.abs(un - usa[None, :])
    d5 = lambda a: a - np.concatenate([np.full((a.shape[0], 5), np.nan), a[:, :-5]], 1)
    gpr = _cy(RAW / "energy" / "gpr_cy.csv", codes, ["gprhc"])["gprhc"]
    return {"feat__intl_crisis_recent": _decay(icb["icb_crises_new"], 0.7),
            "feat__intl_war_crisis": _hold(icb["icb_war_crises_active"], 0),
            "feat__un_dist_usa": dist, "feat__un_dist_usa_change_5y": d5(dist),
            "feat__log_geopol_risk": np.log1p(_hold(gpr, 2) * 10)}


@extra("politics")
def load_politics(codes):
    """Party, cabinet, leader, regime, election and protest data (see data/raw/politics/README.md).
    Hold limits keep stale sources from reaching far past their last year."""
    P = RAW / "politics"
    out = {}
    v = _cy(P / "vparty_cy.csv", codes, ["gov_antiplural", "gov_populism", "all_antiplural_votew",
                                         "max_antiplural_major", "gov_reject_violence", "coalition_n_parties",
                                         "vparty_election_year"])
    # V-Party's last coded elections are in 2019. Governing-party changes since then that were a matter of
    # record by the end of 2024 get the new governing party's own V-Party scores (from its latest coded
    # election), following V-Party's convention that an election year carries the winner.
    UPD = {("USA", 2020): (0.276, 0.319), ("USA", 2024): (0.719, 0.538), ("POL", 2023): (0.040, 0.343),
           ("BRA", 2022): (0.031, 0.777), ("GBR", 2024): (0.116, 0.514), ("ITA", 2022): (0.848, 0.652),
           ("SVK", 2023): (0.388, 0.374),
           # same governing party re-elected or succeeding itself: scores continue
           ("MEX", 2024): None, ("JPN", 2021): None, ("JPN", 2024): None, ("IND", 2024): None, ("HUN", 2022): None,
           ("TUR", 2023): None}
    for (c, y), sc in UPD.items():
        if c in codes:
            i, t = codes.index(c), int(np.searchsorted(YEARS, y))
            v["vparty_election_year"][i, t:] = y
            if sc is None:
                continue
            ap, pp = sc
            v["gov_antiplural"][i, t:] = ap
            v["gov_populism"][i, t:] = pp
            v["gov_reject_violence"][i, t:] = np.nan
            v["vparty_election_year"][i, t:] = y
    for k, a in v.items():
        out["feat__" + ("vp_" + k)] = _hold(a, 6)
    # a new governing party's rejection of violence is unknown: blank it after the hold, which would otherwise
    # carry the ousted party's score forward (code review 2026-10-28: POL, BRA, GBR, ITA, SVK)
    for (c, y), sc in UPD.items():
        if sc is not None and c in codes:
            out["feat__vp_gov_reject_violence"][codes.index(c), int(np.searchsorted(YEARS, y)):] = np.nan
    w = _cy(P / "whogov_cy.csv", codes, ["share_new_core_adj", "cabinet_size_change_pct", "share_military_core",
                                         "retention_rate_core", "n_parties_in_cabinet", "leader_party_share_core"])
    for k, a in w.items():
        out["feat__wg_" + k] = _hold(a, 1)
    ar = _cy(P / "archigos_cy.csv", codes, ["irregular_exits_past20y", "military_removals_past20y", "leader_entry_irregular"])
    for k, a in ar.items():
        out["feat__ag_" + k] = _hold(a, 9)
    g = _cy(P / "gwf_cy.csv", codes, ["gwf_personalism_theta", "gwf_personal", "gwf_military", "gwf_party"])
    for k, a in g.items():
        out["feat__" + k] = _hold(a, 14)
    n = _cy(P / "nelda_cy.csv", codes, ["elections_with_protest_past10y", "elections_with_violence_past10y",
                                        "elections_boycotted_past10y"])
    for k, a in n.items():
        out["feat__ne_" + k] = _hold(a, 14)
    m = _cy(P / "mmp_cy.csv", codes, ["protest_events", "protest_events_violent", "protest_events_repressed"],
            fill0_until=2019)
    m = {k: np.where(YEARS[None, :] < 1990, np.nan, a) for k, a in m.items()}
    # MMP ends in 2019: the decayed memory stops at the last data year and is held (decaying it further would read
    # 2020-2025 as years without protest); held 6 years to reach the 2025 origin, like NAVCO (also ends 2019)
    post = YEARS[None, :] > 2019
    for k_out, k_in in (("protest_recent", "protest_events"), ("protest_violent_recent", "protest_events_violent"),
                        ("protest_repressed_recent", "protest_events_repressed")):
        out["feat__" + k_out] = _hold(np.where(post, np.nan, np.log1p(_decay(m[k_in], 0.6))), 6)
    # NAVCO ends in 2019: an ongoing campaign is assumed to continue with probability 0.7 a year after that (campaigns
    # typically last a few years), so the 2025 value is the expected one, 0.7^6 of the last; carrying the 2019 status
    # unchanged kept e.g. Russia's opposition campaign "ongoing" through 2025
    nv = _cy(P / "navco_cy.csv", codes, ["navco_campaigns_ongoing", "navco_nonviolent_ongoing", "navco_violent_ongoing"])
    for k, a in nv.items():
        h = _hold(a, 6)
        last = np.where(np.isfinite(a).any(0))[0].max()
        fade = np.where(np.arange(a.shape[1]) > last, 0.7 ** (np.arange(a.shape[1]) - last), 1.0)
        out["feat__" + k] = h * fade[None, :]
    return out


@extra("migration")
def load_migration(codes):
    """Migration (data/raw/migration/README.md): UN WPP 2024 net migration (estimates to 2023), UN DESA migrant
    stock (1990-2024, every five years, interpolated), UNHCR refugees, asylum seekers and other people in need of
    international protection hosted, and World Bank remittances."""
    M = RAW / "migration"
    pop = pd.read_csv(RAW / "population-with-un-projections.csv")
    pop["v"] = pop.population__sex_all__age_all__variant_estimates.fillna(pop.population__sex_all__age_all__variant_medium__projected)
    P = _yearly(pop, codes, "v")
    nm = pd.read_csv(M / "net_migration_cy.csv").rename(columns={"iso3": "code"})
    nm = nm[nm.is_projection == 0]
    rate = _yearly(nm, codes, "net_migration_rate_per_1000")
    rate3 = pd.DataFrame(rate).T.rolling(3, min_periods=1).mean().T.to_numpy()
    st = pd.read_csv(M / "migrant_stock_cy.csv").rename(columns={"iso3": "code"})
    share = _yearly(st, codes, "migrant_share_pct")
    share = pd.DataFrame(share).T.interpolate(limit_area="inside").T.to_numpy()      # linear between survey years
    ref = pd.read_csv(M / "refugees_cy.csv").rename(columns={"iso3": "code"})
    ref["hosted"] = ref[["refugees_hosted", "asylum_seekers_hosted", "oip_hosted"]].fillna(0).sum(1)
    hosted = _yearly(ref, codes, "hosted")
    rem = pd.read_csv(M / "remittances_cy.csv").rename(columns={"iso3": "code"})
    remit = _yearly(rem, codes, "remittances_received_pct_gdp")
    share_h = _hold(share, 2)
    d5 = np.full(share_h.shape, np.nan)
    d5[:, 5:] = share_h[:, 5:] - share_h[:, :-5]
    return {"feat__net_migration_rate": _hold(rate3, 2), "feat__migrant_share": share_h, "feat__migrant_share_change_5y": d5,
            "feat__log_refugees_hosted_pc": _hold(np.log1p(hosted / P * 1e3), 1),
            "feat__remittances_pct_gdp": _hold(remit, 2)}


@extra("food")
def load_food(codes):
    """Food (data/raw/food/README.md): world real food prices (World Bank food index 1960-2025, FAO real FFPI for the
    partial 2026), FAOSTAT food balances (cereal import dependency, production per person, calories) and food CPI.
    World price changes enter only through each country's import dependency (a world series on its own would act as
    a calendar dummy)."""
    F = RAW / "food"
    w = pd.read_csv(F / "world_food_prices_y.csv").set_index("year")
    lp = np.log(w.wb_food_index_real)
    lp = lp.combine_first(np.log(w.ffpi_real) - np.log(w.ffpi_real.get(2025)) + lp.get(2025))     # 2026 from FFPI
    lpw = np.array([lp.get(y, np.nan) for y in YEARS])
    d1 = np.full(len(YEARS), np.nan); d1[1:] = lpw[1:] - lpw[:-1]
    d3 = np.full(len(YEARS), np.nan); d3[3:] = lpw[3:] - lpw[:-3]
    fb = pd.read_csv(F / "food_balance_cy.csv").rename(columns={"iso3": "code"})
    fb["dep"] = fb.cereal_import_dependency.clip(-4, 1)      # large exporters produce several times their use
    fb["prod_pc"] = np.log1p(fb.cereal_production_t / fb.population_fbs * 1000)      # kg per person, log
    dep = _hold(_yearly(fb, codes, "dep"), 3)
    prod = _hold(_yearly(fb, codes, "prod_pc"), 3)
    cpi = pd.read_csv(F / "food_cpi_cy.csv").rename(columns={"iso3": "code"})
    cpi["fi"] = np.log1p(cpi.food_inflation_pct.clip(-50, 500) / 100)
    cpi["rel"] = np.log(cpi.relative_food_price_2015)
    imp = dep.clip(0, None)
    return {"feat__food_shock_1y": imp * d1[None, :], "feat__food_shock_3y": imp * d3[None, :],
            "feat__cereal_import_dep": dep, "feat__cereal_prod_pc": prod,
            "feat__food_inflation": _hold(_yearly(cpi, codes, "fi"), 1),
            "feat__relative_food_price": _hold(_yearly(cpi, codes, "rel"), 1),
            # data for the simulator's food balance (not features)
            "feat__cereal_area_ha": _hold(_yearly(fb, codes, "cereal_area_ha"), 3),
            "feat__arable_ha": _hold(_yearly(fb, codes, "arable_ha"), 3)}


@extra("personal")
def load_personal(codes):
    """Personal rule and the regime's support coalition (V-Dem v15, data/raw/newdata/personal/vdem_personal_cy.csv):
    leader-cult legitimation (v2exl_legitlead) and its 5-year change, presidentialism (v2xnp_pres), regime corruption
    (v2xnp_regcorr), support from the military, foreign governments and an ethnic group (v2regsupgroups 5, 13, 6), and
    a head of state who took power by force (v2expathhs = 0). Levels held up to 3 years past V-Dem's last year.
    Adopted 2026-10-28: tuned GBM dev +0.004 (country bootstrap 90% interval +0.000 to +0.008), lockbox +0.006
    (+0.001 to +0.012); placebos with countries shuffled +0.001; production ensemble +0.002 on both. Simulator
    hindcasts (scripts/sim_eval.py pr0 / pr1): test origins 2000-2012 conflict Brier -0.5% (6/7), new-onset count CRPS
    -10.7%, coup Brier -1.0% (7/7), coup count CRPS -6.3%; dev 1970-1998 world share -4.7%, onsets -3.5%, coup count
    -7.7%, country conflict and coup Brier +0.6% / +0.8%."""
    cols = ["legit_personalist", "personalization_5y", "presidentialism", "regime_corruption", "support_military",
            "support_foreign", "support_ethnic", "path_force"]
    a = _cy(RAW / "newdata" / "personal" / "vdem_personal_cy.csv", codes, cols)
    out = {}
    for c in cols:
        v = pd.DataFrame(a[c]).T.ffill(limit=3).T.to_numpy(float)
        out["feat__" + c] = v
    return out


@extra("coalition")
def load_coalition(codes):
    """Who backs the regime and who opposes it, beyond the military, foreign and ethnic backers of load_personal (V-Dem
    v15 v2regoppgroups / v2regsupgroups, data/raw/newdata/personal/vdem_coalition_cy.csv): the military and business
    elites in opposition; religious groups, civil servants, rural workers and agrarian elites among the backers.
    Chosen from 22 coalition columns on development years only (scripts/new_data_test.py coalition_core). Tuned GBM:
    dev +0.0126 (country bootstrap 90% interval +0.004 to +0.021), lockbox +0.0069 (-0.007 to +0.021); placebo
    -0.003 / -0.008; production ensemble headline +0.009 dev, +0.014 lockbox (coups: 5-year AUC +0.012 / +0.019).
    The military in opposition is a lasting trait, not a coding made with hindsight: before coups that follow five
    calm years it is 0.43 six years ahead and 0.47 in the coup year (0.25 for all country-years).
    Adopted 2026-10-28 for coups and autocratisation only (features.TASK_ONLY): in simulator hindcasts feeding every
    task blurred civil conflict, while for coups alone (sim_eval co2 vs co0) coup Brier -2.9% dev (13/15), -3.7% test
    (7/7), conflict unchanged. Levels held up to 3 years past V-Dem's last year."""
    cols = ["opp_military", "opp_business_elites", "sup_religious", "sup_civil_servants", "sup_rural_workers",
            "sup_agrarian_elites"]
    a = _cy(RAW / "newdata" / "personal" / "vdem_coalition_cy.csv", codes, cols)
    return {"feat__" + c: pd.DataFrame(a[c]).T.ffill(limit=3).T.to_numpy(float) for c in cols}


@extra("events")
def load_events(codes):
    """Machine-coded events (ICEWS 1995-2022, POLECAT 2018-2024 with location-imputed domestic actors).
    Levels differ between and within sources, so only within-country surges are used: log(1 + count this year)
    minus log(1 + mean of the previous three years), each computed inside one source. ICEWS supplies surges up
    to 2022, POLECAT for 2023-2024 (2024 counts scaled to a full year)."""
    E = RAW / "events"
    cols = ["n_material_conflict", "n_protest", "n_repression", "n_mass_violence", "n_threats", "n_events"]

    def surges(path, last_year):
        d = pd.read_csv(path).rename(columns={"iso3": "code"})
        if "days_covered" in d:
            d[cols] = d[cols].mul(365 / d.days_covered.clip(lower=30), axis=0)
        out = {}
        for c in cols:
            a = _yearly(d, codes, c)
            a[:, YEARS > last_year] = np.nan
            la = np.log1p(a)
            prev = np.log1p(np.stack([np.roll(a, s_, 1) for s_ in (1, 2, 3)], 0).mean(0))
            prev[:, :3] = np.nan
            out[c] = la - prev
        tot = _yearly(d, codes, "n_events")
        out["share"] = np.where(tot > 20, _yearly(d, codes, "n_material_conflict") / np.maximum(tot, 1), np.nan)
        return out
    ic = surges(E / "icews_cy.csv", 2022)
    pc = surges(E / "polecat_loc_cy.csv", 2024)
    late = (YEARS >= 2023)[None, :]
    pick = lambda k: np.where(late, pc[k], ic[k])
    return {"feat__ev_conflict_surge": pick("n_material_conflict"), "feat__ev_protest_surge": pick("n_protest"),
            "feat__ev_repression_surge": pick("n_repression"), "feat__ev_violence_surge": pick("n_mass_violence"),
            "feat__ev_threat_surge": pick("n_threats"), "feat__ev_attention_surge": pick("n_events")}


def observed_oil(origin=LAST_OBS_YEAR, cpi_growth=0.027):
    """((year, d log real oil price vs the previous year, sd), ...) for the year after the origin, from the World Bank
    monthly Pink Sheet (crude oil average, data/raw/energy/wb_cmo_monthly). The year's observed months are averaged;
    the unobserved months add sd = 0.25 * share missing (monthly log price changes ~0.08, compounded). Deflated by
    an assumed US CPI growth of 2.7%."""
    f = RAW / "energy" / "wb_cmo_monthly" / "CMO-Historical-Data-Monthly.xlsx"
    if not f.exists():
        return ()
    x = pd.read_excel(f, sheet_name="Monthly Prices", header=4)
    x = x.rename(columns={x.columns[0]: "month"})
    x = x[x.month.astype(str).str.match(r"^\d{4}M\d{2}$")]
    x["year"] = x.month.str[:4].astype(int)
    x["v"] = pd.to_numeric(x["Crude oil, average"], errors="coerce")
    y0, y1 = x[x.year == origin].v.dropna(), x[x.year == origin + 1].v.dropna()
    if len(y0) < 12 or not len(y1):
        return ()
    ch = float(np.log(y1.mean() / y0.mean()) - np.log1p(cpi_growth))
    return ((origin + 1, round(ch, 4), round(0.25 * (12 - len(y1)) / 12, 4)),)

