"""Shared rivers, water disputes and glacier loss.

Data
- Transboundary Freshwater Dispute Database (TFDD, Oregon State University), spatial database 2024 update: the 313
  international river basins and each country's area in each basin (basin-country units). data/raw/water/tfdd/
- TFDD International Water Events (Basins at Risk), 1948-2008 (events typed as water interactions run 1948-1999): about 7,000 events between countries over shared
  water, scored -7 (formal war over water) to +7 (merger into one nation) on the BAR scale. data/raw/water/events/
- Glacier peak water by basin: Wimberly et al. (2025), The Cryosphere 19, 1491-1511, three glacier models x 12 CMIP6
  models; median year of peak glacier runoff under SSP2-4.5, read from the authors' published figure
  (github.com/finnwimberly/glacial_runoff_intercomparison, Manuscript Figs & Data/RF_PeakWater.ipynb).

Pair features (undirected pairs, as in interstate.py)
  shared_basin     the two share at least one international basin
  log_shared_area  log(1 + sum over shared basins of the smaller of the two countries' areas in the basin, km2)
  basin_dep_max    the larger of the two countries' share of their own territory in basins shared with the other
  water_conf_hist  log(1 + conflictive water events between them (BAR -1 or lower) before the year); events stop in
                   1999, so later years carry the 1948-1999 total
  water_coop_hist  the same for cooperative events (BAR +1 or higher), treaties included
  scarcity_shared  shared_basin x log(1 + the larger of the two countries' water stress, % of internal resources
                   withdrawn); water stress is observed from 2000 only, so each country's 2000-2010 mean is used
                   for every year (a slow structural trait; slight look-ahead before 2000)
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from .data import RAW

W = RAW / "water"
# old or non-ISO codes in the events file
EV_CODE_FIX = {"CZS": "CZE", "ROM": "ROU", "ZAR": "COD", "YUG": "SRB", "USR": "RUS", "GDR": "DDR", "BRM": "MMR",
               "FRG": "DEU", "GFR": "DEU", "SUD": "SDN", "KAM": "KHM", "BNG": "BGD"}

# glacier-fed international basins: median year of peak glacier runoff, SSP2-4.5 (Wimberly et al. 2025, figure of
# peak-water years; the three glacier models' medians GloGEM / OGGM / PyGEM, averaged). TFDD basin name -> year.
# Basins whose peak has passed lose glacier water from now on; later peaks lose it after that year.
PEAK_WATER = {
    "Indus": (2062 + 2052 + 2054) / 3,
    "Aral Sea": (2053 + 2051 + 2052) / 3,
    "Tarim": (2054 + 2058 + 2057) / 3,
    "Ganges-Brahmaputra-Meghna": ((2028 + 2040 + 2034) / 3 + (2022 + 2027 + 2024) / 3) / 2,
    "Mekong": (2025 + 2026 + 2028) / 3,
    "Salween": (2022 + 2022 + 2023) / 3,
    "Irrawaddy": (2022 + 2023 + 2018) / 3,
    "Shu/Chu": (2026 + 2034 + 2029) / 3,
    "Talas": (2024 + 2028 + 2027) / 3,
    "Ili/Kunes He": (2028 + 2033 + 2032) / 3,       # Lake Balkhash basin
    "Ob": (2027 + 2031 + 2028) / 3,
    "Danube": (2020 + 2024 + 2024) / 3,
    "Rhine-Meuse": (2022 + 2018 + 2022) / 3,
    "Po": (2020 + 2020 + 2024) / 3,
    "Rhone": (2026 + 2024 + 2025) / 3,
    "Amazon": (2021 + 2024 + 2024) / 3,
    "Columbia": (2036 + 2036 + 2029) / 3,
    "Yukon": (2069 + 2061 + 2058) / 3,
}


# share of each basin's river flow that is glacier melt (assumption, order of magnitude; each world scales all of
# them by glacier_flow_mult, 0.5-1.5). Upper-basin shares from Lutz et al. (2014, Nature Climate Change: glacier melt
# ~41% of upper Indus flow, ~16% upper Brahmaputra, ~12% upper Ganges, ~9% upper Salween, ~1% upper Mekong) and
# Immerzeel et al. (2010, Science: the Indus the most melt-dependent Asian basin), reduced for rain-fed lower basins;
# Tarim and the Central Asian rivers are fed largely by Tien Shan and Pamir glaciers; Alpine rivers carry a few %.
GLACIER_FLOW = {
    "Indus": 0.25, "Aral Sea": 0.15, "Tarim": 0.40, "Ganges-Brahmaputra-Meghna": 0.05, "Mekong": 0.01,
    "Salween": 0.03, "Irrawaddy": 0.01, "Shu/Chu": 0.20, "Talas": 0.20, "Ili/Kunes He": 0.15, "Ob": 0.005,
    "Danube": 0.005, "Rhine-Meuse": 0.02, "Po": 0.02, "Rhone": 0.05, "Amazon": 0.002,
    "Columbia": 0.02, "Yukon": 0.05,
}

# live interstate water disputes at the end of 2025 (each world adds water_dispute to the pair's onset log-odds)
WATER_DISPUTES = {
    ("IND", "PAK"): "Indus Waters Treaty held in abeyance by India since April 2025",
    ("CHN", "IND"): "Yarlung Tsangpo (Medog) mega-dam under construction since July 2025; no water treaty",
    ("EGY", "ETH"): "Grand Ethiopian Renaissance Dam in operation without an agreement with Egypt",
    ("AFG", "IRN"): "Helmand flows under the 1973 treaty; border clash in 2023",
    ("AFG", "UZB"): "Qosh Tepa canal diverting the Amu Darya",
    ("AFG", "TKM"): "Qosh Tepa canal diverting the Amu Darya",
    ("IRQ", "TUR"): "Tigris-Euphrates dams and Iraqi shortages",
    ("SYR", "TUR"): "Euphrates dams",
}


@functools.lru_cache(maxsize=1)
def bcu():
    import pyogrio
    b = pyogrio.read_dataframe(W / "tfdd" / "BCUMaster313_20240807.shp", read_geometry=False)
    b = b[~b.CCODE.str.contains("/")]                       # disputed areas belong to neither claimant
    return b[["CCODE", "BCODE", "Basin_Name", "Area_km2", "Pop_2020", "Dams_Exist", "Dam_Plnd"]].rename(columns={"CCODE": "iso3"})


@functools.lru_cache(maxsize=1)
def country_area():
    import pyogrio
    g = pyogrio.read_dataframe(W / "tfdd" / "Gual_Countries_WCylinEA_20240807.shp", read_geometry=False)
    g = g[~g.CCODE.str.contains("/")]
    return g.groupby("CCODE").Area_km2.sum().to_dict()


@functools.lru_cache(maxsize=1)
def pair_basins():
    """{(a, b): {"basins": [names], "shared_area": km2, "dep_a": share, "dep_b": share}} for pairs sharing a basin."""
    b = bcu()
    area = country_area()
    by_basin = b.groupby("Basin_Name")
    out = {}
    for name, g in by_basin:
        ar = g.groupby("iso3").Area_km2.sum()
        cs = sorted(ar.index)
        for i, x in enumerate(cs):
            for y in cs[i + 1:]:
                d = out.setdefault((x, y), {"basins": [], "shared_area": 0.0, "own_a": 0.0, "own_b": 0.0})
                d["basins"].append(name)
                d["shared_area"] += float(min(ar[x], ar[y]))
                d["own_a"] += float(ar[x])
                d["own_b"] += float(ar[y])
    for (x, y), d in out.items():
        d["dep_a"] = min(d["own_a"] / area.get(x, np.inf), 1.0)
        d["dep_b"] = min(d["own_b"] / area.get(y, np.inf), 1.0)
    return out


@functools.lru_cache(maxsize=1)
def events():
    """water events between two countries: DataFrame a, b, year, bar (BAR scale)."""
    e = pd.read_excel(W / "events" / "EventMaster111710.xls", sheet_name="EventMaster2")
    e = e[e.EVENT_TYPE.astype(str).str.startswith("y") & e.BAR_Scale.notna()]
    e["year"] = pd.to_datetime(e.DATE, errors="coerce").dt.year
    a = e.CCODE1.astype(str).str.strip().replace(EV_CODE_FIX)
    b = e.CCODE2.astype(str).str.strip().replace(EV_CODE_FIX)
    out = pd.DataFrame({"a": np.where(a < b, a, b), "b": np.where(a < b, b, a), "year": e.year, "bar": e.BAR_Scale})
    return out[out.year.notna() & (out.a.str.len() == 3) & (out.b.str.len() == 3)].astype({"year": int})


@functools.lru_cache(maxsize=1)
def water_stress():
    w = pd.read_csv(RAW / "freshwater-withdrawals-as-a-share-of-internal-resources.csv")
    w = w[(w.year >= 2000) & (w.year <= 2010)]
    return w.groupby("code")[w.columns[-1]].mean().to_dict()


FEATS = ["shared_basin", "log_shared_area", "basin_dep_max", "water_conf_hist", "water_coop_hist", "scarcity_shared"]


def add_features(df):
    """add the water pair features to an interstate dyad-year frame (columns a, b, year)."""
    pb = pair_basins()
    ws = water_stress()
    key = list(zip(df.a, df.b))
    info = [pb.get(k) for k in key]
    df = df.copy()
    df["shared_basin"] = [float(d is not None) for d in info]
    df["log_shared_area"] = [np.log1p(d["shared_area"]) if d else 0.0 for d in info]
    df["basin_dep_max"] = [max(d["dep_a"], d["dep_b"]) if d else 0.0 for d in info]
    st = np.array([np.nanmax([ws.get(a, np.nan), ws.get(b, np.nan)]) if (a in ws or b in ws) else np.nan
                   for a, b in key])
    df["scarcity_shared"] = df.shared_basin * np.log1p(st)
    ev = events()
    for name, sel in (("water_conf_hist", ev.bar <= -1), ("water_coop_hist", ev.bar >= 1)):
        c = ev[sel].groupby(["a", "b", "year"]).size().rename("n").reset_index()
        m = df[["a", "b", "year"]].merge(c, on=["a", "b", "year"], how="left").n.fillna(0).to_numpy()
        s = pd.Series(m, index=df.index)
        order = df.sort_values(["a", "b", "year"]).index
        cum = s.loc[order].groupby([df.a.loc[order], df.b.loc[order]]).transform(lambda x: x.shift(1).cumsum()).fillna(0)
        df[name] = np.log1p(cum.reindex(df.index))
    return df


@functools.lru_cache(maxsize=1)
def _pop2020():
    w = pd.read_csv(RAW / "population.csv")
    w = w[w.year == 2020]
    return w.groupby("code")[w.columns[-1]].sum().to_dict()


def glacier_share(iso3):
    """share of a country's population living in each glacier-fed international basin (TFDD 2020 population by
    basin-country unit; population stands in for where water is used): {basin: share}."""
    b = bcu()
    pop = _pop2020().get(iso3, np.nan)
    mine = b[(b.iso3 == iso3) & b.Basin_Name.isin(PEAK_WATER)]
    return {k: float(min(v / pop, 1.0)) for k, v in mine.groupby("Basin_Name").Pop_2020.sum().items() if np.isfinite(v / pop)}


def binding(codes, origin=9999):
    """how binding water is for each country's farms, 0-1: water stress (withdrawals as % of renewable resources
    after environmental flows, SDG 6.4.2), latest five-year mean, /100 and capped at 1. A change in supply matters
    little where withdrawals are a small share of it (Bangladesh 6%) and fully where they exceed it (Pakistan,
    Egypt, Uzbekistan)."""
    w = pd.read_csv(RAW / "freshwater-withdrawals-as-a-share-of-internal-resources.csv")
    w = w[w.year <= max(origin, int(w.year.min()))]        # (observed from 2000: earlier origins use the first years)
    w = w[w.year >= w.year.max() - 4]
    m = w.groupby("code")[w.columns[-1]].mean()
    return np.clip(np.array([m.get(c, np.nan) for c in codes], float) / 100, 0, 1)
