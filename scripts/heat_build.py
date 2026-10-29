"""Humid heat by country-year from ERA5 (scripts/heat_download.py).

Per 1-degree cell and month: wet-bulb temperature (Stull 2011, from 2 m temperature and dew point via the Magnus
relative humidity) at 00, 06, 12 and 18 UTC of the month's mean day, maximum over the four hours (roughly the mean
afternoon peak); per year the hottest month. A monthly mean of a 6-hourly peak is well below the daily extremes that
kill (a wet-bulb of 31-35 C), so the thresholds are relative: the series is a measure of how humid-hot a country's
hot season is, and of how unusual a year is against its own past.
Countries: cells whose centre falls in the Natural Earth 110 m outline (area-weighted by cos latitude); a country with
no cell centre takes the cell nearest its representative point.
Columns: tw_mean (area-mean of each cell's hottest-month wet-bulb), tw_p90 (90th percentile cell), tw_hot24 (share of
area with a hottest-month wet-bulb of 24 C or more), ta_mean (the same for dry 2 m temperature, for comparison).
python -P scripts/heat_build.py -> data/raw/heat/wetbulb_country_year.csv"""
import glob
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
HEAT = os.path.join(RAW, "heat")


def stull(t, rh):
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def wetbulb(t2m_k, d2m_k):
    t, td = t2m_k - 273.15, d2m_k - 273.15
    rh = np.clip(100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t)), 1, 100)
    return stull(t, rh), t


def cell_map(lat, lon):
    """country (ISO3) of each cell centre, and cos-latitude weights"""
    g = gpd.read_file(os.path.join(RAW, "countries-110m.json"), layer="countries")
    iso = pd.read_csv(os.path.join(RAW, "iso3166.csv"), dtype=str)
    num2a3 = dict(zip(iso["country-code"].str.zfill(3), iso["alpha-3"]))
    g["iso3"] = g["id"].astype(str).str.zfill(3).map(num2a3)
    g.loc[g.name == "Kosovo", "iso3"] = "XKX"
    g = g.dropna(subset=["iso3"])
    LON, LAT = np.meshgrid(np.where(lon > 180, lon - 360, lon), lat)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(LON.ravel(), LAT.ravel()), crs=g.crs)
    j = gpd.sjoin(pts, g[["iso3", "geometry"]], predicate="within", how="left")
    j = j[~j.index.duplicated()]
    owner = j["iso3"].reindex(range(len(pts))).to_numpy().reshape(LAT.shape)
    cells = {c: np.argwhere(owner == c) for c in g.iso3.unique()}
    for c, rp in zip(g.iso3, g.representative_point()):
        if not len(cells.get(c, [])):
            i = int(np.abs(lat - rp.y).argmin())
            k = int(np.abs(((np.where(lon > 180, lon - 360, lon) - rp.x) + 180) % 360 - 180).argmin())
            cells[c] = np.array([[i, k]])
    return cells, np.cos(np.deg2rad(LAT))


def main():
    files = sorted(glob.glob(os.path.join(HEAT, "era5_td_*.nc")))
    rows, cells = [], None
    for f in files:
        y = int(re.search(r"(\d{4})\.nc$", f).group(1))
        ds = xr.open_dataset(f)
        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        tw, ta = wetbulb(ds["t2m"].values, ds["d2m"].values)              # [time, lat, lon]
        month = pd.DatetimeIndex(ds[tdim].values).month
        tw_m = np.stack([tw[month == m].max(0) for m in range(1, 13)])     # monthly max over the 4 hours
        ta_m = np.stack([ta[month == m].max(0) for m in range(1, 13)])
        tw_y, ta_y = tw_m.max(0), ta_m.max(0)                              # hottest month
        if cells is None:
            cells, w = cell_map(ds.latitude.values, ds.longitude.values)
        for c, idx in cells.items():
            i, k = idx[:, 0], idx[:, 1]
            ww = w[i, k]
            v, d = tw_y[i, k], ta_y[i, k]
            rows.append({"code": c, "year": y, "n_cells": len(idx), "tw_mean": float(np.average(v, weights=ww)),
                         "tw_p90": float(np.quantile(v, 0.9)), "tw_hot24": float(np.average(v >= 24, weights=ww)),
                         "ta_mean": float(np.average(d, weights=ww))})
        print(y, flush=True)
    out = pd.DataFrame(rows).round(4)
    out.to_csv(os.path.join(HEAT, "wetbulb_country_year.csv"), index=False)
    print(out.groupby("code").tw_mean.mean().sort_values().tail(10))


if __name__ == "__main__":
    main()
