"""Tropical-cyclone exposure by country from IBTrACS v04r01 (NOAA NCEI, public domain), 1980-2024: the average share of a
country's land under hurricane-force winds (>= 64 kt) each year, as in Hsiang & Jina (2014). Each 3-hourly track point
at hurricane strength is buffered by 60 km (a typical radius of hurricane-force winds); the union of each year's
buffers is intersected with country boundaries (world-atlas 110m, Natural Earth) in an equal-area projection.
Small island states missing from the 110m boundaries use a 100 km disc around their centroid.

Replaces the proxy (population near the coast x tropical share) behind the warming-driven extra storms in
simulate.py (storm_trend, storm_loss): the exposure is rescaled to the proxy's mean over exposed countries.
python -P scripts/build_cyclone_exposure.py -> data/cyclone_exposure.npz (codes, area_share, hit_prob)"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
EA = "ESRI:54009"          # Mollweide equal-area


def main():
    from psychohistory.data import get_panel
    p = get_panel()
    tr = pd.read_csv(os.path.join(RAW, "climate", "ibtracs", "ibtracs.since1980.list.v04r01.csv"), skiprows=[1],
                     usecols=["SID", "SEASON", "LAT", "LON", "WMO_WIND", "USA_WIND", "DIST2LAND"], low_memory=False)
    for c in ("LAT", "LON", "WMO_WIND", "USA_WIND", "DIST2LAND", "SEASON"):
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
    tr["wind"] = tr.USA_WIND.fillna(tr.WMO_WIND)
    tr = tr[(tr.wind >= 64) & (tr.DIST2LAND <= 200) & (tr.SEASON >= 1980) & (tr.SEASON <= 2024)]
    print("hurricane-force points near land:", len(tr), flush=True)
    world = gpd.read_file(os.path.join(RAW, "countries-110m.json"), layer="countries")
    if world.crs is None:
        world = world.set_crs("EPSG:4326")
    iso = pd.read_csv(os.path.join(RAW, "iso3166.csv"), dtype=str, keep_default_na=False)
    num2a3 = {n: a for n, a in zip(iso["country-code"], iso["alpha-3"])}
    world["iso3"] = world["id"].map(lambda i: num2a3.get(str(i).zfill(3)) if i is not None else None)
    world.loc[world["name"] == "Kosovo", "iso3"] = "XKX"
    world.loc[world["name"] == "Taiwan", "iso3"] = "TWN"
    world["geometry"] = world.geometry.make_valid()
    world = world[world.iso3.isin(p.codes)].dissolve("iso3").to_crs(EA)
    cen = pd.read_csv(os.path.join(RAW, "country_centroids.csv"), keep_default_na=False)
    a2to3 = dict(zip(iso["alpha-2"], iso["alpha-3"]))
    cen["iso3"] = cen.ISO.map(a2to3)
    missing = [c for c in p.codes if c not in world.index]
    extra = []
    for c in missing:
        r = cen[cen.iso3 == c]
        if len(r):
            extra.append({"iso3": c, "geometry": Point(float(r.longitude.iloc[0]), float(r.latitude.iloc[0]))})
    if extra:
        ex = gpd.GeoDataFrame(extra, crs="EPSG:4326").to_crs(EA)
        ex["geometry"] = ex.buffer(100_000)
        world = pd.concat([world[["geometry"]], ex.set_index("iso3")[["geometry"]]])
    print("countries with geometry:", len(world), "of", len(p.codes), "(discs for", len(extra), "small states)", flush=True)
    area = world.area
    years = range(1980, 2025)
    share = pd.DataFrame(0.0, index=world.index, columns=list(years))
    pts = gpd.GeoDataFrame(tr, geometry=gpd.points_from_xy(tr.LON, tr.LAT), crs="EPSG:4326").to_crs(EA)
    for y in years:
        g = pts[pts.SEASON == y]
        if g.empty:
            continue
        u = g.buffer(60_000).union_all()
        hit = world.intersection(u).area / area
        share[y] = hit.fillna(0.0)
    out_share = np.full(len(p.codes), np.nan)
    out_prob = np.full(len(p.codes), np.nan)
    for i, c in enumerate(p.codes):
        if c in share.index:
            out_share[i] = float(share.loc[c].mean())
            out_prob[i] = float((share.loc[c] > 0.01).mean())
    np.savez_compressed(os.path.join(ROOT, "data", "cyclone_exposure.npz"), codes=np.array(p.codes), area_share=out_share,
                        hit_prob=out_prob)
    s = pd.Series(out_share, index=p.codes).sort_values(ascending=False)
    print("mean share of land under hurricane-force winds per year, top 20:")
    print((s.head(20) * 100).round(1).to_string())
    for c in ("USA", "CHN", "IND", "JPN", "PHL", "MEX", "BGD", "VNM"):
        print(c, round(float(s.get(c, np.nan)) * 100, 2), "% ; P(year with >1% hit)", round(float(out_prob[list(p.codes).index(c)]), 2))


if __name__ == "__main__":
    main()
