"""Generational memory: the share of a country's population today who were at least 8 years old during its most
recent war, civil conflict, hyperinflation or economic collapse (so who can remember it).
Ages: UN WPP five-year age groups via Our World in Data (1950-2023; earlier years take the 1950 structure, later the
last available), linear within each band. Events:
  interstate war   COW inter-state war participation, a country's battle deaths >= 500 (1816-2007), then UCDP
                   war-level interstate conflict (>= 1000 deaths, 2008-2024)
  civil conflict   the panel's civil-conflict indicator (1850-)
  hyperinflation   consumer-price inflation >= 100% a year (Global Macro Database, World Bank where missing)
  collapse         GDP per person >= 15% below its previous 10-year peak (Maddison/World Bank, 1850-)
python -P scripts/build_memory.py -> data/raw/newdata/memory/memory_cy.csv"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import pyreadr

from psychohistory import interstate as I
from psychohistory.data import get_panel, RAW
from psychohistory.features import base_arrays

OUT = RAW / "newdata" / "memory"
MIN_AGE = 8


def age_shares(codes, years):
    """cum[c, t, a] = share of the population aged >= a (a = 0..105)."""
    d = pd.read_csv(OUT / "pop5.csv")
    cols = [c for c in d.columns if c.startswith("population__sex_all__age_")]
    lo = [int(c.split("__age_")[1].split("_")[0].replace("plus", "")) for c in cols]
    order = np.argsort(lo)
    cols, lo = [cols[i] for i in order], [lo[i] for i in order]
    d = d[d.code.isin(codes)]
    cum = np.full((len(codes), len(years), 106), np.nan)
    ci = {c: i for i, c in enumerate(codes)}
    yi = {y: i for i, y in enumerate(years)}
    for (code, y), r in d.set_index(["code", "year"])[cols].iterrows():
        if y not in yi:
            continue
        v = r.to_numpy(float)
        tot = v.sum()
        if not tot > 0:
            continue
        above = np.cumsum(v[::-1])[::-1] / tot                 # share aged >= lower bound of each band
        a = np.arange(106)
        cum[ci[code], yi[y]] = np.interp(a, lo + [105], list(above) + [0.0])
    # years outside 1950-2023: nearest available structure
    for c in range(len(codes)):
        ok = np.flatnonzero(np.isfinite(cum[c, :, 0]))
        if len(ok):
            cum[c, :ok[0]] = cum[c, ok[0]]
            cum[c, ok[-1] + 1:] = cum[c, ok[-1]]
    return cum


def main():
    p = get_panel()
    b = base_arrays(p)
    codes, years = list(p.codes), [int(y) for y in p.years]
    C, T = len(codes), len(years)
    yi = {y: i for i, y in enumerate(years)}
    ev = {k: np.zeros((C, T), bool) for k in ("interstate_war", "civil_conflict", "hyperinflation", "collapse")}
    # interstate wars
    w = list(pyreadr.read_r(RAW / "system" / "cow_war_inter.rda").values())[0]
    c2i = I._ccode_to_iso(set(codes))
    ci = {c: i for i, c in enumerate(codes)}
    for cc, y, bd in zip(w.ccode1, w.year, w.batdeath1):
        iso = c2i.get(int(cc))
        if iso in ci and int(y) in yi and (bd >= 500 or bd < 0):        # (-9 = unknown deaths: kept)
            ev["interstate_war"][ci[iso], yi[int(y)]] = True
    for (a, b_, y), inten in I._ucdp_interstate(set(codes)).items():
        if y >= 2008 and inten == 2 and y in yi:
            for x in (a, b_):
                if x in ci:
                    ev["interstate_war"][ci[x], yi[y]] = True
    ev["civil_conflict"] = np.nan_to_num(b["conflict"]) > 0
    # inflation: Global Macro Database (long history), the World Bank series where it is missing
    g = pd.read_csv(RAW / "GMD.csv", usecols=["ISO3", "year", "infl"])
    gi = g.pivot_table(index="ISO3", columns="year", values="infl").reindex(index=codes, columns=years).to_numpy(float)
    infl = np.where(np.isfinite(gi), gi, b["inflation"])
    ev["hyperinflation"] = np.nan_to_num(infl) >= 100
    lg = b["lg"]
    peak = np.full_like(lg, np.nan)
    for t in range(1, T):
        peak[:, t] = np.nanmax(lg[:, max(0, t - 10):t], 1)
    ev["collapse"] = np.nan_to_num(lg - peak, nan=0) <= np.log(0.85)
    cum = age_shares(codes, years)
    rows = {"iso3": np.repeat(codes, T), "year": np.tile(years, C)}
    for k, e in ev.items():
        mem = np.zeros((C, T))
        last = np.full(C, -10_000)
        for t in range(T):
            last = np.where(e[:, t], years[t], last)
            age = years[t] - last + MIN_AGE                     # youngest age that remembers the last event
            a = np.clip(age, 0, 105).astype(int)
            share = np.take_along_axis(cum[:, t], a[:, None], 1)[:, 0]
            mem[:, t] = np.where(age > 105, 0.0, share)
        rows[f"mem_{k}"] = mem.ravel()
    out = pd.DataFrame(rows)
    out = out[out.year >= 1946]
    out.round(4).to_csv(OUT / "memory_cy.csv", index=False)
    s = out.set_index(["iso3", "year"])
    for c, y in (("DEU", 1980), ("DEU", 2025), ("JPN", 2025), ("USA", 2025), ("SYR", 2010), ("RWA", 2025), ("ARG", 2001), ("RUS", 2021)):
        print(c, y, s.loc[(c, y)].round(2).to_dict())


if __name__ == "__main__":
    main()
