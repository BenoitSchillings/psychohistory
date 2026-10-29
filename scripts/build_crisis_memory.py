"""Long crisis memory (Reinhart & Rogoff's "debt intolerance": serial defaulters and inflation-prone countries stay so for
centuries). Country-year, 1800-2025:
  default_years_since1800   years in external default since 1800 (Reinhart-Rogoff, 70 countries, to 2016; then
                            BoC-BoE private-creditor default >= US$100 mn)
  log_years_since_default   years since the last year in external default (80 = none since 1800)
  inflation_crises_since1800  years with inflation above 20% since 1800 (Reinhart-Rogoff to 2016, then the Global
                            Macro Database / World Bank)
  log_years_since_inflation_crisis
  banking_crises_since1800  banking-crisis years since 1800 (Reinhart-Rogoff to 2016, then GMD)
Countries outside Reinhart-Rogoff's 70 are missing before 1960 and use BoC-BoE / GMD from 1960, with the count marked
missing (unknown history) rather than zero.
python -P scripts/build_crisis_memory.py -> data/raw/newdata/religion/../crisis_memory_cy.csv"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT = os.path.join(RAW, "newdata", "crisis_memory_cy.csv")


def main():
    rr = pd.read_csv(os.path.join(RAW, "political_economy", "reinhart_rogoff_crises.csv"))
    bb = pd.read_csv(os.path.join(RAW, "political_economy", "sovereign_defaults_boc_boe.csv"), usecols=["iso3", "year", "in_default_ge100mn"])
    g = pd.read_csv(os.path.join(RAW, "GMD.csv"), usecols=["ISO3", "year", "infl", "BankingCrisis"]).rename(columns={"ISO3": "iso3"})
    rr_codes = set(rr.iso3)
    codes = sorted(rr_codes | set(bb.iso3))
    years = np.arange(1800, 2026)
    idx = pd.MultiIndex.from_product([codes, years], names=["iso3", "year"])
    df = pd.DataFrame(index=idx).reset_index()
    df = df.merge(rr[["iso3", "year", "external_default_rr1", "inflation_crisis", "banking_crisis", "inflation_pct"]], how="left", on=["iso3", "year"])
    df = df.merge(bb, how="left", on=["iso3", "year"]).merge(g, how="left", on=["iso3", "year"])
    in_rr = df.iso3.isin(rr_codes)
    late = df.year > 2016
    default = np.where(in_rr & ~late, df.external_default_rr1, df.in_default_ge100mn)
    infl = np.where(in_rr & ~late, df.inflation_crisis, np.where(df.infl.notna(), (df.infl > 20).astype(float), np.nan))
    bank = np.where(in_rr & ~late, df.banking_crisis, df.BankingCrisis)
    df["d"], df["i"], df["b"] = default, infl, bank
    out = []
    for c, s in df.groupby("iso3", sort=False):
        s = s.sort_values("year")
        known_from = 1800 if c in rr_codes else 1960
        rec = {}
        for k, name in (("d", "default"), ("i", "inflation_crisis"), ("b", "banking")):
            v = s[k].to_numpy(float)
            cnt, last, cnts, since = 0.0, None, [], []
            for y, x in zip(s.year, v):
                if y < known_from:
                    cnts.append(np.nan); since.append(np.nan)
                    continue
                if x == 1:
                    cnt += 1; last = y
                cnts.append(cnt); since.append(np.log1p(min(y - last, 80)) if last is not None else np.log1p(80))
            rec[name] = (cnts, since)
        part = pd.DataFrame({"iso3": c, "year": s.year.to_numpy(),
                             "default_years_since1800": rec["default"][0], "log_years_since_default": rec["default"][1],
                             "inflation_crises_since1800": rec["inflation_crisis"][0],
                             "log_years_since_inflation_crisis": rec["inflation_crisis"][1],
                             "banking_crises_since1800": rec["banking"][0]})
        if c not in rr_codes:          # history before 1960 unknown: counts are not comparable
            for k in ("default_years_since1800", "inflation_crises_since1800", "banking_crises_since1800"):
                part[k] = np.nan
        out.append(part)
    o = pd.concat(out)
    o = o[o.year >= 1850]
    o.round(4).to_csv(OUT, index=False)
    s = o.set_index(["iso3", "year"])
    for c in ("ARG", "GRC", "DEU", "USA", "TUR", "NGA"):
        print(c, s.loc[(c, 2000)].round(2).to_dict() if (c, 2000) in s.index else None)


if __name__ == "__main__":
    main()
