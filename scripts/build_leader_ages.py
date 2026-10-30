"""Leader birth years for 2021-2025 (after REIGN ends) on the panel grid: V-Dem's effective leader each year (as in
psychohistory.data.load_vdem) joined to data/raw/politics/leaders_wikidata.csv (fetch_leader_ages.py).
python -P scripts/build_leader_ages.py -> data/leader_birth_wd.npz (codes, years, birth [C, Y], NaN where unknown)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    from psychohistory.data import get_panel
    p = get_panel()
    d = pd.read_parquet(os.path.join(ROOT, "data", "raw", "vdem_extra_v3.parquet"))
    d = d.rename(columns={"country_text_id": "code"}) if "country_text_id" in d else d
    use_hog = (d.v2ex_hosw < 0.5) & d.v2exnamhog.notna() & (d.v2exnamhog.astype(str).str.len() > 0)
    d["leader"] = np.where(use_hog, d.v2exnamhog, d.v2exnamhos)
    wd = pd.read_csv(os.path.join(ROOT, "data", "raw", "politics", "leaders_wikidata.csv"))
    wd = wd[wd.birth_year.notna()].drop_duplicates(["code", "leader"])
    d = d[d.year >= 2021].merge(wd[["code", "leader", "birth_year"]], on=["code", "leader"], how="left")
    ci, yi = {c: i for i, c in enumerate(p.codes)}, {int(y): j for j, y in enumerate(p.years)}
    birth = np.full((len(p.codes), len(p.years)), np.nan)
    for r in d.itertuples():
        if r.code in ci and int(r.year) in yi and pd.notna(r.birth_year):
            birth[ci[r.code], yi[int(r.year)]] = r.birth_year
    np.savez_compressed(os.path.join(ROOT, "data", "leader_birth_wd.npz"), codes=np.array(p.codes), years=p.years, birth=birth)
    t = yi[2025]
    print(f"2025: {np.isfinite(birth[:, t]).sum()} of {len(p.codes)} countries with a leader birth year")


if __name__ == "__main__":
    main()
