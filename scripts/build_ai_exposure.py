"""Each country's exposure to near-free cognition from its occupational structure: the share of employment in ISCO-08
major groups 1-4 (managers, professionals, technicians, clerical support), the groups most exposed to generative AI
(Gmyrek, Berg & Bescond, ILO 2023/2025), from the ILO modelled estimates (data/raw/tech/ilo_occupations).

world.py used 0.3 + 0.5 x the country's rank on a readiness index (energy, tertiary enrolment, internet, rule of law,
income) as the cognitive share of output. The occupation share replaces the rank, mapped linearly so that the
across-country mean and spread of the cognitive share are unchanged: the data decide which countries are exposed, the
earlier calibration how much.
python -P scripts/build_ai_exposure.py -> data/ai_exposure.npz (codes, occ14 share, year used, cog_share)"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
TARGET_YEAR = 2023


def main():
    from psychohistory.data import get_panel
    p = get_panel()
    emp = {}
    for f in glob.glob(os.path.join(ROOT, "data", "raw", "tech", "ilo_occupations", "emp_occ_*.json")):
        for s in json.load(open(f))["series"]["docs"]:
            dm = s["dimensions"]
            emp.setdefault(dm["ref_area"], {})[dm["classif1"]] = dict(zip(s["period"], s["value"]))
    occ, yr = np.full(len(p.codes), np.nan), np.full(len(p.codes), np.nan)
    for i, c in enumerate(p.codes):
        e = emp.get(c, {})
        tot = e.get("OCU_DETAILS_TOTAL", {})
        ys = [y for y in tot if int(y) <= TARGET_YEAR and all(str(y) in e.get(f"OCU_DETAILS_{g}", {}) for g in (1, 2, 3, 4))]
        ys = [y for y in ys if isinstance(tot[y], (int, float)) and tot[y] > 0]
        if ys:
            y = max(ys, key=int)
            occ[i] = sum(e[f"OCU_DETAILS_{g}"][y] for g in (1, 2, 3, 4)) / tot[y]
            yr[i] = int(y)
    # the readiness-rank cognitive share it replaces: uniform on 0.3..0.8 across countries -> mean 0.55, sd 0.144
    ok = np.isfinite(occ)
    z = (occ - occ[ok].mean()) / occ[ok].std()
    cog = np.clip(0.55 + 0.5 / np.sqrt(12) * z, 0.2, 0.9)
    np.savez_compressed(os.path.join(ROOT, "data", "ai_exposure.npz"), codes=np.array(p.codes), occ14=occ, year=yr, cog_share=cog)
    print(f"{ok.sum()} of {len(p.codes)} countries; ISCO 1-4 share: median {np.nanmedian(occ):.2f}, range {np.nanmin(occ):.2f}-{np.nanmax(occ):.2f}")
    for c in ("USA", "GBR", "DEU", "JPN", "KOR", "CHN", "IND", "PHL", "BRA", "NGA", "ETH", "BGD", "VNM", "SAU", "RUS"):
        if c in p.codes:
            i = list(p.codes).index(c)
            print(f"  {c} {occ[i]:.2f} ({int(yr[i]) if np.isfinite(yr[i]) else '-'}) -> cognitive share {cog[i]:.2f}")


if __name__ == "__main__":
    main()
