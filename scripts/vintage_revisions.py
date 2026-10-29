"""How do UCDP Candidate monthly counts (first release, the month after) compare with final GED 26.1, per country-month,
2021-2025? Writes outputs/vintage_revisions.csv (country_id, month_id, candidate, final) and prints the summary.
python -P scripts/vintage_revisions.py"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

from psychohistory import cm_fatal as CF
from psychohistory.data import ROOT

fin = CF._final()
rows = []
for y in range(2021, 2026):
    for mo in range(1, 13):
        f = CF.O / "ucdp" / "candidate_monthly" / f"GEDEvent_v{y % 100}_0_{mo}.csv"
        c = CF._agg(f, (y, mo))
        m = CF.mid(y, mo)
        fm = fin[fin.index.get_level_values(1) == m]
        keys = set(c.index) | set(fm.index)
        for k in keys:
            rows.append({"country_id": k[0], "month_id": m, "year": y, "candidate": float(c.get(k, 0)), "final": float(fm.get(k, 0))})
d = pd.DataFrame(rows)
d.to_csv(ROOT / "outputs" / "vintage_revisions.csv", index=False)
print("country-months with any deaths:", len(d))
for y, g in d.groupby("year"):
    print(y, "total cand", int(g.candidate.sum()), "final", int(g.final.sum()), "ratio", round(g.final.sum() / g.candidate.sum(), 2),
          "| 25+ months: cand", int((g.candidate >= 25).sum()), "final", int((g.final >= 25).sum()))
cs = CF.countries().set_index("gwcode")
t = d.groupby(["country_id", "year"])[["candidate", "final"]].sum().reset_index()
t["iso3"] = t.country_id.map(cs.iso3)
t["ratio"] = (t.final + 25) / (t.candidate + 25)
big = t[t.final > 2000].pivot(index="iso3", columns="year", values="ratio").round(2)
print(big.to_string())
