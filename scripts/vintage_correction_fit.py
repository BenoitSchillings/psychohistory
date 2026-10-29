"""Choose the damping w of the vintage correction: candidate counts of year Y are multiplied, per country, by
exp(w * r_c), r_c = log((final + 25) / (candidate + 25)) summed over the country's twelve months of year Y-1 (the
revision last seen). w is picked by the squared error of log(1 + deaths) against final GED on target years 2022-2023 only
(the live challenge window uses 2024 candidates; 2024-2025 are reported as a check, not used).
python -P scripts/vintage_correction_fit.py -> outputs/vintage_correction_fit.json"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

from psychohistory.data import ROOT

d = pd.read_csv(ROOT / "outputs" / "vintage_revisions.csv")
yr = d.groupby(["country_id", "year"])[["candidate", "final"]].sum()
r = np.log((yr.final + 25) / (yr.candidate + 25))
out = {}
for w in (0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
    res = {}
    for Y in range(2022, 2026):
        g = d[d.year == Y]
        rc = np.array([r.get((c, Y - 1), 0.0) for c in g.country_id])
        corr = g.candidate * np.exp(w * rc)
        e = lambda x: float(np.mean((np.log1p(x) - np.log1p(g.final)) ** 2))
        res[Y] = round(e(corr), 4)
    out[str(w)] = {**{str(k): v for k, v in res.items()}, "fit_2022_2023": round((res[2022] + res[2023]) / 2, 4)}
    print(w, out[str(w)])
best = min(out, key=lambda k: out[k]["fit_2022_2023"])
out["chosen_w"] = float(best)
print("chosen", best)
json.dump(out, open(ROOT / "outputs" / "vintage_correction_fit.json", "w"), indent=1)
