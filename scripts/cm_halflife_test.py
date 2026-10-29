"""Recency weighting of the monthly country model's training months (cm_fatal.HALFLIFE, years), with the production
LightGBM + neural blend. Dev windows 2010-2017 only (forecast from October of the year before, final GED).
python -P scripts/cm_halflife_test.py HL   (HL = none or years) -> outputs/cm_halflife_test_<HL>.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from psychohistory import cm_fatal as CF
from predcomp_eval import crps_ensemble

ROOT = os.path.join(os.path.dirname(__file__), "..")
hl = sys.argv[1]
CF.HALFLIFE = None if hl == "none" else float(hl)
truth = CF.deaths(CF.mid(2023, 12))
by_year = {}
for y in range(2010, 2018):
    months = [CF.mid(y, m) for m in range(1, 13)]
    dr, _ = CF.train_and_forecast(CF.mid(y - 1, 10), months)
    by_year[y] = float(np.mean([crps_ensemble(truth[:, m], np.clip(dr[m], 0, None)).mean() for m in months]))
    print(y, round(by_year[y], 3), flush=True)
res = {"halflife": CF.HALFLIFE, "dev_crps": round(float(np.mean(list(by_year.values()))), 4), "by_year": by_year}
print(res["dev_crps"])
json.dump(res, open(os.path.join(ROOT, "outputs", f"cm_halflife_test_{hl}.json"), "w"), indent=1)
