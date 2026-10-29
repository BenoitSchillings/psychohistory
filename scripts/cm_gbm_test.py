"""LightGBM settings for the monthly country model, with the production blend (LightGBM + neural, residual target).
Dev windows 2010-2017 only. python -P scripts/cm_gbm_test.py TAG '{"num_leaves": 15}' -> outputs/cm_gbm_test_<TAG>.json"""
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
tag, params = sys.argv[1], json.loads(sys.argv[2])
truth = CF.deaths(CF.mid(2023, 12))
by_year = {}
for y in range(2010, 2018):
    months = [CF.mid(y, m) for m in range(1, 13)]
    dr, _ = CF.train_and_forecast(CF.mid(y - 1, 10), months, params=params)
    by_year[y] = float(np.mean([crps_ensemble(truth[:, m], np.clip(dr[m], 0, None)).mean() for m in months]))
    print(y, round(by_year[y], 3), flush=True)
res = {"params": params, "dev_crps": round(float(np.mean(list(by_year.values()))), 4), "by_year": by_year}
print(res["dev_crps"])
json.dump(res, open(os.path.join(ROOT, "outputs", f"cm_gbm_test_{tag}.json"), "w"), indent=1)
