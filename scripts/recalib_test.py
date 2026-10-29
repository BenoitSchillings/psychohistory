"""Does recalibrating on the most recent out-of-sample years fix coup over-prediction and collapse over-confidence?
Compares plain gbm2 with RecentCalibrated(gbm2, window) per task on dev and held-out origins (Brier, lower is better)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from psychohistory.pipeline import prepare
from psychohistory.evaluate import evaluate, RecentCalibrated, DEV_ORIGINS, LOCK_ORIGINS
from psychohistory.features import TASKS
from exp import FACT
p, base, X, Yl = prepare()
for name, orig in [("dev", DEV_ORIGINS), ("held-out", LOCK_ORIGINS)]:
    res = {}
    for lab, fac in [("plain", FACT["gbm2"]), ("recal8", lambda: RecentCalibrated(FACT["gbm2"], window=8)),
                     ("recal12", lambda: RecentCalibrated(FACT["gbm2"], window=12))]:
        _, out, _ = evaluate(p, X, Yl, fac, orig)
        res[lab] = out
    for t in ("coup_5y", "econ_collapse_5y", "civil_conflict_5y", "autocratization_5y"):
        print(name, f"{t:20s}", {lab: (round(o[t]["auc"], 3), round(o[t]["bss"], 3)) for lab, o in res.items()}, flush=True)
