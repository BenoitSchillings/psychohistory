"""Calibration of the 5-year crisis forecasts at the top of the risk ranking (dev + lockbox origins)."""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from psychohistory.pipeline import prepare
from psychohistory.evaluate import evaluate, HEADLINE, DEV_ORIGINS, LOCK_ORIGINS
from psychohistory.features import TASKS
from exp import FACT
p, base, X, Yl = prepare()
_, _, preds = evaluate(p, X, Yl, FACT["gbm2"], DEV_ORIGINS + LOCK_ORIGINS)
cat = lambda i: np.concatenate([a[i] for a in preds])
y, pr = cat(1), cat(2)
res = {}
for t in HEADLINE:
    k = TASKS.index(t); ok = np.isfinite(y[:, k]) & np.isfinite(pr[:, k])
    yy, pp = y[ok, k], pr[ok, k]
    rows = []
    for lo, hi in [(0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .7), (.7, 1.01)]:
        m = (pp >= lo) & (pp < hi)
        if m.sum() >= 10:
            rows.append((f"{lo:.2f}-{hi:.2f}", int(m.sum()), round(float(pp[m].mean()), 3), round(float(yy[m].mean()), 3)))
    top = pp >= np.quantile(pp, .9)
    res[t] = {"bins": rows, "top_decile_pred": float(pp[top].mean()), "top_decile_obs": float(yy[top].mean())}
    print(t, "top decile: predicted %.3f observed %.3f" % (pp[top].mean(), yy[top].mean()))
    for r in rows: print("   ", r)
json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "calibration_top.json"), "w"), indent=1)
