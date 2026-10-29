"""Companion to uncertainty_attribution.py for yes/no outcomes, where R^2 is small by construction: how far each
outcome's probability moves with the assumptions. Cross-validated surrogate (all assumptions) -> spread of each world's
probability (10th-90th percentile); and for the main drivers, the probability among worlds in the lowest vs highest
fifth of that assumption. python -P scripts/uncertainty_ranges.py -> outputs/uncertainty_ranges.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold

from psychohistory.data import ROOT

z = np.load(ROOT / "outputs" / "uncertainty_sims.npz")
P = {k[2:]: z[k] for k in z.files if k.startswith("p_") and np.nanstd(z[k]) > 0}
names = sorted(P)
Xa = np.column_stack([P[k] for k in names]).astype(np.float32)
att = json.load(open(ROOT / "outputs" / "uncertainty_attribution.json"))
out = {}
for k in z.files:
    if not k.startswith("y_"):
        continue
    y = z[k].astype(float)
    name = k[2:]
    binary = set(np.unique(y)) <= {0.0, 1.0}
    pred = np.zeros_like(y)
    for a, b in KFold(5, shuffle=True, random_state=0).split(Xa):
        m = (lgb.LGBMClassifier if binary else lgb.LGBMRegressor)(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                                                  min_child_samples=300, verbose=-1).fit(Xa[a], y[a])
        pred[b] = m.predict_proba(Xa[b])[:, 1] if binary else m.predict(Xa[b])
    tr = (lambda v: np.exp(v)) if name.startswith("log_world_gdp") else (lambda v: v)
    r = {"overall": round(float(tr(np.mean(y)) if not name.startswith("log_world") else np.exp(np.median(y))), 4),
         "p10_p90_given_assumptions": [round(float(tr(q)), 4) for q in np.quantile(pred, [.1, .9])], "drivers": {}}
    drv = sorted(att.get(name, {}).get("total", {}).items(), key=lambda kv: -kv[1])[:4]
    for d, v in drv:
        if v < 0.002:
            continue
        x = P[d]
        lo, hi = np.quantile(x, [.2, .8])
        f = (lambda s: float(np.mean(y[s]))) if binary else (lambda s: float(tr(np.median(y[s]))))
        r["drivers"][d] = {"low_fifth": round(f(x <= lo), 4), "high_fifth": round(f(x >= hi), 4)}
    out[name] = r
    print(name, json.dumps(r), flush=True)
json.dump(out, open(ROOT / "outputs" / "uncertainty_ranges.json", "w"), indent=1)
