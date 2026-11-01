"""What drives the spread of the 2065 outcomes? From outputs/uncertainty_sims.npz (per world: every sampled assumption
and the outcomes), fit a gradient-boosted surrogate of each outcome on all assumptions (5-fold cross-validated R^2):
- "assumptions, all": share of the variance the assumptions explain together; the rest is luck inside the simulator
  (shocks, wars, coups, pandemics) that no amount of learning about the assumptions would remove;
- per assumption, first-order share (R^2 of a surrogate on that assumption alone) and total share (drop in the full
  surrogate's R^2 when the assumption is left out, so it includes its interactions);
- interactions: full R^2 minus the sum of first-order shares.
The assumptions are drawn independently (except the robotics year, drawn with the AI year), so these are Sobol-style
indices estimated with a surrogate. python -P scripts/uncertainty_attribution.py -> outputs/uncertainty_attribution.json"""
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
P = {k[2:]: z[k] for k in z.files if k.startswith("p_")}
P = {k: v for k, v in P.items() if np.nanstd(v) > 0}
names = sorted(P)
Xa = np.column_stack([P[k] for k in names]).astype(np.float32)
Y = {k[2:]: z[k] for k in z.files if k.startswith("y_")}
PRM = dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, verbose=-1)


def cv_r2(X, y, binary):
    prm = PRM if X.shape[1] > 1 else {**PRM, "n_estimators": 120, "num_leaves": 7}
    pred = np.zeros_like(y, dtype=float)
    for a, b in KFold(5, shuffle=True, random_state=0).split(X):
        m = (lgb.LGBMClassifier if binary else lgb.LGBMRegressor)(**prm).fit(X[a], y[a])
        pred[b] = m.predict_proba(X[b])[:, 1] if binary else m.predict(X[b])
    return max(0.0, 1 - np.mean((y - pred) ** 2) / np.var(y))


# correlated or jointly interpretable assumptions, credited as groups (total share: the drop in R^2 when the whole
# group is left out; the robotics year is drawn correlated with the AI year, code review 2026-10-28)
GROUPS = {"AI and robot timing": ["ai_arrival", "robot_arrival", "robot_lag", "arrival_sd"],
          "climate": ["clim_response", "dmg_family", "climate_damage", "amoc_p", "amoc_hit", "amoc_eu_loss",
                      "amoc_eu_yield", "amoc_rain", "slr_ice", "slr_ice_mult", "slr_mult", "slr_cost", "storm_trend",
                      "storm_loss", "harvest_amp", "dry_area_elast", "clim_conflict", "clim_migr"],
          "nuclear": ["nuclear_war_factor", "nuc_use_both", "nuc_use_one", "nuc_escalate", "nuc_gdp_large", "nuc_beta"],
          "AI rent capture": ["firm_capture", "clawback"],
          "technology": ["tech_" + n for n in __import__("psychohistory.tech", fromlist=["TECHS"]).TECHS] + ["space_war", "sat_open", "orbit_dc"]}
out = {}
for k, y in Y.items():
    ok = np.isfinite(y) & np.isfinite(Xa).all(1)
    X_, y_ = Xa[ok], y[ok].astype(float)
    if np.var(y_) == 0:
        continue
    binary = set(np.unique(y_)) <= {0.0, 1.0}
    full = cv_r2(X_, y_, binary)
    first = {n: cv_r2(X_[:, [j]], y_, binary) for j, n in enumerate(names)}
    top = sorted(first, key=lambda n: -first[n])[:12]
    total = {n: max(0.0, full - cv_r2(np.delete(X_, names.index(n), 1), y_, binary)) for n in top}
    out[k] = {"n": int(ok.sum()), "mean": round(float(y_.mean()), 4), "assumptions_all": round(full, 4),
              "luck": round(1 - full, 4), "interactions": round(max(0.0, full - sum(first.values())), 4),
              "first_order": {n: round(first[n], 4) for n in top},
              "total": {n: round(total[n], 4) for n in top},
              "groups": {g: round(max(0.0, full - cv_r2(np.delete(X_, [names.index(n) for n in m if n in names], 1), y_, binary)), 4)
                         for g, m in GROUPS.items() if any(n in names for n in m)}}
    print(k, json.dumps(out[k]), flush=True)
json.dump(out, open(ROOT / "outputs" / "uncertainty_attribution.json", "w"), indent=1)
