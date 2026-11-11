"""Are the relationships the forecast rests on stable across the end of the Cold War? (11 November 2026)

Lucas (1976): fitted coefficients describe behaviour under the regime they were estimated in. The risk models and the
simulator's equations are fitted on 1946-2024 as one regime and run to 2065. This is a diagnostic, written and
committed before its run; it adopts nothing.

Design. Four one-year outcomes: civil-war onset, war ending, coup attempt (logistic) and next-year growth of GDP per
person (linear). Twelve predictors fixed here, standardised on the whole sample:
  log_gdp, growth_5y, democracy, democracy_sq, log_pop, youth_share, log_years_since_conflict, nb_conflict,
  coup_recent, regime_military, log_oil_pc, urban
Each equation is fitted separately on outcome years 1950-1989 and 1990-2024 (ridge penalty C = 1 / lambda = 1).
Standard errors: 300 bootstrap resamples of countries. z = (b_late - b_early) / sqrt(se_early^2 + se_late^2).

Pre-registered reading.
  An equation is called unstable if more than 3 of its 12 slopes have |z| > 2 (about 0.6 expected by chance), or if its
  intercept (the level at average predictors) has |z| > 2.
  Any slope with |z| > 3 is named.
  Consequence: stated in the methodology's limits with the names; no model change from this test alone. A named slope
  that is also a scenario lever becomes a candidate for a wider sampled range, under the adopt rule.
python -P scripts/coef_stability_test.py -> outputs/coef_stability.json"""
import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
warnings.simplefilter("ignore")
from psychohistory.features import FEATURES, TASKS      # noqa: E402
from psychohistory.pipeline import prepare             # noqa: E402

COLS = ["log_gdp", "growth_5y", "democracy", "democracy_sq", "log_pop", "youth_share", "log_years_since_conflict",
        "nb_conflict", "coup_recent", "regime_military", "log_oil_pc", "urban"]
p, base, X, Yl = prepare()
C, Yn, _ = X.shape
years = np.broadcast_to(np.asarray(p.years)[None], (C, Yn))
ci = np.broadcast_to(np.arange(C)[:, None], (C, Yn))
Z = X[..., [FEATURES.index(c) for c in COLS]].astype(float)
g_next = np.full((C, Yn), np.nan)
g_next[:, :-1] = base["lg"][:, 1:] - base["lg"][:, :-1]
targets = {"onset": Yl[..., TASKS.index("onset_1y")], "ending": Yl[..., TASKS.index("termination_1y")],
           "coup": Yl[..., TASKS.index("coup_1y")], "growth": g_next}
rng = np.random.default_rng(0)
out = {}
for name, y in targets.items():
    ok = np.isfinite(y) & np.isfinite(Z).all(-1) & (years + 1 >= 1950) & (years + 1 <= 2024)
    mu, sd = Z[ok].mean(0), Z[ok].std(0) + 1e-9

    def fit(rows):
        A, b = (Z[rows] - mu) / sd, y[rows]
        if name == "growth":
            m = Ridge(alpha=1.0).fit(A, b)
            return np.r_[m.intercept_, m.coef_]
        m = LogisticRegression(C=1.0, max_iter=500).fit(A, b)
        return np.r_[m.intercept_[0], m.coef_[0]]
    res = {}
    for era, sel in (("early", ok & (years + 1 <= 1989)), ("late", ok & (years + 1 >= 1990))):
        b = fit(sel)
        boots = []
        for _ in range(300):
            cs = rng.integers(0, C, C)
            w = np.zeros((C, Yn), bool)
            idx = np.concatenate([np.flatnonzero(sel[c]) + c * Yn for c in cs])
            A_, y_ = ((Z.reshape(C * Yn, -1)[idx] - mu) / sd), y.reshape(-1)[idx]
            if name != "growth" and (y_.sum() < 5):
                continue
            if name == "growth":
                m = Ridge(alpha=1.0).fit(A_, y_); boots.append(np.r_[m.intercept_, m.coef_])
            else:
                m = LogisticRegression(C=1.0, max_iter=500).fit(A_, y_); boots.append(np.r_[m.intercept_[0], m.coef_[0]])
        res[era] = (b, np.std(boots, 0), int(sel.sum()), float(np.nansum(y[sel])) if name != "growth" else None)
    z = (res["late"][0] - res["early"][0]) / np.sqrt(res["early"][1] ** 2 + res["late"][1] ** 2)
    names = ["(level)"] + COLS
    out[name] = {"n_early": res["early"][2], "n_late": res["late"][2], "events_early": res["early"][3], "events_late": res["late"][3],
                 "coef": {n_: {"early": round(float(a), 3), "late": round(float(b_), 3), "z": round(float(z_), 2)}
                          for n_, a, b_, z_ in zip(names, res["early"][0], res["late"][0], z)},
                 "slopes_beyond_2": int((np.abs(z[1:]) > 2).sum()), "level_z": round(float(z[0]), 2),
                 "named_beyond_3": [n_ for n_, z_ in zip(names[1:], z[1:]) if abs(z_) > 3],
                 "unstable": bool((np.abs(z[1:]) > 2).sum() > 3 or abs(z[0]) > 2)}
    print(name, {k: out[name][k] for k in ("n_early", "n_late", "slopes_beyond_2", "level_z", "named_beyond_3", "unstable")})
    print("   ", {n_: v["z"] for n_, v in out[name]["coef"].items() if abs(v["z"]) > 2})
json.dump(out, open(os.path.join(ROOT, "outputs", "coef_stability.json"), "w"), indent=1)
