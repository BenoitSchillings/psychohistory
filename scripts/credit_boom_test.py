"""Do private credit booms predict banking-crisis starts beyond what the simulator's crisis hazard uses?
Banking-crisis starts (the panel's fin_crisis, Laeven-Valencia / GMD) on the 5-year rise in private debt / GDP (IMF
Global Debt Database, PVD_LS), its household and corporate parts where available, with last year's growth collapse
(the simulator's crisis_given_growth) and short-term debt as controls. Logit, clustered by country; fit 1975-2004,
scored out of sample 2005-2024 (log loss and AUC against the same model without credit), and the same with year effects
in-sample (identification from which countries boomed, not from global waves).
python -P scripts/credit_boom_test.py -> outputs/credit_boom_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import log_loss, roc_auc_score

from psychohistory.data import ROOT, get_panel
from psychohistory.features import base_arrays
from candidate_screen import gdd


def main():
    p = get_panel()
    b = base_arrays(p)
    cand = gdd(p)
    Y = np.asarray(p.years)
    fc = b["fin_crisis"]
    start = (np.nan_to_num(fc) > 0) & (np.concatenate([np.ones((fc.shape[0], 1)), np.nan_to_num(fc[:, :-1], nan=1)], 1) == 0)
    g = np.diff(b["lg"], axis=1, prepend=np.nan)
    rows = []
    for i, c in enumerate(p.codes):
        for j, y in enumerate(Y):
            if not (1975 <= y <= 2024) or not np.isfinite(fc[i, j]) or (j and np.nan_to_num(fc[i, j - 1]) > 0):
                continue                                  # at risk: not already in a crisis last year
            rows.append({"code": c, "year": int(y), "start": float(start[i, j]),
                         "collapse_prev": float(np.nan_to_num(g[i, j - 1]) < -0.05) if j else 0.0,
                         "boom5": cand["gdd_PVD_LS_chg5"][i, j - 1] if j else np.nan,
                         "nfc5": cand["gdd_NFC_LS_chg5"][i, j - 1] if j else np.nan,
                         "lg": b["lg"][i, j - 1] if j else np.nan})
    d = pd.DataFrame(rows).dropna(subset=["boom5", "lg"])
    d["boom5"] /= 100
    out = {"n": len(d), "starts": int(d.start.sum())}
    tr, te = d[d.year <= 2004], d[d.year >= 2005]
    res = {}
    for name, cols in (("base", ["collapse_prev", "lg"]), ("credit", ["collapse_prev", "lg", "boom5"])):
        m = sm.Logit(tr.start, sm.add_constant(tr[cols])).fit(disp=0, cov_type="cluster", cov_kwds={"groups": tr.code})
        pr = m.predict(sm.add_constant(te[cols]))
        res[name] = {"coef": {k: round(float(v), 3) for k, v in m.params.items()},
                     "t": {k: round(float(v), 2) for k, v in m.tvalues.items()},
                     "test_logloss": round(float(log_loss(te.start, pr)), 5), "test_auc": round(float(roc_auc_score(te.start, pr)), 4)}
    out["fit_1975_2004_test_2005_2024"] = res
    # in-sample with year effects, whole period
    X = pd.concat([d[["collapse_prev", "lg", "boom5"]], pd.get_dummies(d.year, prefix="y", drop_first=True, dtype=float)], axis=1)
    keep = X.columns[(X != 0).any()]
    try:
        m = sm.Logit(d.start, sm.add_constant(X[keep])).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.code}, maxiter=200)
        out["year_effects"] = {"boom5": round(float(m.params["boom5"]), 3), "t": round(float(m.tvalues["boom5"]), 2)}
    except Exception as e:
        out["year_effects"] = str(e)
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "outputs", "credit_boom_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
