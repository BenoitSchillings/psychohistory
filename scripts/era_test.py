"""Cross-era test of the pair war model (second debate, 6 November 2026): every test of the simulator lives inside
1946-2025, so its great-power war numbers assume the post-1945 order persists. Fit the pair model on one era and
score it on the other, both ways, on the one outcome coded consistently since 1816: a militarised dispute with more
than 25 deaths between the pair within five years (fatal_mid_5y, Gibler-Miller-Little MIDs, 1816-2010).

Era A: rows 1816-1938 (windows to 1943). Era B: rows 1946-2005 (windows to 2010).
Reported for all pairs and for pairs of two major powers: AUC, and the ratio of observed to predicted events (above 1:
the other era had more fighting than this era's model expects). The ratio for major-power pairs, fitted on B and
scored on A, is what an "era" assumption would scale (a world where the pre-1945 regime returns).
python -P scripts/era_test.py -> outputs/era_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn.metrics import roc_auc_score

from psychohistory import interstate as I

# features defined in both eras (no UCDP-only or event-data columns)
FE = [f for f in I.FEATS if f not in ("ongoing", "war_now", "conflict_run_years")]


def score(m, d):
    X, y = d[FE].to_numpy(float), d["fatal_mid_5y"].to_numpy(float)
    p = m.predict(X)
    r = {"rows": int(len(d)), "events": int(y.sum()), "expected": round(float(p.sum()), 1),
         "obs_over_pred": round(float(y.sum() / max(p.sum(), 1e-9)), 3),
         "auc": round(float(roc_auc_score(y, p)), 3) if 0 < y.sum() < len(y) else None}
    mj = (d.major_pair.to_numpy() > 0)
    if mj.any() and 0 < y[mj].sum():
        r["major_pairs"] = {"rows": int(mj.sum()), "events": int(y[mj].sum()), "expected": round(float(p[mj].sum()), 2),
                            "obs_over_pred": round(float(y[mj].sum() / max(p[mj].sum(), 1e-9)), 3)}
    return r


def main():
    df = I.build(False)
    d = df[df.fatal_mid_5y.notna()].copy()
    A = d[(d.year >= 1816) & (d.year <= 1938)]
    B = d[(d.year >= 1946) & (d.year <= 2005)]
    out = {"features": FE}
    for name, tr, te in (("fit_1816_1938_score_1946_2005", A, B), ("fit_1946_2005_score_1816_1938", B, A)):
        m = I.DyadModel().fit(tr[FE].to_numpy(float), tr["fatal_mid_5y"].to_numpy(float))
        out[name] = {"in_sample": score(m, tr), "other_era": score(m, te)}
        print(name, json.dumps(out[name]), flush=True)
    # base rates for comparison
    for nm, e in (("1816_1938", A), ("1946_2005", B)):
        mj = e.major_pair > 0
        out[f"rate_{nm}"] = {"all": round(float(e.fatal_mid_5y.mean()), 4), "major_pairs": round(float(e[mj].fatal_mid_5y.mean()), 4)}
    print({k: v for k, v in out.items() if k.startswith("rate")})
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "era_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
