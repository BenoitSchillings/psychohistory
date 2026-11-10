"""12-month coup risk recalibrated from earlier backtests (10 November 2026). scripts/coup_dated_test.py found the
12-month coup forecast over-confident in the same way as the 5-year one (pipeline.COUP_RECAL, adopted in version 55):
recalibration alone cut log loss 5.9%. That was a side result, not a test; this is the test, with the design of
scripts/coup_recal_test.py.

Rolling and honest: the recalibration used at origin O is fitted only on backtest forecasts of origins O' <= O - 5,
whose 1-year outcomes (years O'..O'+4, each looking one year ahead) were all known before O. Variants shift / platt /
iso as in coup_recal_test.py.
Pre-registered (10 November 2026, before the run): adopt "shift" for the February 2027 release if, pooled over origins
1990-2015, Brier AND log loss both improve on the raw ensemble and at least 4 of the 6 origins improve on Brier.
"platt" and "iso" are reported, adopted only if they beat "shift" on both pooled scores.
python -P scripts/coup1y_recal_test.py -> outputs/coup1y_recal_test.json"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from coup_recal_test import ROOT, fit, sc


def main():
    d = pd.read_parquet(os.path.join(ROOT, "outputs", "backtest_predictions.parquet"))
    d = d[(d.task == "coup_1y") & (d.model == "Seldon ensemble")].dropna(subset=["p", "y"])
    res, pooled = {"by_origin": {}}, {k: ([], []) for k in ("raw", "shift", "platt", "iso")}
    for O in range(1990, 2016, 5):
        tr, te = d[d.origin <= O - 5], d[d.origin == O]
        y = te.y.to_numpy(float)
        row = {"n": int(len(te)), "events": int(y.sum()), "mean_p": float(te.p.mean()), "mean_y": float(y.mean())}
        for k in pooled:
            q = te.p.to_numpy(float) if k == "raw" else fit(k, tr.p.to_numpy(float), tr.y.to_numpy(float))(te.p.to_numpy(float))
            row[k] = dict(zip(("brier", "logloss"), sc(q, y)))
            pooled[k][0].append(q)
            pooled[k][1].append(y)
        res["by_origin"][O] = row
    y = np.concatenate(pooled["raw"][1])
    res["pooled"] = {k: dict(zip(("brier", "logloss"), sc(np.concatenate(v[0]), y))) for k, v in pooled.items()}
    P, B = res["pooled"], res["by_origin"]
    wins = sum(B[O]["shift"]["brier"] < B[O]["raw"]["brier"] for O in B)
    res["shift_wins"] = int(wins)
    res["adopt_shift"] = bool(P["shift"]["brier"] < P["raw"]["brier"] and P["shift"]["logloss"] < P["raw"]["logloss"] and wins >= 4)
    res["better_than_shift"] = [k for k in ("platt", "iso") if P[k]["brier"] < P["shift"]["brier"] and P[k]["logloss"] < P["shift"]["logloss"]]
    json.dump(res, open(os.path.join(ROOT, "outputs", "coup1y_recal_test.json"), "w"), indent=1)
    for k, v in P.items():
        print(f"{k:6s} Brier {v['brier']:.5f}  log loss {v['logloss']:.5f}")
    for O, r in B.items():
        print(O, f"n {r['n']} events {r['events']} mean p {r['mean_p']:.4f} y {r['mean_y']:.4f}", {k: round(r[k]["brier"], 5) for k in pooled})
    print("shift wins", wins, "of 6; adopt shift:", res["adopt_shift"], "; better than shift:", res["better_than_shift"])


if __name__ == "__main__":
    main()
