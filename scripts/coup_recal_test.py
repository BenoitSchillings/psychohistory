"""Five-year coup risk recalibrated from earlier backtests (third debate, 9 November 2026: the coup model forecasts too
high in every probability band of outputs/calibration.csv; an earlier Platt scaling fitted once on development years
made fresh forecasts worse, docs/methodology.md).

Rolling and honest: the recalibration used at origin O is fitted only on backtest forecasts of origins O' <= O - 10,
whose 5-year outcomes (years O'..O'+4, each looking 5 years ahead) were all known before O. Variants:
  shift   one log-odds intercept (calibration in the large)
  platt   intercept and slope in log-odds
  iso     isotonic regression
Pre-registered (before the run): adopt "shift" for the February 2027 release if, pooled over origins 1990-2015, Brier
AND log loss both improve on the raw ensemble and at least 4 of the 6 origins improve on Brier. "platt" and "iso" are
reported, adopted only if they beat "shift" on both pooled scores.

python -P scripts/coup_recal_test.py -> outputs/coup_recal_test.json"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
sg = lambda z: 1 / (1 + np.exp(-z))


def fit(kind, p, y):
    z = lg(p)
    if kind == "shift":
        lo, hi = -4.0, 4.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if sg(z + mid).mean() < y.mean() else (lo, mid)
        c = (lo + hi) / 2
        return lambda q: sg(lg(q) + c)
    if kind == "platt":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=1e6).fit(z[:, None], y)
        return lambda q: m.predict_proba(lg(q)[:, None])[:, 1]
    from sklearn.isotonic import IsotonicRegression
    m = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
    return lambda q: m.predict(q)


def sc(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(np.mean((p - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    d = pd.read_parquet(os.path.join(ROOT, "outputs", "backtest_predictions.parquet"))
    d = d[(d.task == "coup_5y") & (d.model == "Seldon ensemble")].dropna(subset=["p", "y"])
    res, pooled = {"by_origin": {}}, {k: ([], []) for k in ("raw", "shift", "platt", "iso")}
    for O in range(1990, 2016, 5):
        tr, te = d[d.origin <= O - 10], d[d.origin == O]
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
    best = [k for k in ("platt", "iso") if P[k]["brier"] < P["shift"]["brier"] and P[k]["logloss"] < P["shift"]["logloss"]]
    res["better_than_shift"] = best
    json.dump(res, open(os.path.join(ROOT, "outputs", "coup_recal_test.json"), "w"), indent=1)
    for k, v in P.items():
        print(f"{k:6s} Brier {v['brier']:.5f}  log loss {v['logloss']:.5f}")
    for O, r in B.items():
        print(O, f"mean p {r['mean_p']:.3f} y {r['mean_y']:.3f}", {k: round(r[k]["brier"], 5) for k in pooled})
    print("shift wins", wins, "of 6; adopt shift:", res["adopt_shift"], "; better than shift:", best)


if __name__ == "__main__":
    main()
