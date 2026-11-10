"""Civil conflict coded by where the deaths occur (data.CONFLICT_CODING "deaths") against UCDP's location field
("location"), third debate, 9 November 2026.

The deaths coding removes 53 of 1,127 conflict-years since 1989 (United States 2002-2019 for the conflict with
al-Qaida, Rwanda's and Uganda's wars abroad, Israel 1989-90 / 1997-99, Burundi) and adds none.

Pre-registered (written before the run): rolling-origin backtests of the risk ensemble, origins 1990-2015 every 5
years (the codings differ only from 1989), 3 members, trained once on each coding's labels and features. Both are
scored on the deaths-coded outcomes (the quantity the forecast is meant to describe: fighting on the country's soil).
Adopt for the February 2027 release if, on that truth, the ensemble's Brier score AND log loss for civil_conflict_5y
both improve, and no other task's Brier score gets worse by more than 1%. Also reported: the same on the location
truth, and the published countries most changed (forecast origin).

python -P scripts/conflict_coding_test.py -> outputs/conflict_coding_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

ORIGINS = range(1990, 2016, 5)


def run(coding):
    from psychohistory import data as D, pipeline as PL
    from psychohistory.backtest import run_backtest
    D.CONFLICT_CODING = coding
    p, base, X, Yl = PL.prepare(False)
    bt = run_backtest(p, X, Yl, origins=ORIGINS, members=3, verbose=True)
    return bt[bt.model == "Seldon ensemble"][["origin", "task", "code", "year", "p", "y"]]


def scores(df):
    out = {}
    for t, g in df.groupby("task"):
        p = np.clip(g.p.to_numpy(float), 1e-4, 1 - 1e-4)
        y = g.y.to_numpy(float)
        out[t] = {"n": int(len(y)), "events": int(y.sum()), "brier": float(np.mean((p - y) ** 2)),
                  "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}
    return out


def main():
    a, b = run("location"), run("deaths")
    k = ["origin", "task", "code", "year"]
    m = a.merge(b, on=k, suffixes=("_loc", "_dth"))
    res = {}
    for truth in ("dth", "loc"):
        res[f"truth_{truth}"] = {"location_model": scores(m.rename(columns={"p_loc": "p", f"y_{truth}": "y"})[["task", "p", "y"]]),
                                 "deaths_model": scores(m.rename(columns={"p_dth": "p", f"y_{truth}": "y"})[["task", "p", "y"]])}
    T = res["truth_dth"]
    rel = {t: {s: T["deaths_model"][t][s] / T["location_model"][t][s] - 1 for s in ("brier", "logloss")} for t in T["location_model"]}
    cc = rel.get("civil_conflict_5y", {})
    worse = [t for t, r in rel.items() if t != "civil_conflict_5y" and r["brier"] > 0.01]
    res["relative_on_deaths_truth"] = rel
    res["adopt"] = bool(cc and cc["brier"] < 0 and cc["logloss"] < 0 and not worse)
    res["worse_tasks"] = worse
    out = os.path.join(os.path.dirname(__file__), "..", "outputs", "conflict_coding_test.json")
    json.dump(res, open(out, "w"), indent=1)
    for t, r in rel.items():
        print(f"{t:22s} Brier {100 * r['brier']:+.2f}%  log loss {100 * r['logloss']:+.2f}%")
    print("adopt:", res["adopt"], "worse:", worse)


if __name__ == "__main__":
    main()
