"""Choose the grid model's local tail bound on development windows the challenge does not score (2014-2017, each from
data to October of the year before, final GED). A cell's draws are capped at max(c0, k x the largest monthly death
count in the cell and its 8 neighbours over the 36 months to the origin): the extreme-quantile models put the top 1%
of draws in the thousands for some cells that, with their neighbours, have seen no death in years.
python scripts/pgm_tune.py  -> outputs/pgm_tune.csv
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

from psychohistory import cm_fatal as CF, pgm_fatal as PG
sys.path.insert(0, os.path.dirname(__file__))
from predcomp_eval import crps_ensemble, ignorance, interval

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEV = [2014, 2015, 2016, 2017]
GRID = [(None, None)] + [(k, c0) for k in (2, 5, 20) for c0 in (5, 25, 100)]


def main():
    truth = PG.deaths(CF.mid(2023, 12))
    rows = []
    for y in DEV:
        origin = CF.mid(y - 1, 10)
        months = [CF.mid(y, m) for m in range(1, 13)]
        draws, _, _ = PG.train_and_forecast(origin, months)
        base = PG.local_max(PG.deaths(origin), origin)
        for k, c0 in GRID:
            cr, ig, mi = [], [], []
            for m in months:
                X = draws[m] if k is None else np.minimum(draws[m], np.maximum(c0, k * base)[:, None])
                yv = truth[:, m]
                for i in range(0, len(yv), 20000):
                    cr.append(crps_ensemble(yv[i:i + 20000], X[i:i + 20000]))
                    ig.append(ignorance(yv[i:i + 20000], X[i:i + 20000]))
                    mi.append(interval(yv[i:i + 20000], X[i:i + 20000]))
            r = {"year": y, "k": k, "c0": c0, "crps": float(np.concatenate(cr).mean()), "ign": float(np.concatenate(ig).mean()),
                 "mis": float(np.concatenate(mi).mean())}
            rows.append(r)
            print(r, flush=True)
        del draws
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "outputs", "pgm_tune.csv"), index=False)
    print(df.fillna(-1).groupby(["k", "c0"])[["crps", "ign", "mis"]].mean().sort_values("crps").round(4).to_string())


if __name__ == "__main__":
    main()
