"""Cascades: do events cluster in particular years more than the risk models' independent country probabilities allow?
For each one-year task, the yearly world total of events against the sum of out-of-sample probabilities (tuned GBM,
development origins 1975-2005 and lockbox 2010/2015, years origin..origin+4): dispersion = sum((obs - exp)^2) /
sum(sum p(1-p)). 1 = independent countries; > 1 = common shocks or cascades (the simulator's mood and year shocks are
meant to supply them). Also the largest yearly surprises.
python -P scripts/cascade_dispersion.py -> outputs/new_data/cascade_dispersion.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from psychohistory import models as M
from psychohistory.backtest import flatten
from psychohistory.data import ROOT
from psychohistory.evaluate import evaluate, DEV_ORIGINS, LOCK_ORIGINS
from psychohistory.features import TASKS
from psychohistory.pipeline import prepare
from exp import Wrap, TUNED_GBM


def main():
    p, base, X, Yl = prepare()
    Xf, Yf, ci, year = flatten(p, X, Yl)
    rows = []
    for origins in (DEV_ORIGINS, LOCK_ORIGINS):
        _, _, preds = evaluate(p, X, Yl, lambda: Wrap(M.GBMModel(TUNED_GBM)), origins)
        for T, y_, pr, _, c_ in preds:
            te = (year >= T) & (year < T + 5)
            rows.append((T, year[te], y_, pr))
    out = {}
    for task in ("onset_1y", "coup_1y", "termination_1y"):
        k = TASKS.index(task)
        per = {}
        for T, yr, y_, pr in rows:
            for y in np.unique(yr):
                s = (yr == y) & np.isfinite(y_[:, k])
                o, e, v = per.get((T, int(y)), (0.0, 0.0, 0.0))
                per[(T, int(y))] = (o + y_[s, k].sum(), e + pr[s, k].sum(), v + (pr[s, k] * (1 - pr[s, k])).sum())
        o, e, v = map(np.array, zip(*per.values()))
        z = (o - e) / np.sqrt(v)
        keys = list(per)
        top = sorted(range(len(z)), key=lambda i: -abs(z[i]))[:6]
        out[task] = {"dispersion": round(float(np.sum((o - e) ** 2) / np.sum(v)), 2), "n_years": len(o),
                     "obs_over_exp": round(float(o.sum() / e.sum()), 3),
                     "largest": [{"origin": keys[i][0], "year": keys[i][1], "obs": int(o[i]), "exp": round(float(e[i]), 1),
                                  "z": round(float(z[i]), 1)} for i in top]}
        print(task, json.dumps(out[task]), flush=True)
    json.dump(out, open(os.path.join(ROOT, "outputs", "new_data", "cascade_dispersion.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
