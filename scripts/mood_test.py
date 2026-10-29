"""Mood: does life satisfaction (Gallup World Poll Cantril ladder via the World Happiness Report and Our World in Data,
2011-2025) explain what the risk models get wrong? Too short for the development origins, so only an association
with the tuned GBM's out-of-sample errors at origins 2010, 2015 and 2020: logit(y ~ logit(p) + z), errors clustered
by country (scripts/residual_screen.py). Candidates: the ladder, its 3-year change, and the change against GDP growth
(falling satisfaction while the economy grows: Egypt and Tunisia before 2011).
python -P scripts/mood_test.py -> outputs/new_data/mood.json"""
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

from psychohistory import models as M
from psychohistory.backtest import flatten
from psychohistory.data import RAW, ROOT
from psychohistory.evaluate import evaluate
from psychohistory.features import TASKS
from psychohistory.pipeline import prepare
from exp import Wrap, TUNED_GBM


def main():
    p, base, X, Yl = prepare()
    d = pd.read_csv(RAW / "newdata" / "mood" / "ladder.csv")
    lad = d.pivot_table(index="code", columns="year", values="cantril_ladder_score").reindex(index=list(p.codes), columns=list(p.years)).to_numpy(float)
    lad = pd.DataFrame(lad).T.ffill(limit=1).T.to_numpy(float)
    lag3 = lambda a: np.concatenate([np.full((a.shape[0], 3), np.nan), a[:, :-3]], 1)
    dl, dg = lad - lag3(lad), base["lg"] - lag3(base["lg"])
    z = lambda a: (a - np.nanmean(a)) / np.nanstd(a)
    cand = {"ladder": lad, "ladder_change_3y": dl, "ladder_vs_growth_3y": z(dl) - z(dg)}
    _, _, preds = evaluate(p, X, Yl, lambda: Wrap(M.GBMModel(TUNED_GBM)), (2010, 2015, 2020))
    Xf, Yf, ci, year = flatten(p, X, Yl)
    yi = {int(y): i for i, y in enumerate(p.years)}
    Y, P, C, Yr = [], [], [], []
    for T, y_, pr, _, c_ in preds:
        te = (year >= T) & (year < T + 5)
        Y.append(y_), P.append(pr), C.append(c_), Yr.append(year[te])
    Y, P, C, Yr = map(np.concatenate, (Y, P, C, Yr))
    ti = np.array([yi[int(y)] for y in Yr])
    res = {}
    for k, task in enumerate(TASKS):
        for name, arr in cand.items():
            zz = arr[C, ti]
            y, pr = Y[:, k], np.clip(P[:, k], 1e-4, 1 - 1e-4)
            ok = np.isfinite(y) & np.isfinite(zz)
            if ok.sum() < 300 or y[ok].sum() < 15:
                continue
            Z = np.column_stack([np.log(pr[ok] / (1 - pr[ok])), (zz[ok] - zz[ok].mean()) / zz[ok].std()])
            m = sm.Logit(y[ok], sm.add_constant(Z)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": C[ok]})
            res[f"{task}:{name}"] = {"coef": round(float(m.params[2]), 3), "se": round(float(m.bse[2]), 3),
                                     "p": round(float(m.pvalues[2]), 4), "n": int(ok.sum()), "events": int(y[ok].sum())}
            print(f"{task:20s} {name:22s} {res[f'{task}:{name}']}", flush=True)
    json.dump(res, open(os.path.join(ROOT, "outputs", "new_data", "mood.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
