"""A data family from scripts/new_data_test.py (default: warstructure) in the production ensemble (SeldonEnsemble, all families,
4 members for speed), not just the tuned GBM: rolling origins 1975-2005 (development) and 2010/2015 (lockbox), AUC and
Brier skill per task (skill against the training base rate), with and without the columns.
python -P scripts/warstructure_ensemble_test.py [NAME] -> outputs/new_data/<NAME>_ensemble.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.metrics import roc_auc_score

from psychohistory import models as M
from psychohistory.backtest import flatten, train_mask, ALL_FAMILIES
from psychohistory.data import ROOT
from psychohistory.evaluate import DEV_ORIGINS, LOCK_ORIGINS
from psychohistory.features import TASKS
from psychohistory.pipeline import prepare
from new_data_test import SOURCES


def run(p, X, Yl, origins):
    Xf, Yf, ci, year = flatten(p, X, Yl)
    P, Y, B = [], [], []
    for T in origins:
        Ytr, tr = train_mask(Yf, year, T, first_year=1860)      # (as production: the G60 family trains from 1860)
        m = M.SeldonEnsemble(M.load_combine(), members=4, families=ALL_FAMILIES).fit(Xf[tr], Ytr[tr], ci[tr], year[tr])
        te = (year >= T) & (year < T + 5)
        P.append(m.predict(Xf[te]))
        Y.append(Yf[te])
        B.append(np.repeat(np.nanmean(Ytr[tr], 0)[None], te.sum(), 0))
        print(" origin", T, flush=True)
    P, Y, B = map(np.concatenate, (P, Y, B))
    out = {}
    for k, t in enumerate(TASKS):
        ok = np.isfinite(Y[:, k]) & np.isfinite(P[:, k])
        y, pr, b = Y[ok, k], P[ok, k], B[ok, k]
        out[t] = {"auc": round(float(roc_auc_score(y, pr)), 4), "bss": round(float(1 - np.mean((pr - y) ** 2) / np.mean((b - y) ** 2)), 4),
                  "n": int(ok.sum()), "events": int(y.sum())}
    return out


def main():
    p, base, X, Yl = prepare()
    name_ = sys.argv[1] if len(sys.argv) > 1 else "warstructure"
    arrs = SOURCES[name_](p)
    Xa = np.concatenate([X, np.stack(list(arrs.values()), -1).astype(np.float32)], -1)
    res = {}
    for name, XX in (("base", X), (name_, Xa)):
        for split, origins in (("dev", DEV_ORIGINS), ("lock", LOCK_ORIGINS)):
            res[f"{name}:{split}"] = run(p, XX, Yl, origins)
            print(name, split, json.dumps(res[f"{name}:{split}"]), flush=True)
    json.dump(res, open(ROOT / "outputs" / "new_data" / f"{name_}_ensemble.json", "w"), indent=1)


if __name__ == "__main__":
    main()
