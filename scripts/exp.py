"""python scripts/exp.py NAME [--drop group,group] [--model ensemble|gbm|logit|psychonet] [--lock]"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from psychohistory.pipeline import prepare
from psychohistory.features import FEATURES, GROUPS
from psychohistory import models as M
from psychohistory.evaluate import evaluate, log, DEV_ORIGINS, LOCK_ORIGINS, RecentCalibrated

class Wrap:
    def __init__(self, m, groups=False): self.m, self.g = m, groups
    def fit(self, X, Y, g):
        (self.m.fit(X, Y, g) if self.g else self.m.fit(X, Y)); return self
    def predict(self, X): return self.m.predict(X)

import json
TUNED_GBM = dict(num_leaves=7, min_child_samples=20, learning_rate=0.02, n_estimators=150, colsample_bytree=0.8,
                 reg_lambda=30, subsample=0.8, min_split_gain=0.01)
TUNED_GBM2 = dict(num_leaves=7, min_child_samples=80, learning_rate=0.01, n_estimators=600, colsample_bytree=0.4,
                  reg_lambda=30, subsample=0.6)
GBM2_TASK = {"coup_5y": dict(num_leaves=4, min_child_samples=80, learning_rate=0.01, n_estimators=300,
                             colsample_bytree=0.2, reg_lambda=10, subsample=0.6)}
TUNED_NET = dict(hidden=384, dropout=0.25, epochs=20, lr=0.01, wd=0.001)
FACT = {"ensemble": lambda: Wrap(M.Ensemble(members=8), True),
        "ens_tuned": lambda: Wrap(M.Ensemble(members=8, gbm_params=TUNED_GBM, net_params=TUNED_NET), True),
        "ens_tuned_nologit": lambda: Wrap(M.Ensemble(members=8, gbm_params=TUNED_GBM, net_params=TUNED_NET, use_logit=False), True),
        "gbm_tuned": lambda: Wrap(M.GBMModel(TUNED_GBM)),
        "gbm2": lambda: Wrap(M.GBMModel(TUNED_GBM2, GBM2_TASK)),
        "net_tuned": lambda: Wrap(M.PsychoNet(members=8, **TUNED_NET), True), "gbm": lambda: Wrap(M.GBMModel()),
        "logit": lambda: Wrap(M.LogitModel()), "psychonet": lambda: Wrap(M.PsychoNet(members=8), True)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--drop", default=""); ap.add_argument("--model", default="ensemble")
    ap.add_argument("--lock", action="store_true"); ap.add_argument("--notes", default="")
    ap.add_argument("--first-year", type=int, default=None)
    ap.add_argument("--drop-feat", default="", help="comma-separated individual features to blank")
    ap.add_argument("--calib", type=int, default=0, help="recent-window recalibration (years), 0 = off")
    a = ap.parse_args()
    p, base, X, Yl = prepare()
    drop = [f for g in a.drop.split(",") if g for f in GROUPS.get(g, [])] + [f for f in a.drop_feat.split(",") if f]
    if drop:
        X = X.copy(); X[..., [FEATURES.index(f) for f in drop]] = np.nan
    fac = FACT[a.model]
    if a.calib:
        base_f = fac
        fac = lambda: RecentCalibrated(base_f, window=a.calib)
    s, out, _ = evaluate(p, X, Yl, fac, LOCK_ORIGINS if a.lock else DEV_ORIGINS, first_year=a.first_year)
    log(a.name, s, out, a.notes or f"drop={a.drop} model={a.model} first_year={a.first_year} calib={a.calib}", "lock" if a.lock else "dev")

if __name__ == "__main__":
    main()
