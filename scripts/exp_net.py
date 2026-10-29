import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from psychohistory.pipeline import prepare
from psychohistory.features import FEATURES, GROUPS, TASKS
from psychohistory import models as M
from psychohistory.evaluate import evaluate, log, DEV_ORIGINS
from exp import TUNED_GBM2, GBM2_TASK, TUNED_NET

class NetTopK:
    """GBM ranks features (gain, summed over tasks) on the training rows; the net sees the top K,
    optionally rank-gauss transformed."""
    def __init__(self, k, quantile=False, members=8):
        self.k, self.q, self.members = k, quantile, members
    def fit(self, X, Y, g):
        if self.k:
            gb = M.GBMModel({**TUNED_GBM2, "n_estimators": 300}).fit(X, Y)
            imp = np.zeros(X.shape[1])
            for m in gb.models.values():
                imp += m.booster_.feature_importance("gain")
            self.cols = np.argsort(-imp)[:self.k]
        else:
            self.cols = np.arange(X.shape[1])
        Xs = X[:, self.cols]
        if self.q:
            from sklearn.preprocessing import QuantileTransformer
            self.qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=100000).fit(Xs)
            Xs = self._q(Xs)
        self.m = M.PsychoNet(members=self.members, **TUNED_NET).fit(Xs, Y, g)
        return self
    def _q(self, Xs):
        Z = self.qt.transform(np.nan_to_num(Xs, nan=0.0))
        return np.where(np.isnan(Xs), np.nan, Z)
    def predict(self, X):
        Xs = X[:, self.cols]
        return self.m.predict(self._q(Xs) if self.q else Xs)

if __name__ == "__main__":
    p, base, X, Yl = prepare()
    X = X.copy(); X[..., [FEATURES.index(f) for f in GROUPS.get("Macro-finance", []) + GROUPS.get("Commodities", [])]] = np.nan
    for name, f in [("net:all", lambda: NetTopK(0)), ("net:top40", lambda: NetTopK(40)), ("net:top80", lambda: NetTopK(80)),
                    ("net:all-quantile", lambda: NetTopK(0, True)), ("net:top60-quantile", lambda: NetTopK(60, True))]:
        s, out, _ = evaluate(p, X, Yl, f, DEV_ORIGINS)
        log("e12:" + name, s, out)
