import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from psychohistory.pipeline import prepare
from psychohistory.features import FEATURES, GROUPS, TASKS
from psychohistory import models as M
from psychohistory.evaluate import evaluate, log, DEV_ORIGINS

class SingleTask:
    """QuantileNet trained on a subset of tasks (others hidden)."""
    def __init__(self, keep, members=8, seed=0):
        self.keep = [TASKS.index(t) for t in keep]; self.m = M.QuantileNet(members=members, seed=seed)
    def fit(self, X, Y, g):
        Y2 = np.full_like(Y, np.nan); Y2[:, self.keep] = Y[:, self.keep]
        r = np.isfinite(Y2).any(1)
        self.m.fit(X[r], Y2[r], g[r]); return self
    def predict(self, X): return self.m.predict(X)

p, base, X, Yl = prepare()
X = X.copy(); X[..., [FEATURES.index(f) for f in GROUPS.get("Macro-finance", []) + GROUPS.get("Commodities", [])]] = np.nan
for name, f in [("Nq multi-task", lambda: SingleTask(TASKS)),
                ("Nq econ only", lambda: SingleTask(["econ_collapse_5y"])),
                ("Nq econ+5y tasks", lambda: SingleTask(["civil_conflict_5y", "econ_collapse_5y", "autocratization_5y", "coup_5y"]))]:
    s, out, _ = evaluate(p, X, Yl, f, DEV_ORIGINS)
    e = out["econ_collapse_5y"]
    print(f"[econ] {name:20s} econ auc {e['auc']:.3f} bss {e['bss']:+.3f}", flush=True)
