"""Cache dev-origin predictions of individual models for offline combination search."""
import os, sys, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from psychohistory.pipeline import prepare
from psychohistory.features import FEATURES, GROUPS
from psychohistory import models as M
from psychohistory.evaluate import evaluate, DEV_ORIGINS, LOCK_ORIGINS
sys.path.insert(0, os.path.dirname(__file__))
from exp import Wrap, TUNED_GBM, TUNED_NET
split = sys.argv[1] if len(sys.argv) > 1 else "dev"
origins = DEV_ORIGINS if split == "dev" else LOCK_ORIGINS
p, base, X, Yl = prepare()
X = X.copy(); X[..., [FEATURES.index(f) for f in GROUPS.get("Macro-finance", []) + GROUPS.get("Commodities", [])]] = np.nan
cache = {}
specs = {
    "gbm1955": (lambda: Wrap(M.GBMModel(TUNED_GBM)), None),
    "gbm1860": (lambda: Wrap(M.GBMModel(TUNED_GBM)), 1860),
    "gbm1955_s2": (lambda: Wrap(M.GBMModel({**TUNED_GBM, "random_state": 7, "colsample_bytree": 0.6})), None),
    "net1955": (lambda: Wrap(M.PsychoNet(members=8, **TUNED_NET), True), None),
    "net1955_s2": (lambda: Wrap(M.PsychoNet(members=8, seed=3, **TUNED_NET), True), None),
    "logit1955": (lambda: Wrap(M.LogitModel()), None),
}
for name, (f, fy) in specs.items():
    s, out, preds = evaluate(p, X, Yl, f, origins, first_year=fy)
    cache[name] = preds
    print(name, round(s, 4), flush=True)
pickle.dump(cache, open(f"outputs/pred_cache_{split}.pkl", "wb"))
