"""python scripts/cache_preds2.py SPLIT TAG [--leaders]: cache per-model predictions."""
import os, sys, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from psychohistory.pipeline import prepare
from psychohistory.features import FEATURES, GROUPS
from psychohistory import models as M
from psychohistory.evaluate import evaluate, DEV_ORIGINS, LOCK_ORIGINS
sys.path.insert(0, os.path.dirname(__file__))
from exp import Wrap, TUNED_GBM2, GBM2_TASK, TUNED_NET
from exp_net import NetTopK
split, tag = sys.argv[1], sys.argv[2]
origins = DEV_ORIGINS if split == "dev" else LOCK_ORIGINS
p, base, X, Yl = prepare()
drop = GROUPS.get("Macro-finance", []) + GROUPS.get("Commodities", []) + ([] if "--leaders" in sys.argv else GROUPS["Leaders"])
X = X.copy(); X[..., [FEATURES.index(f) for f in drop]] = np.nan
G = lambda seed: (lambda: Wrap(M.GBMModel({**TUNED_GBM2, "random_state": seed}, GBM2_TASK)))
specs = {"G_a": (G(0), None), "G_b": (G(1), None), "G_c": (G(2), None),
         "G60_a": (G(0), 1860), "G60_b": (G(1), 1860),
         "N_a": (lambda: Wrap(M.PsychoNet(members=8, seed=0, **TUNED_NET), True), None),
         "N_b": (lambda: Wrap(M.PsychoNet(members=8, seed=5, **TUNED_NET), True), None),
         "Nq_a": (lambda: Wrap(M.QuantileNet(members=8, seed=0), True), None),
         "Nq_b": (lambda: Wrap(M.QuantileNet(members=8, seed=1), True), None),
         "F": (lambda: Wrap(M.ForestModel()), None), "F60": (lambda: Wrap(M.ForestModel()), 1860),
         "L": (lambda: Wrap(M.LogitModel()), None)}
cache = {}
for name, (f, fy) in specs.items():
    s, out, preds = evaluate(p, X, Yl, f, origins, first_year=fy)
    cache[name] = preds
    print(split, tag, name, round(s, 4), flush=True)
pickle.dump(cache, open(f"outputs/pred_cache2_{split}_{tag}.pkl", "wb"))
