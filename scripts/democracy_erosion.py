"""Does a model trained only on democracies predict their erosion better than the general ensemble?

Target: autocratization_5y (electoral democracy falls >= 0.10 within five years) among country-years with
democracy >= 0.6 at the origin. Rolling origins 1990-2015 (dev + held-out), paired comparison on the same rows.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from psychohistory.pipeline import prepare
from psychohistory.backtest import flatten, train_mask
from psychohistory.features import FEATURES, GROUPS, TASKS
from exp import FACT

p, base, X, Yl = prepare()
Xf, Yf, ci, year = flatten(p, X, Yl)
k = TASKS.index("autocratization_5y")
dem = Xf[:, FEATURES.index("democracy")]
SPEC = [f for g in ("State & regime", "Institutions", "Parties", "Protest", "Ideology & religion", "Elites & inequality",
                    "Economy / immiseration") for f in GROUPS.get(g, [])]
cols = [FEATURES.index(f) for f in SPEC]
rows = []
for T in range(1990, 2021, 5):
    Ytr, tr = train_mask(Yf, year, T)
    te = (year >= T) & (year < T + 5) & (dem >= 0.6) & np.isfinite(Yf[:, k])
    if te.sum() == 0:
        continue
    gen = FACT["gbm2"]().fit(Xf[tr], Ytr[tr], ci[tr]).predict(Xf[te])[:, k]
    trd = tr & (dem >= 0.5) & np.isfinite(Ytr[:, k])
    spec = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=40, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.5, reg_lambda=20, verbose=-1)
    spec.fit(Xf[trd][:, cols], Ytr[trd, k])
    sp = spec.predict_proba(Xf[te][:, cols])[:, 1]
    rows.append((T, Yf[te, k], gen, sp, ci[te]))
y = np.concatenate([r[1] for r in rows]); g = np.concatenate([r[2] for r in rows]); s_ = np.concatenate([r[3] for r in rows])
cc = np.concatenate([r[4] for r in rows])
b = (g + s_) / 2
base_ = y.mean()
for n, pr in [("general", g), ("democracies-only", s_), ("average", b)]:
    print(f"{n:18s} AUC {roc_auc_score(y, pr):.3f}  Brier {np.mean((pr - y) ** 2):.4f}  (base {np.mean((base_ - y) ** 2):.4f})")
print("rows", len(y), "events", int(y.sum()))
rng = np.random.default_rng(0); us = np.unique(cc); d = []
for _ in range(500):
    pick = np.concatenate([np.where(cc == c)[0] for c in rng.choice(us, len(us))])
    if 0 < y[pick].sum() < len(pick):
        d.append(roc_auc_score(y[pick], b[pick]) - roc_auc_score(y[pick], g[pick]))
print("average minus general AUC: %.3f, 90%% [%.3f, %.3f]" % (np.mean(d), np.quantile(d, .05), np.quantile(d, .95)))
