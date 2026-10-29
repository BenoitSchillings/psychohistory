"""Random search of LightGBM settings on the dev origins; per-task scores logged."""
import os, sys, warnings, json, random
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from psychohistory.pipeline import prepare
from psychohistory.features import FEATURES, GROUPS, TASKS
from psychohistory.models import GBMModel
from psychohistory.evaluate import evaluate, DEV_ORIGINS
class W:
    def __init__(self, prm): self.m = GBMModel(prm)
    def fit(self, X, Y, g): self.m.fit(X, Y); return self
    def predict(self, X): return self.m.predict(X)
p, base, X, Yl = prepare()
X = X.copy(); X[..., [FEATURES.index(f) for f in GROUPS.get("Macro-finance", [])]] = np.nan
rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
space = dict(num_leaves=[4, 7, 15, 31], min_child_samples=[20, 40, 80, 160], learning_rate=[0.01, 0.02, 0.05],
             n_estimators=[150, 300, 600, 1000], colsample_bytree=[0.2, 0.4, 0.6, 0.8], reg_lambda=[0, 2, 10, 30],
             subsample=[0.6, 0.8, 1.0], min_split_gain=[0, 0.01])
rows = []
out_path = "outputs/tune_gbm.csv"
for it in range(int(sys.argv[2]) if len(sys.argv) > 2 else 24):
    prm = {k: rng.choice(v) for k, v in space.items()}
    for fy in (1955, 1860):
        s, out, _ = evaluate(p, X, Yl, lambda: W(prm), DEV_ORIGINS, first_year=fy)
        r = {"it": it, "first_year": fy, **prm, "score": s}
        for t in TASKS:
            if out.get(t): r[f"{t}_auc"], r[f"{t}_bss"] = out[t]["auc"], out[t]["bss"]
        pd.DataFrame([r]).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
        print(it, fy, round(s, 4), {t[:6]: (round(out[t]['auc'], 3), round(out[t]['bss'], 3)) for t in TASKS[:4]}, prm, flush=True)
