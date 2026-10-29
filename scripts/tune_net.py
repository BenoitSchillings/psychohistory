import os, sys, warnings, random
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from psychohistory.pipeline import prepare
from psychohistory.features import FEATURES, GROUPS, TASKS
from psychohistory.models import PsychoNet
from psychohistory.evaluate import evaluate, DEV_ORIGINS
class W:
    def __init__(self, prm): self.m = PsychoNet(**prm)
    def fit(self, X, Y, g): self.m.fit(X, Y, g); return self
    def predict(self, X): return self.m.predict(X)
p, base, X, Yl = prepare()
X = X.copy(); X[..., [FEATURES.index(f) for f in GROUPS.get("Macro-finance", [])]] = np.nan
rng = random.Random(int(sys.argv[1]))
space = dict(hidden=[96, 192, 384, 768], dropout=[0.1, 0.25, 0.4, 0.5], epochs=[10, 20, 40], lr=[1e-3, 3e-3, 1e-2],
             wd=[1e-3, 1e-2, 5e-2], members=[8, 16])
out_path = "outputs/tune_net.csv"
for it in range(int(sys.argv[2])):
    prm = {k: rng.choice(v) for k, v in space.items()}
    for fy in (1955, 1860):
        s, out, _ = evaluate(p, X, Yl, lambda: W(prm), DEV_ORIGINS, first_year=fy)
        r = {"it": it, "first_year": fy, **prm, "score": s}
        for t in TASKS:
            if out.get(t): r[f"{t}_auc"], r[f"{t}_bss"] = out[t]["auc"], out[t]["bss"]
        pd.DataFrame([r]).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
        print(it, fy, round(s, 4), {t[:6]: (round(out[t]['auc'], 3), round(out[t]['bss'], 3)) for t in TASKS[:4]}, prm, flush=True)
