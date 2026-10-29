"""Does the country-year network gain from explicit history? Each feature's value 1 and 3 years earlier is appended
(X -> [X, X(t-1), X(t-3)]); PsychoNet (production settings, 8 members) on dev origins, paired country bootstrap of the
score difference against the same network without lags. python -P scripts/net_lags_test.py -> outputs/net_lags_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory.models import PsychoNet, NET_TUNED
from psychohistory.pipeline import prepare
from psychohistory.evaluate import evaluate, metrics, HEADLINE, DEV_ORIGINS
from psychohistory.features import TASKS
from psychohistory.data import ROOT


class Wrap:
    def __init__(self, m): self.m = m
    def fit(self, X, Y, g): self.m.fit(X, Y, g); return self
    def predict(self, X): return self.m.predict(X)


p, base, X, Yl = prepare()
lag = lambda a, k: np.concatenate([np.full(a[:, :k].shape, np.nan), a[:, :-k]], 1)
XL = np.concatenate([X, lag(X, 1), lag(X, 3)], -1).astype(X.dtype)
print("X", X.shape, "XL", XL.shape, flush=True)
res, cats = {}, {}
for name, Xa in (("base", X), ("lags", XL)):
    s, out, preds = evaluate(p, Xa, Yl, lambda: Wrap(PsychoNet(members=8, **NET_TUNED)), DEV_ORIGINS)
    res[name] = round(s, 4)
    cats[name] = [np.concatenate([a[i] for a in preds]) for i in (1, 2, 3, 4)]
    print(name, s, flush=True)
y, _, b, g = cats["base"]
ug = np.unique(g)
rows = {c: np.where(g == c)[0] for c in ug}
sc = lambda pr, idx: np.mean([sum(metrics(y[idx, k], pr[idx, k], b[idx, k]).get(m_, np.nan) for m_ in ("auc", "bss"))
                              for k in [TASKS.index(t) for t in HEADLINE]])
rng = np.random.default_rng(0)
d = []
for _ in range(300):
    idx = np.concatenate([rows[c] for c in rng.choice(ug, len(ug))])
    d.append(sc(cats["lags"][1], idx) - sc(cats["base"][1], idx))
res["diff"] = {"point": round(res["lags"] - res["base"], 4), "ci90": np.quantile(d, [.05, .95]).round(4).tolist()}
print(res)
json.dump(res, open(ROOT / "outputs" / "net_lags_test.json", "w"), indent=1)
