import pickle, json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from psychohistory.evaluate import metrics, HEADLINE, log
from psychohistory.features import TASKS
split = sys.argv[1]; wfile = sys.argv[2] if len(sys.argv) > 2 else "outputs/ensemble_weights.json"
c = pickle.load(open(f"outputs/pred_cache_{split}.pkl", "rb")); W = json.load(open(wfile))
st = lambda n, k: np.concatenate([a[2][:, k] for a in c[n]])
comps = {"G": lambda k: (st("gbm1955", k) + st("gbm1955_s2", k)) / 2, "G60": lambda k: st("gbm1860", k),
         "N": lambda k: (st("net1955", k) + st("net1955_s2", k)) / 2, "L": lambda k: st("logit1955", k)}
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
out = {}
for k, t in enumerate(TASKS):
    y = np.concatenate([a[1][:, k] for a in c["gbm1955"]]); b = np.concatenate([a[3][:, k] for a in c["gbm1955"]])
    w, mode = W[t]["w"], W[t]["mode"]
    P = {n: f(k) for n, f in comps.items()}
    pr = sum(w[n] * P[n] for n in P) if mode == "prob" else 1 / (1 + np.exp(-sum(w[n] * lg(P[n]) for n in P)))
    out[t] = metrics(y, pr, b)
s = float(np.mean([out[t]["auc"] + out[t]["bss"] for t in HEADLINE]))
log(f"{split}:weighted-ensemble({os.path.basename(wfile)})", s, out, "per-task weights from dev search", split)
