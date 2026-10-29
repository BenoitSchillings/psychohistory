import pickle, itertools, sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from psychohistory.evaluate import metrics, HEADLINE
from psychohistory.features import TASKS
c = pickle.load(open("outputs/pred_cache_dev.pkl", "rb"))
def stack(name, k):
    return np.concatenate([a[2][:, k] for a in c[name]])
y_all = {k: np.concatenate([a[1][:, k] for a in c["gbm1955"]]) for k in range(len(TASKS))}
b_all = {k: np.concatenate([a[3][:, k] for a in c["gbm1955"]]) for k in range(len(TASKS))}
comps = {"G": lambda k: (stack("gbm1955", k) + stack("gbm1955_s2", k)) / 2, "G60": lambda k: stack("gbm1860", k),
         "N": lambda k: (stack("net1955", k) + stack("net1955_s2", k)) / 2, "L": lambda k: stack("logit1955", k)}
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
grid = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=4) if abs(sum(w) - 1) < 1e-6]
best = {}
for t in TASKS:
    k = TASKS.index(t)
    P = {n: f(k) for n, f in comps.items()}
    res = []
    for w in grid:
        for mode in ("prob", "logit"):
            if mode == "prob":
                pr = sum(wi * P[n] for wi, n in zip(w, P))
            else:
                pr = 1 / (1 + np.exp(-sum(wi * lg(P[n]) for wi, n in zip(w, P))))
            m = metrics(y_all[k], pr, b_all[k])
            if m:
                res.append((m["auc"] + m["bss"], w, mode, m["auc"], m["bss"]))
    res.sort(key=lambda r: -r[0])
    eq = [r for r in res if r[2] == "prob" and np.allclose(r[1], (0.5, 0, 0.5, 0))][0]
    best[t] = res[0]
    print(f"{t:20s} best {res[0][0]:.4f} w={dict(zip(P, np.round(res[0][1], 1)))} {res[0][2]} (auc {res[0][3]:.3f} bss {res[0][4]:+.3f}) | G+N equal {eq[0]:.4f}")
print("score best", np.mean([best[t][0] for t in HEADLINE]))
json.dump({t: {"w": dict(zip(comps, map(float, best[t][1]))), "mode": best[t][2]} for t in TASKS},
          open("outputs/ensemble_weights.json", "w"), indent=1)
