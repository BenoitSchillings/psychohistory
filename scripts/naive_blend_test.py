"""Would blending the naive (Fearon-Laitin) model into the production ensemble help? On the civil-war tasks it beat
the ensemble at the 2000 and 2015 origins. Fresh dev-origin predictions of both (same rows); per task, the blend weight
w in {0, 0.1, ..., 0.5} (probability average) is chosen by nested leave-one-origin-out: for each held-out origin, w is
picked on the other origins and scored on it. Only tasks whose nested score improves are candidates.
python -P scripts/naive_blend_test.py -> outputs/naive_blend_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory.models import SeldonEnsemble, load_combine, NaiveModel
from psychohistory.pipeline import prepare
from psychohistory.evaluate import evaluate, metrics, DEV_ORIGINS
from psychohistory.features import TASKS
from psychohistory.data import ROOT


class W:
    def __init__(self, m, years=False): self.m, self.uses_years = m, years
    def fit(self, X, Y, g, years=None):
        (self.m.fit(X, Y, g, years) if self.uses_years else self.m.fit(X, Y)); return self
    def predict(self, X): return self.m.predict(X)


p, base, X, Yl = prepare()
_, _, pe = evaluate(p, X, Yl, lambda: W(SeldonEnsemble(load_combine(), members=10), True), DEV_ORIGINS)
_, _, pn = evaluate(p, X, Yl, lambda: W(NaiveModel()), DEV_ORIGINS)
WS = np.round(np.arange(0, 0.51, 0.1), 2)
sc = lambda y, pr, b: (lambda m: m["auc"] + m["bss"])(metrics(y, pr, b))
out = {}
for k, t in enumerate(TASKS):
    per = {}
    for (O, y, pr_e, b, _), (_, _, pr_n, _, _) in zip(pe, pn):
        per[O] = {w: sc(y[:, k], (1 - w) * pr_e[:, k] + w * pr_n[:, k], b[:, k]) for w in WS}
    nested = []
    for O in per:
        others = [o for o in per if o != O]
        w_best = max(WS, key=lambda w: np.nanmean([per[o][w] for o in others]))
        nested.append((O, float(w_best), per[O][w_best] - per[O][0.0]))
    out[t] = {"nested_gain_mean": round(float(np.nanmean([g for _, _, g in nested])), 4),
              "origins_better": int(sum(g > 0 for _, _, g in nested)), "chosen_w_per_origin": [w for _, w, _ in nested],
              "w_all_dev": float(max(WS, key=lambda w: np.nanmean([per[o][w] for o in per])))}
    print(t, out[t], flush=True)
json.dump(out, open(ROOT / "outputs" / "naive_blend_test.json", "w"), indent=1)
