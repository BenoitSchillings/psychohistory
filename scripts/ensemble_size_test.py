"""Does a larger neural ensemble help? Production Seldon ensemble (combination rule from ensemble_config.json) with
10 network members per seed (the forecast setting) vs 20 and 40, and with 2 vs 4 network seeds. Dev origins only;
paired country bootstrap of each score difference vs the 10-member production setting.
python -P scripts/ensemble_size_test.py -> outputs/ensemble_size_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import models as M
from psychohistory.models import SeldonEnsemble, load_combine, PsychoNet, QuantileNet, NET_TUNED
from psychohistory.pipeline import prepare
from psychohistory.evaluate import evaluate, metrics, HEADLINE, DEV_ORIGINS
from psychohistory.features import TASKS
from psychohistory.data import ROOT


def factory(members, net_seeds=2):
    def f():
        e = SeldonEnsemble(load_combine(), members=members)
        for fam, cls in (("N", lambda s: PsychoNet(members=members, seed=s * 5, **NET_TUNED)), ("Nq", lambda s: QuantileNet(members=members, seed=s * 5))):
            if fam in e.fam:
                e.fam[fam] = [cls(s) for s in range(net_seeds)]
        return e
    return f


p, base, X, Yl = prepare()
CONF = {"m10": factory(10), "m20": factory(20), "m40": factory(40), "m10_seeds4": factory(10, 4), "m20_seeds4": factory(20, 4)}
res, cats = {}, {}
for name, fac in CONF.items():
    s, out, preds = evaluate(p, X, Yl, fac, DEV_ORIGINS)
    cats[name] = [np.concatenate([a[i] for a in preds]) for i in (1, 2, 3, 4)]
    res[name] = {"score": round(s, 4)}
    print(name, s, flush=True)
y, _, b, g = cats["m10"]
ug = np.unique(g)
rows = {c: np.where(g == c)[0] for c in ug}
sc = lambda pr, idx: np.mean([sum(metrics(y[idx, k], pr[idx, k], b[idx, k]).get(m_, np.nan) for m_ in ("auc", "bss"))
                              for k in [TASKS.index(t) for t in HEADLINE]])
rng = np.random.default_rng(0)
boots = [np.concatenate([rows[c] for c in rng.choice(ug, len(ug))]) for _ in range(300)]
for name in CONF:
    if name == "m10":
        continue
    d = [sc(cats[name][1], i) - sc(cats["m10"][1], i) for i in boots]
    res[name]["diff_vs_m10"] = {"point": round(res[name]["score"] - res["m10"]["score"], 4), "ci90": np.quantile(d, [.05, .95]).round(4).tolist()}
    print(name, res[name]["diff_vs_m10"], flush=True)
json.dump(res, open(ROOT / "outputs" / "ensemble_size_test.json", "w"), indent=1)
