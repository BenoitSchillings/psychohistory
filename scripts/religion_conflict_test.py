"""Should religious groups among the regime's backers (sup_religious, adopted for coups and autocratisation only via
features.TASK_ONLY) also feed the civil-conflict models? In the logit screen it went with civil conflict (+0.27 sd,
s.e. 0.07, development years). Tuned GBM rolling-origin backtest (dev 1975-2005, lockbox 2010/2015) with the current
TASK_ONLY against a version that also lets sup_religious reach civil_conflict_5y and onset_1y.
python -P scripts/religion_conflict_test.py -> outputs/new_data/religion_conflict.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from psychohistory import features as FT, models as M
from psychohistory.data import ROOT
from psychohistory.evaluate import evaluate, DEV_ORIGINS, LOCK_ORIGINS
from psychohistory.pipeline import prepare
from exp import Wrap, TUNED_GBM


def main():
    p, base, X, Yl = prepare()
    fac = lambda: Wrap(M.GBMModel(TUNED_GBM))
    orig = dict(FT.TASK_ONLY)
    res = {}
    try:
        for name, extra in (("current", set()), ("religion_to_conflict", {"civil_conflict_5y", "onset_1y"})):
            FT.TASK_ONLY.clear()
            FT.TASK_ONLY.update({f: (ts | extra if f == "sup_religious" else ts) for f, ts in orig.items()})
            for split, o in (("dev", DEV_ORIGINS), ("lock", LOCK_ORIGINS)):
                s, out, _ = evaluate(p, X, Yl, fac, o)
                res[f"{name}:{split}"] = {"score": round(s, 4), **{t: {m: round(v, 4) for m, v in out[t].items() if m in ("auc", "bss")}
                                                                    for t in ("civil_conflict_5y", "onset_1y")}}
                print(name, split, json.dumps(res[f"{name}:{split}"]), flush=True)
    finally:
        FT.TASK_ONLY.clear(); FT.TASK_ONLY.update(orig)
    json.dump(res, open(os.path.join(ROOT, "outputs", "new_data", "religion_conflict.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
