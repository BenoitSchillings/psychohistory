"""Search for missing causes in the risk models' errors: screen every V-Dem v15 index the model does not use for an
association with what the tuned GBM gets wrong out of sample.
For each task and candidate z: logit(y ~ logit(p) + z) on the out-of-sample predictions p (standardised z, errors
clustered by country). Development: origins 1975-2005; confirmation: the 2010/2015 lockbox. A candidate passes if
|t| > 3.3 (p < 0.001) on development and the lockbox coefficient has the same sign with p < 0.05.
python -P scripts/residual_screen.py [indicators] -> outputs/residual_screen[_indicators].json
(indicators: the individual V-Dem indicators (v2..., model estimates, not the v2x indices) instead of the indices)"""
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import pyreadr
import statsmodels.api as sm

from psychohistory.data import RAW, ROOT
from psychohistory.evaluate import evaluate, DEV_ORIGINS, LOCK_ORIGINS
from psychohistory.features import TASKS
from psychohistory.pipeline import prepare
from psychohistory import models as M
from exp import Wrap, TUNED_GBM

USED = set(re.findall(r"v2[a-z_0-9]+", open(os.path.join(ROOT, "psychohistory", "data.py")).read()))


def candidates(p):
    v = list(pyreadr.read_r(RAW / "vdem_v15.RData").values())[0]
    used = USED | set(re.findall(r"v2[a-z_0-9]+", open(os.path.join(ROOT, "psychohistory", "data.py")).read()))
    if "indicators" in sys.argv:
        cols = [c for c in v.columns if re.match(r"^v2[a-wyz]", c) and not c.startswith("v2x")
                and not re.search(r"_(codelow|codehigh|sd|osp|ord|mean|nr|\d+)$", c) and c not in used]
    else:
        cols = [c for c in v.columns if re.match(r"^(v2x|v2xnp|v2xcl|v2xel|v2xeg|v2xdd|v2xcs|v2xme|v2xpe|v2xps|v2xlg|v2xex|v2xca|v2xdl)", c)
                and not re.search(r"_(codelow|codehigh|sd|osp|ord|mean|nr)$", c) and c not in used]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(v[c]) and v[c].notna().mean() > 0.3 and v[c].nunique() > 5]
    v = v[["country_text_id", "year"] + cols]
    idx = pd.MultiIndex.from_product([list(p.codes), list(p.years)])
    v = v.set_index(["country_text_id", "year"]).reindex(idx)
    return {c: v[c].to_numpy(float).reshape(len(p.codes), len(p.years)) for c in cols}


def main():
    p, base, X, Yl = prepare()
    fac = lambda: Wrap(M.GBMModel(TUNED_GBM))
    cand = candidates(p)
    print(len(cand), "candidates", flush=True)
    yi = {int(y): i for i, y in enumerate(p.years)}
    ci_of = {c: i for i, c in enumerate(p.codes)}
    res = {}
    for split, origins in (("dev", DEV_ORIGINS), ("lock", LOCK_ORIGINS)):
        _, _, preds = evaluate(p, X, Yl, fac, origins)
        # (each prediction row's country and year: the test window of origin T is years T..T+4, rows in flatten order)
        from psychohistory.backtest import flatten
        Xf, Yf, ci, year = flatten(p, X, Yl)
        Y, P, C, Yr = [], [], [], []
        for T, y_, pr, _, c_ in preds:
            te = (year >= T) & (year < T + 5)
            Y.append(y_), P.append(pr), C.append(c_), Yr.append(year[te])
        Y, P, C, Yr = map(np.concatenate, (Y, P, C, Yr))
        ti = np.array([yi[int(y)] for y in Yr])
        for k, task in enumerate(TASKS):
            for name, arr in cand.items():
                z = arr[C, ti]
                y, pr = Y[:, k], np.clip(P[:, k], 1e-4, 1 - 1e-4)
                ok = np.isfinite(y) & np.isfinite(z)
                if ok.sum() < 500 or y[ok].sum() < 20 or np.std(z[ok]) == 0:
                    continue
                Z = np.column_stack([np.log(pr[ok] / (1 - pr[ok])), (z[ok] - z[ok].mean()) / z[ok].std()])
                try:
                    m = sm.Logit(y[ok], sm.add_constant(Z)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": C[ok]})
                except Exception:
                    continue
                res.setdefault(f"{task}:{name}", {})[split] = {"coef": round(float(m.params[2]), 3), "t": round(float(m.tvalues[2]), 2),
                                                              "p": float(m.pvalues[2]), "n": int(ok.sum())}
        print(split, "done", flush=True)
    passed = {k: v for k, v in res.items() if "lock" in v and abs(v["dev"]["t"]) > 3.3
              and np.sign(v["dev"]["coef"]) == np.sign(v["lock"]["coef"]) and v["lock"]["p"] < 0.05}
    dev_hits = sum(abs(v["dev"]["t"]) > 3.3 for v in res.values() if "dev" in v)
    print(f"{len(res)} tests, {dev_hits} pass development, {len(passed)} confirmed on the lockbox")
    for k, v in sorted(passed.items(), key=lambda kv: -abs(kv[1]["dev"]["t"])):
        print(f"  {k:55s} dev {v['dev']['coef']:+.3f} (t {v['dev']['t']:+.1f})  lock {v['lock']['coef']:+.3f} (p {v['lock']['p']:.3f})")
    json.dump({"tests": res, "passed": sorted(passed), "dev_hits": dev_hits}, open(os.path.join(ROOT, "outputs", "residual_screen_indicators.json" if "indicators" in sys.argv else "residual_screen.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
