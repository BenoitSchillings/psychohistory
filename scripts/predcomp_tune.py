"""Tune the country-month fatality model on development windows that the challenge does not score (calendar years
2010-2017, each forecast from data to October of the year before, final GED), so the 2018-2023 and live results stay
out of sample. Mean CRPS over the dev windows (the challenge's scorer).
python scripts/predcomp_tune.py  -> outputs/predcomp_tune.csv (appends)
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

from psychohistory import cm_fatal as CF
sys.path.insert(0, os.path.dirname(__file__))
from predcomp_eval import crps_ensemble, ignorance, interval

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEV = list(range(2010, 2018))
TRUTH = CF.deaths(CF.mid(2023, 12))


def conflictology(D, origin, n=1000):
    """the country's last 12 observed months as an empirical distribution"""
    last = D[:, origin - 11:origin + 1]
    idx = (np.arange(n) % 12)
    return last[:, idx]


CACHE = os.path.join(ROOT, "outputs", "predcomp_tune_cache")


def draws_for(train_key, cfg, years):
    """trained draws per dev year, cached by training configuration"""
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for y in years:
        f = os.path.join(CACHE, f"{train_key}_{y}.npz")
        if os.path.exists(f):
            z = np.load(f)
            out[y] = {int(k): z[k] for k in z.files}
            continue
        CF.USE_EVENTS = bool(cfg.get("events"))
        CF.USE_SUSTAIN = bool(cfg.get("sustain"))
        CF.RESID = cfg.get("resid")
        origin = CF.mid(y - 1, 10)
        months = [CF.mid(y, m) for m in range(1, 13)]
        dr, _ = CF.train_and_forecast(origin, months, quants=np.array(cfg.get("quants", CF.QUANTS)),
                                      params=cfg.get("params"), start=cfg.get("start", CF.mid(1995, 1)))
        np.savez_compressed(f, **{str(k): v.astype(np.float32) for k, v in dr.items()})
        out[y] = dr
    CF.USE_EVENTS = False
    CF.USE_SUSTAIN = False
    CF.RESID = None
    return out


def post(X, cfg, cz):
    if cfg.get("spread", 1.0) != 1.0:
        lx = np.log1p(X)
        med = np.median(lx, 1, keepdims=True)
        X = np.expm1(np.clip(med + cfg["spread"] * (lx - med), 0, None))
    w = cfg.get("mix_conflictology", 0.0)
    if w > 0:
        k = int(round(w * X.shape[1]))
        X = np.concatenate([X[:, :X.shape[1] - k], cz[:, :k]], 1)
    return X


def run(cfg, years=DEV):
    train_key = cfg.get("train", "v1")
    dr = draws_for(train_key, TRAIN[train_key], years)
    res = []
    for y in years:
        origin = CF.mid(y - 1, 10)
        cz = conflictology(CF.deaths(origin), origin)
        for m, X in dr[y].items():
            X = post(np.asarray(X, float), cfg, cz)
            yv = TRUTH[:, m]
            res.append({"year": y, "crps": crps_ensemble(yv, X).mean(), "ign": ignorance(yv, X).mean(), "mis": interval(yv, X).mean()})
    r = pd.DataFrame(res).groupby("year").mean()
    return float(r.crps.mean()), float(r.ign.mean()), float(r.mis.mean()), r.crps.round(2).to_dict()


TRAIN = {
    "v1": {},
    "tail": {"quants": [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999]},
    "events": {"events": True},
    "leaves63": {"params": {"num_leaves": 63, "min_child_samples": 50}},
    "resid3": {"resid": "ld_m3"}, "resid12": {"resid": "ld_m12"}, "resid1": {"resid": "ld"},
    "resid3_sus": {"resid": "ld_m3", "sustain": True},
}


CONFIGS = {
    "v1": {}, "mix15": {"mix_conflictology": 0.15}, "mix30": {"mix_conflictology": 0.3}, "mix50": {"mix_conflictology": 0.5},
    "spread0.9": {"spread": 0.9}, "spread1.15": {"spread": 1.15},
    "tail": {"train": "tail"}, "spread1.3": {"spread": 1.3}, "spread1.5": {"spread": 1.5},
    "spread1.3_mix30": {"spread": 1.3, "mix_conflictology": 0.3}, "spread1.15_mix30": {"spread": 1.15, "mix_conflictology": 0.3},
    "tail_spread1.3": {"train": "tail", "spread": 1.3}, "events": {"train": "events"}, "leaves63": {"train": "leaves63"},
    "resid3": {"train": "resid3"}, "resid12": {"train": "resid12"}, "resid1": {"train": "resid1"},
    "resid3_sus": {"train": "resid3_sus"},
    "resid3_s1.15": {"train": "resid3", "spread": 1.15}, "resid12_s1.15": {"train": "resid12", "spread": 1.15},
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(CONFIGS)
    out = os.path.join(ROOT, "outputs", "predcomp_tune.csv")
    for nm in names:
        t0 = time.time()
        c, i, m, by = run(CONFIGS[nm])
        row = pd.DataFrame([{"config": nm, "crps": round(c, 3), "ign": round(i, 4), "mis": round(m, 1), "by_year": json.dumps(by),
                             "secs": round(time.time() - t0)}])
        row.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(row.to_string(index=False, header=False), flush=True)
