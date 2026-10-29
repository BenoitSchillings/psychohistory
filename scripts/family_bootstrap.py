"""Paired country-cluster bootstrap of the score gain from adding a data family of scripts/new_data_test.py (the
tuned GBM used there), as scripts/paired_bootstrap.py does for dropping a feature group.
python -P scripts/family_bootstrap.py NAME [--lock] [--n 500]"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from psychohistory import models as M
from psychohistory.evaluate import evaluate, DEV_ORIGINS, LOCK_ORIGINS
from psychohistory.pipeline import prepare
from exp import Wrap, TUNED_GBM
from new_data_test import SOURCES
from paired_bootstrap import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--lock", action="store_true"); ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    p, base, X, Yl = prepare()
    arrs = SOURCES[a.name](p)
    Xa = np.concatenate([X, np.stack(list(arrs.values()), -1).astype(np.float32)], -1)
    fac = lambda: Wrap(M.GBMModel(TUNED_GBM))
    orig = LOCK_ORIGINS if a.lock else DEV_ORIGINS
    out = []
    for XX in (X, Xa):
        _, _, preds = evaluate(p, XX, Yl, fac, orig)
        out.append([np.concatenate([q[i] for q in preds]) for i in (1, 2, 3, 4)])
    (yA, pA, bA, cA), (yB, pB, bB, cB) = out
    full = np.arange(len(cA))
    d0 = score(yB, pB, bB, full) - score(yA, pA, bA, full)
    rng = np.random.default_rng(0)
    countries = np.unique(cA)
    rows = {c: np.where(cA == c)[0] for c in countries}
    ds = np.array([score(yB, pB, bB, idx) - score(yA, pA, bA, idx)
                   for idx in (np.concatenate([rows[c] for c in rng.choice(countries, len(countries))]) for _ in range(a.n))])
    print(f"{a.name}: gain {d0:+.4f}; 90% interval [{np.quantile(ds, .05):+.4f}, {np.quantile(ds, .95):+.4f}]; "
          f"P(gain > 0) = {(ds > 0).mean():.2f} ({'lock' if a.lock else 'dev'})")


if __name__ == "__main__":
    main()
