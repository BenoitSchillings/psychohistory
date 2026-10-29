"""Paired cluster bootstrap of the dev score difference between two feature configurations.

python scripts/paired_bootstrap.py --a-drop "Parties,Protest" --b-drop "" [--model gbm2] [--n 1000]
Resamples countries (all their test rows together) and reports the score difference B - A with a 90% interval.
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from psychohistory.pipeline import prepare
from psychohistory.features import FEATURES, GROUPS
from psychohistory.evaluate import evaluate, metrics, HEADLINE, DEV_ORIGINS, LOCK_ORIGINS
from psychohistory.features import TASKS
from exp import FACT


def run(p, base, X, Yl, drop, model, origins):
    cols = [FEATURES.index(f) for g in drop.split(",") if g for f in GROUPS[g]]
    Xa = X.copy()
    if cols:
        Xa[..., cols] = np.nan
    _, _, preds = evaluate(p, Xa, Yl, FACT[model], origins)
    cat = lambda i: np.concatenate([a[i] for a in preds])
    return cat(1), cat(2), cat(3), cat(4)


def score(y, pr, b, idx):
    return np.mean([sum(metrics(y[idx, k], pr[idx, k], b[idx, k]).get(m, np.nan) for m in ("auc", "bss"))
                    for k in [TASKS.index(t) for t in HEADLINE]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-drop", default=""); ap.add_argument("--b-drop", default="")
    ap.add_argument("--model", default="gbm2"); ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--origins", default="", help="comma-separated origin years (overrides dev/lock)")
    a = ap.parse_args()
    p, base, X, Yl = prepare()
    orig = tuple(int(x) for x in a.origins.split(",")) if a.origins else (LOCK_ORIGINS if a.lock else DEV_ORIGINS)
    yA, pA, bA, cA = run(p, base, X, Yl, a.a_drop, a.model, orig)
    yB, pB, bB, cB = run(p, base, X, Yl, a.b_drop, a.model, orig)
    assert (cA == cB).all()
    full = np.arange(len(cA))
    d0 = score(yB, pB, bB, full) - score(yA, pA, bA, full)
    rng = np.random.default_rng(0)
    countries = np.unique(cA)
    rows = {c: np.where(cA == c)[0] for c in countries}
    ds = []
    for _ in range(a.n):
        pick = rng.choice(countries, len(countries))
        idx = np.concatenate([rows[c] for c in pick])
        ds.append(score(yB, pB, bB, idx) - score(yA, pA, bA, idx))
    ds = np.array(ds)
    for t in TASKS:
        k = TASKS.index(t)
        mA, mB = metrics(yA[:, k], pA[:, k], bA[:, k]), metrics(yB[:, k], pB[:, k], bB[:, k])
        if mA and mB:
            print(f"   {t:20s} AUC {mA['auc']:.3f} -> {mB['auc']:.3f}   BSS {mA['bss']:+.3f} -> {mB['bss']:+.3f}   events {mA['events']}")
    print(f"score(B) - score(A) = {d0:+.4f}; 90% interval [{np.quantile(ds, .05):+.4f}, {np.quantile(ds, .95):+.4f}]; "
          f"P(B better) = {(ds > 0).mean():.2f}  ({'lock' if a.lock else 'dev'} origins, model {a.model})")


if __name__ == "__main__":
    main()
