"""Per-task ensemble weights over model families, with a nested leave-one-origin-out estimate
of how well the weighting procedure itself generalises.

python scripts/search_weights2.py TAG            -> outputs/ensemble_weights_TAG.json
python scripts/search_weights2.py TAG --apply lock   (score saved weights on the lockbox cache)
"""
import itertools, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from psychohistory.evaluate import HEADLINE, log, metrics
from psychohistory.features import TASKS

FAMILIES = {"G": ["G_a", "G_b", "G_c"], "G60": ["G60_a", "G60_b"], "N": ["N_a", "N_b"], "Nq": ["Nq_a", "Nq_b"], "F": ["F"], "L": ["L"]}
STEP = 0.1


def load(split, tag):
    c = pickle.load(open(f"outputs/pred_cache2_{split}_{tag}.pkl", "rb"))
    ref = c["G_a"]
    origins = np.concatenate([np.full(len(a[1]), a[0]) for a in ref])
    y = np.concatenate([a[1] for a in ref])
    base = np.concatenate([a[3] for a in ref])
    fam = {f: np.mean([np.concatenate([a[2] for a in c[m]]) for m in ms], 0) for f, ms in FAMILIES.items()}
    return origins, y, base, fam


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


GRID = [w for w in itertools.product(np.arange(0, 1 + 1e-9, STEP), repeat=len(FAMILIES)) if abs(sum(w) - 1) < 1e-6]


def combine(fam, k, w, mode):
    if mode == "prob":
        return sum(wi * fam[f][:, k] for wi, f in zip(w, FAMILIES) if wi)
    return 1 / (1 + np.exp(-sum(wi * lg(fam[f][:, k]) for wi, f in zip(w, FAMILIES) if wi)))


def best_weights(fam, y, base, k, rows):
    best = (-9, None, None)
    for mode in ("prob", "logit"):
        for w in GRID:
            m = metrics(y[rows, k], combine(fam, k, w, mode)[rows], base[rows, k])
            if m and m["auc"] + m["bss"] > best[0]:
                best = (m["auc"] + m["bss"], w, mode)
    return best


def main():
    tag = sys.argv[1]
    if "--apply" in sys.argv:
        split = sys.argv[sys.argv.index("--apply") + 1]
        W = json.load(open(f"outputs/ensemble_weights_{tag}.json"))
        origins, y, base, fam = load(split, tag)
        out = {}
        for k, t in enumerate(TASKS):
            w = [W[t]["w"][f] for f in FAMILIES]
            out[t] = metrics(y[:, k], combine(fam, k, w, W[t]["mode"]), base[:, k])
        s = float(np.mean([out[t]["auc"] + out[t]["bss"] for t in HEADLINE]))
        log(f"{split}:weighted[{tag}]", s, out, "weights chosen on dev", split)
        return
    origins, y, base, fam = load("dev", tag)
    nested, final = {}, {}
    for k, t in enumerate(TASKS):
        # nested: choose on 6 origins, score on the 7th
        pred = np.full(len(y), np.nan)
        for o in np.unique(origins):
            _, w, mode = best_weights(fam, y, base, k, origins != o)
            pred[origins == o] = combine(fam, k, w, mode)[origins == o]
        nested[t] = metrics(y[:, k], pred, base[:, k])
        sc, w, mode = best_weights(fam, y, base, k, np.ones(len(y), bool))
        final[t] = {"w": dict(zip(FAMILIES, map(float, w))), "mode": mode, "dev_insample": sc}
        print(f"{t:20s} nested {nested[t]['auc'] + nested[t]['bss']:.4f}  in-sample {sc:.4f}  "
              f"w={ {f: round(v, 1) for f, v in final[t]['w'].items() if v} } {mode}", flush=True)
    s = float(np.mean([nested[t]["auc"] + nested[t]["bss"] for t in HEADLINE]))
    log(f"dev:nested-weighted[{tag}]", s, nested, "weights chosen leave-one-origin-out", "dev")
    json.dump(final, open(f"outputs/ensemble_weights_{tag}.json", "w"), indent=1)
    # reference: equal average of G and N families
    eq = {}
    for k, t in enumerate(TASKS):
        eq[t] = metrics(y[:, k], (fam["G"][:, k] + fam["N"][:, k]) / 2, base[:, k])
    log(f"dev:equal-G+N[{tag}]", float(np.mean([eq[t]["auc"] + eq[t]["bss"] for t in HEADLINE])), eq, "", "dev")
    for f in FAMILIES:
        o = {t: metrics(y[:, k], fam[f][:, k], base[:, k]) for k, t in enumerate(TASKS)}
        log(f"dev:family-{f}[{tag}]", float(np.mean([o[t]["auc"] + o[t]["bss"] for t in HEADLINE])), o, "", "dev")


if __name__ == "__main__":
    main()
