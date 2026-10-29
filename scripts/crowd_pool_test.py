"""Does the model add anything to the crowd on resolved questions? Fixed-weight pools of crowd and model
(logarithmic pool: logit p = (1 - w) logit crowd + w logit model; w = 0 is the crowd alone), on the matched questions of
outputs/crowd_benchmark.json, scored by Brier and log score, with a bootstrap over country pairs (questions on the same
pair are not independent). A weight is worth using only if its 90% interval of the Brier difference with the crowd is
below zero. python -P scripts/crowd_pool_test.py -> outputs/crowd_pool_test.json"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
WS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def main():
    B = json.load(open(os.path.join(ROOT, "outputs", "crowd_benchmark.json")))
    lg = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
    out = {}
    for snap in ("open",):          # (the per-question list keeps the one-week-after-opening snapshot)
        Q = [q for q in B["questions"] if q["snap"] == snap and q["crowd"] is not None and q["model"] is not None]
        y = np.array([q["y"] for q in Q], float)
        c, m = np.array([q["crowd"] for q in Q], float), np.array([q["model"] for q in Q], float)
        pairs = np.array([q["pair"] for q in Q])
        up = np.unique(pairs)
        rng = np.random.default_rng(0)
        boot = [np.concatenate([np.flatnonzero(pairs == p) for p in rng.choice(up, len(up))]) for _ in range(2000)]
        res = {}
        for w in WS:
            p = 1 / (1 + np.exp(-((1 - w) * lg(c) + w * lg(m))))
            br = (p - y) ** 2
            d = br - (c - y) ** 2
            ci = np.quantile([d[i].mean() for i in boot], [0.05, 0.95])
            res[str(w)] = {"brier": round(float(br.mean()), 4), "log": round(float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
                           "brier_minus_crowd_90ci": [round(float(x), 4) for x in ci]}
        out[snap] = {"n": len(Q), "events": int(y.sum()), "pairs": len(up), "pools": res}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "outputs", "crowd_pool_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
