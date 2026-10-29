"""Should the interstate continuation table (P(an active conflict is still active next year) by run length 1, 2, 3+)
weight recent years more? At each origin O the table is estimated from dyad-years before O with weights 0.5^((O-y)/H)
(H = None: equal weights, production), each cell shrunk toward the equal-weight cell with K pseudo-observations, and
scored by the log-loss and Brier of every continuation outcome in O..O+9 (run length as observed).
Dev origins 1970-1998, test origins 2000-2014 (step 2; their 10-year windows overlap, so pool events, don't count origins).
python -P scripts/interstate_cont_test.py -> outputs/interstate_cont_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import interstate as I

ROOT = os.path.join(os.path.dirname(__file__), "..")
HS = (None, 40, 25, 15, 10)
KS = (0, 5, 20)


def cells(d):
    return {r_: (d.conflict_run_years >= 3) if r_ == 3 else (d.conflict_run_years == r_) for r_ in (1, 2, 3)}


def table(d, O, H, K):
    w = np.ones(len(d)) if H is None else 0.5 ** ((O - d.year.to_numpy()) / H)
    y = d["next"].astype(float).to_numpy()
    out = {}
    for r_, m in cells(d).items():
        m = m.to_numpy()
        p0 = y[m].mean() if m.sum() >= 5 else 0.5               # equal-weight cell (production)
        out[r_] = float((np.sum(w[m] * y[m]) + K * p0) / (np.sum(w[m]) + K)) if m.sum() >= 5 else 0.5
    return out


def main():
    df = I.build(False)
    d = df[df.ongoing == 1].copy()
    nxt = df.set_index(["a", "b", "year"]).ev_ucdp
    d["next"] = [nxt.get((a, b, y + 1), np.nan) for a, b, y in zip(d.a, d.b, d.year)]
    d = d[d["next"].notna()]
    res = {}
    for split, origins in (("dev", range(1970, 1999, 2)), ("test", range(2000, 2015, 2))):
        for H in HS:
            for K in KS:
                if H is None and K:
                    continue
                ll, br, n = 0.0, 0.0, 0
                for O in origins:
                    tab = table(d[d.year < O], O, H, K)
                    ev = d[(d.year >= O) & (d.year < O + 10) & (d.year + 1 <= 2024)]
                    for r_, m in cells(ev).items():
                        y = ev.loc[m, "next"].astype(float).to_numpy()
                        p = np.clip(tab[r_], 1e-3, 1 - 1e-3)
                        ll -= np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
                        br += np.sum((y - p) ** 2)
                        n += len(y)
                res.setdefault(split, {})[f"H={H},K={K}"] = {"logloss": round(ll / n, 4), "brier": round(br / n, 4), "n": n}
        print(split, json.dumps(res[split], indent=0), flush=True)
    res["tables_2025"] = {f"H={H},K={K}": table(d, 2025, H, K) for H in HS for K in KS if not (H is None and K)}
    json.dump(res, open(os.path.join(ROOT, "outputs", "interstate_cont_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
