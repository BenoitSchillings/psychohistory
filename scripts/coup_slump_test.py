"""Does the coup model overreact to economic slumps?

At each origin the production risk network is trained on data up to the origin (as in the hindcasts) and predicts the
next-year coup attempt hazard for the following years of real history. Calibration is then checked by bins of last
year's growth and of the GDP drawdown, and a recalibration logit is fitted on the held-out rows:
    coup ~ offset(logit p_model) + a + b_g * growth_1y + b_d * gdp_drawdown
b_g > 0 (b_d > 0) means real coups rose less after slumps than the model says: overreaction.

python -P scripts/coup_slump_test.py [H] -> outputs/coup_slump_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

ORIGINS = {"dev": [1975, 1980, 1985, 1990, 1995], "test": [2000, 2005, 2009]}


def logit_offset(y, off, Z, iters=50, groups=None):
    """logistic regression with an offset; returns coefficients and standard errors (clustered by `groups`)"""
    Z = np.column_stack([np.ones(len(y)), Z])
    b = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(off + Z @ b)))
        W = p * (1 - p)
        H = Z.T @ (Z * W[:, None]) + 1e-6 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, Z.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    Hi = np.linalg.inv(H)
    if groups is None:
        return b, np.sqrt(np.diag(Hi))
    sc = Z * (y - p)[:, None]
    G = np.zeros((Z.shape[1], Z.shape[1]))
    for gk in np.unique(groups):
        u = sc[groups == gk].sum(0)
        G += np.outer(u, u)
    return b, np.sqrt(np.diag(Hi @ G @ Hi))


def main():
    from psychohistory import pipeline as PL
    from psychohistory.backtest import flatten, train_mask
    from psychohistory.features import FEATURES
    from psychohistory.models import NET_TUNED, PsychoNet
    from psychohistory.simulate import COUP

    H = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    p, base, X, Yl = PL.prepare(False)
    Xf, Yf, ci, year = flatten(p, X, Yl)
    jg, jd = FEATURES.index("growth_1y"), FEATURES.index("gdp_drawdown")
    rows = {}
    for split, origins in ORIGINS.items():
        for O in origins:
            Ytr, tr = train_mask(Yf, year, O)
            net = PsychoNet(members=8, **NET_TUNED).fit(Xf[tr], Ytr[tr], ci[tr])
            # features in O .. end-1 predict coups in O+1 .. end; windows end at the next origin so no row counts
            # twice (the last origin of each split runs H years, the dev one stops before the test years)
            nxt = [o for o in origins if o > O]
            end = nxt[0] if nxt else (O + H if split == "test" else ORIGINS["test"][0])
            te = (year >= O) & (year < end) & np.isfinite(Yf[:, COUP]) & np.isfinite(Xf[:, jg]) & np.isfinite(Xf[:, jd])
            pr = net.predict(Xf[te])[:, COUP]
            rows.setdefault(split, []).append(np.column_stack([Yf[te, COUP], pr, Xf[te, jg], Xf[te, jd],
                                                               np.full(te.sum(), O), ci[te]]))
            print(split, O, int(te.sum()), "rows", round(float(Yf[te, COUP].sum()), 0), "coups",
                  round(float(pr.sum()), 1), "predicted", flush=True)
    np.savez(os.path.join(os.path.dirname(__file__), "..", "outputs", "coup_slump_rows.npz"),
             **{k: np.concatenate(v) for k, v in rows.items()})
    out = {}
    for split, r in rows.items():
        r = np.concatenate(r)
        y, pr, g, d = r[:, 0], np.clip(r[:, 1], 1e-5, 1 - 1e-5), r[:, 2], r[:, 3]
        res = {"n": int(len(y)), "coups": float(y.sum()), "predicted": float(pr.sum())}
        for name, v, edges in (("growth_1y", g, [-np.inf, -0.05, -0.02, 0.0, 0.02, 0.05, np.inf]),
                               ("gdp_drawdown", d, [-np.inf, -0.3, -0.15, -0.05, -1e-9, np.inf])):
            bins = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (v > lo) & (v <= hi)
                if m.sum():
                    bins.append({"range": [float(lo), float(hi)], "n": int(m.sum()), "coups": float(y[m].sum()),
                                 "predicted": round(float(pr[m].sum()), 1),
                                 "ratio": round(float(y[m].sum() / max(pr[m].sum(), 1e-9)), 3)})
            res[name] = bins
        off = np.log(pr / (1 - pr))
        b, se = logit_offset(y, off, np.column_stack([g, d]), groups=r[:, 5])
        res["recal"] = {"a": [float(b[0]), float(se[0])], "b_growth": [float(b[1]), float(se[1])],
                        "b_drawdown": [float(b[2]), float(se[2])]}
        # the model's own slope on the same rows (logit p on growth and drawdown), for scale
        A = np.column_stack([np.ones(len(y)), g, d])
        res["model_slope"] = dict(zip(["a", "growth", "drawdown"], np.linalg.lstsq(A, off, rcond=None)[0].round(3).tolist()))
        # the raw historical slope on the same rows (no model)
        b0, se0 = logit_offset(y, np.zeros(len(y)), np.column_stack([g, d]), groups=r[:, 5])
        res["raw_slope"] = {"growth": [float(b0[1]), float(se0[1])], "drawdown": [float(b0[2]), float(se0[2])]}
        out[split] = res
        print(split, json.dumps(res, indent=1))
    # out of sample: recalibration fitted on dev, scored on test (log loss), level only vs level + slump terms
    dv, ts = (np.concatenate(rows[k]) for k in ("dev", "test"))
    def ll(y, q):
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    lo = lambda r: np.log(np.clip(r[:, 1], 1e-5, 1 - 1e-5) / (1 - np.clip(r[:, 1], 1e-5, 1 - 1e-5)))
    a_only, _ = logit_offset(dv[:, 0], lo(dv), np.zeros((len(dv), 0)))
    full, _ = logit_offset(dv[:, 0], lo(dv), dv[:, 2:4])
    dd_, _ = logit_offset(dv[:, 0], lo(dv), dv[:, 3:4])
    oos = {"raw": ll(ts[:, 0], ts[:, 1]),
           "level": ll(ts[:, 0], 1 / (1 + np.exp(-(lo(ts) + a_only[0])))),
           "level_drawdown": ll(ts[:, 0], 1 / (1 + np.exp(-(lo(ts) + dd_[0] + ts[:, 3] * dd_[1])))),
           "level_slump": ll(ts[:, 0], 1 / (1 + np.exp(-(lo(ts) + full[0] + ts[:, 2:4] @ full[1:]))))}
    out["oos_test_logloss"] = oos
    print("out of sample", oos)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "coup_slump_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
