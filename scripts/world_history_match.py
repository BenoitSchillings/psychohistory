"""Test the world layer's assumption ranges against history (history matching with emulators).

Input: scripts/world_hindcast.py runs (each world's sampled assumptions theta and summary statistics of its path, next
to what happened). For each origin and statistic, a ridge emulator mu(theta) (linear and squared terms) and its residual
sd (the luck that no assumption explains). A draw theta is scored on the development origins by
    log L(theta) = sum over origins and statistics of -0.5 ((observed - mu(theta)) / (c sd))^2,
c = sqrt(2) allowing for model error as large as the luck. Weights exp(log L) on the test origins' own worlds then show
whether preferring history-consistent assumptions improves the forecasts there (weighted vs unweighted CRPS of each
statistic). Only if it does are the production ranges narrowed (to the weighted quantiles), which the report states.

python -P scripts/world_history_match.py TAG DEV_ORIGINS TEST_ORIGINS   (origins comma-separated)
-> outputs/world_history_match_<TAG>.json"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "outputs", "world_hindcast")
C_DISC = np.sqrt(2)


def wcrps(x, y, w=None):
    """CRPS of a weighted ensemble x [S] against y"""
    w = np.full(len(x), 1 / len(x)) if w is None else w / w.sum()
    o = np.argsort(x)
    x, w = x[o], w[o]
    F = np.cumsum(w)
    # integral of (F(z) - 1{y <= z})^2 over z, piecewise constant between sorted points
    grid = np.concatenate([[min(x[0], y) - 1e-9], x, [max(x[-1], y) + 1e-9]])
    Fz = np.concatenate([[0.0], F])
    tot = 0.0
    for i in range(len(grid) - 1):
        a, b = grid[i], grid[i + 1]
        if b <= a:
            continue
        f = Fz[i] if i < len(Fz) else 1.0
        if y <= a:
            tot += (f - 1) ** 2 * (b - a)
        elif y >= b:
            tot += f ** 2 * (b - a)
        else:
            tot += f ** 2 * (y - a) + (f - 1) ** 2 * (b - y)
    return tot


def design(th, mu, sd):
    z = (th - mu) / sd
    return np.column_stack([np.ones(len(z)), z, z ** 2 - 1])


def main():
    tag = sys.argv[1]
    dev = [int(o) for o in sys.argv[2].split(",")]
    test = [int(o) for o in sys.argv[3].split(",")]
    runs = {}
    for f in glob.glob(os.path.join(D, f"{tag}_*.npz")):
        z = np.load(f, allow_pickle=True)
        runs[int(z["origin"])] = {k: z[k] for k in z.files}
    pn = sorted(set.intersection(*[set(r["pnames"].tolist()) for r in runs.values()]))
    stats = list(runs[dev[0]]["names"])
    pidx = {o: [list(r["pnames"]).index(n) for n in pn] for o, r in runs.items()}
    th_all = np.concatenate([runs[o]["params"][:, pidx[o]] for o in runs])
    th_mu, th_sd = th_all.mean(0), th_all.std(0) + 1e-9
    keep = th_sd > 1e-6
    pn = [n for n, k in zip(pn, keep) if k]
    th_mu, th_sd = th_mu[keep], th_sd[keep]
    for o in runs:
        pidx[o] = [i for i, k in zip(pidx[o], keep) if k]
    # emulators on the development origins
    em = {}
    out = {"params": pn, "stats": stats, "emulators": {}}
    for o in dev:
        r = runs[o]
        X = design(r["params"][:, pidx[o]], th_mu, th_sd)
        for k, s in enumerate(stats):
            y = r["sim"][:, k]
            ok = np.isfinite(y)
            if ok.sum() < 100:
                continue
            A = X[ok]
            beta = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ y[ok])
            res = y[ok] - A @ beta
            sd = res.std()
            r2 = 1 - res.var() / y[ok].var() if y[ok].var() > 0 else 0.0
            em[(o, s)] = (beta, sd, float(r["obs"][k]))
            # the assumptions that move this statistic most (standardised linear coefficient / luck sd)
            lin = beta[1:1 + len(pn)] / max(sd, 1e-9)
            top = np.argsort(-np.abs(lin))[:4]
            out["emulators"][f"{o}:{s}"] = {"r2": round(float(r2), 3), "obs": round(float(r["obs"][k]), 4),
                                            "sim_median": round(float(np.median(y[ok])), 4),
                                            "obs_pctile": round(float((y[ok] < r["obs"][k]).mean()), 3),
                                            "top": {pn[i]: round(float(lin[i]), 3) for i in top}}

    def loglik(th):
        X = design(th, th_mu, th_sd)
        L = np.zeros(len(th))
        for (o, s), (beta, sd, y) in em.items():
            L += -0.5 * ((y - X @ beta) / (C_DISC * max(sd, 1e-9))) ** 2
        return L

    # held-out test: weights from the development origins applied to the test origins' worlds
    out["test"] = {}
    for o in test:
        if o not in runs:
            continue
        r = runs[o]
        L = loglik(r["params"][:, pidx[o]])
        w = np.exp(L - L.max())
        ess = w.sum() ** 2 / (w ** 2).sum()
        res = {"ess": round(float(ess), 1), "n": len(w)}
        for k, s in enumerate(stats):
            y = r["sim"][:, k]
            ok = np.isfinite(y)
            if ok.sum() < 100:
                continue
            c0, c1 = wcrps(y[ok], float(r["obs"][k])), wcrps(y[ok], float(r["obs"][k]), w[ok])
            res[s] = {"crps": round(c0, 5), "crps_weighted": round(c1, 5), "rel": round(c1 / c0 - 1, 4) if c0 > 0 else None,
                      "obs_pctile": round(float((y[ok] < r["obs"][k]).mean()), 3)}
        out["test"][o] = res
    # pooled verdict per statistic
    out["verdict"] = {}
    for s in stats:
        rels = [out["test"][o][s]["rel"] for o in out["test"] if s in out["test"][o] and out["test"][o][s]["rel"] is not None]
        if rels:
            out["verdict"][s] = {"mean_rel": round(float(np.mean(rels)), 4), "better_at": int(np.sum(np.array(rels) < 0)), "n": len(rels)}
    # posterior of each assumption on the pooled worlds (development weights)
    th = np.concatenate([runs[o]["params"][:, pidx[o]] for o in runs])
    L = loglik(th)
    w = np.exp(L - L.max())
    w /= w.sum()
    out["ess_pooled"] = round(float(1 / (w ** 2).sum()), 1)

    def wq(x, q):
        o = np.argsort(x)
        cw = np.cumsum(w[o])
        return float(np.interp(q, cw, x[o]))

    out["posterior"] = {}
    for j, n in enumerate(pn):
        x = th[:, j]
        pr = np.quantile(x, [0.05, 0.5, 0.95])
        po = [wq(x, q) for q in (0.05, 0.5, 0.95)]
        out["posterior"][n] = {"prior_5_50_95": [round(float(v), 4) for v in pr], "post_5_50_95": [round(v, 4) for v in po],
                               "shift_sd": round(float((po[1] - pr[1]) / (x.std() + 1e-9)), 3),
                               "width_ratio": round(float((po[2] - po[0]) / (pr[2] - pr[0] + 1e-12)), 3)}
    json.dump(out, open(os.path.join(ROOT, "outputs", f"world_history_match_{tag}.json"), "w"), indent=1)
    print("test origins (weights from development):")
    for o, r in out["test"].items():
        print(" ", o, "ESS", r["ess"], {s: v["rel"] for s, v in r.items() if isinstance(v, dict)})
    print("verdict:", json.dumps(out["verdict"]))
    print("pooled ESS", out["ess_pooled"], "of", len(th))
    moved = sorted(out["posterior"].items(), key=lambda kv: -abs(kv[1]["shift_sd"]) - abs(1 - kv[1]["width_ratio"]))[:12]
    for n, v in moved:
        print(f"  {n:22s} prior {v['prior_5_50_95']}  post {v['post_5_50_95']}  shift {v['shift_sd']:+.2f} sd, width x{v['width_ratio']}")
    print("where history falls in the unweighted fans (development origins):")
    for k, v in out["emulators"].items():
        print(f"  {k:28s} obs pctile {v['obs_pctile']:.2f}  R2 {v['r2']:.2f}  top {v['top']}")


if __name__ == "__main__":
    main()
