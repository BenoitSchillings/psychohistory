"""Why do the hindcasts give the safest countries too many civil wars? (safe20_wars: 2-3 simulated vs 0 observed,
origins 1995-2009). Static check of the risk network's low tail: train at each origin as the hindcast does, take the
origin-year onset hazard of countries at peace, apply the simulator's alignment shift, and compare the implied chance of
any onset within H years with what happened, by decile of risk and for the 20 safest.
python -P scripts/safe_tail_check.py H ORIGIN [ORIGIN ...] -> outputs/safe_tail_check.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch


def main():
    from psychohistory import pipeline as PL
    from psychohistory.backtest import flatten, train_mask
    from psychohistory.models import NET_TUNED, PsychoNet
    from psychohistory.data import LAST_OBS_YEAR
    from psychohistory.features import TASKS
    H, origins = int(sys.argv[1]), [int(a) for a in sys.argv[2:sys.argv.index("--v") if "--v" in sys.argv else None]]
    p, base, X, Yl = PL.prepare(False)
    Xf, Yf, ci, year = flatten(p, X, Yl)
    ON = TASKS.index("onset_1y")
    out = {}
    variants = sys.argv[sys.argv.index("--v") + 1].split(",") if "--v" in sys.argv else ["align", "raw", "recal", "rw20", "rw40"]
    lg = lambda q: np.log(q / (1 - q))
    sg = lambda z: 1 / (1 + np.exp(-z))

    def align(pr, i0):
        cf = base["conflict"][:, i0 - 10:i0 + 1]
        r_on = float((((cf[:, 1:] == 1) & (cf[:, :-1] == 0)).sum()) / max((cf[:, :-1] == 0).sum(), 1))
        peace = np.isfinite(base["conflict"][:, i0]) & (base["conflict"][:, i0] == 0) & np.isfinite(pr)
        z = lg(pr[peace])
        lo, hi = -4.0, 4.0
        for _ in range(40):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if sg(z + m).mean() < r_on else (lo, m)
        return sg(lg(pr) + (lo + hi) / 2)

    def train(O, hl=None):
        Ytr, tr = train_mask(Yf, year, O)
        w = None if hl is None else 0.5 ** ((O - year[tr]) / hl)
        return PsychoNet(members=8, **NET_TUNED).fit(Xf[tr], Ytr[tr], ci[tr], sample_weight=w)

    for O in origins:
        E = min(O + H, LAST_OBS_YEAR)
        i0, i1 = p.yi(O), p.yi(E)
        T = i1 - i0
        act = base["conflict"][:, i0:i1 + 1]
        any_obs = (act[:, 1:].max(1) > 0.5).astype(float)
        nets = {}
        res = {}
        for v in variants:
            hl = int(v[2:]) if v.startswith("rw") else None
            key = hl
            if key not in nets:
                nets[key] = train(O, hl)
            pr = nets[key].predict(X[:, i0])[:, ON]
            if v == "raw":
                pa = pr
            elif v == "recal":
                # slope and intercept from the previous decade, out of sample: a network trained ten years earlier,
                # its 1-year onset predictions for countries at peace in each of those ten years, against what happened
                if "prev" not in nets:
                    nets["prev"] = train(O - 10)
                zs, ys = [], []
                for t in range(i0 - 10, i0):
                    c0, c1 = base["conflict"][:, t], base["conflict"][:, t + 1]
                    q = nets["prev"].predict(X[:, t])[:, ON]
                    m_ = np.isfinite(c0) & np.isfinite(c1) & (c0 == 0) & np.isfinite(q)
                    zs.append(lg(np.clip(q[m_], 1e-5, 1 - 1e-5)))
                    ys.append(c1[m_])
                import statsmodels.api as sm
                fit = sm.Logit(np.concatenate(ys), sm.add_constant(np.concatenate(zs))).fit(disp=0)
                a, b = fit.params
                res["recal_ab"] = [round(float(a), 3), round(float(b), 3)]
                pa = sg(a + b * lg(np.clip(pr, 1e-5, 1 - 1e-5)))
            elif v.startswith("strat"):
                # calibration intercept per risk stratum from the previous decade: the origin network's predictions for
                # countries at peace in each of those years against what happened, strata by the origin's risk
                # quantiles; shrunk with `a` pseudo-events at the predicted rate (no events would give -inf)
                ns, a_ = int(v[5]), 2.0
                peace0 = np.isfinite(base["conflict"][:, i0]) & (base["conflict"][:, i0] == 0) & np.isfinite(pr)
                cuts = np.quantile(lg(pr[peace0]), np.linspace(0, 1, ns + 1)[1:-1])
                zs, ys = [], []
                for t in range(i0 - 10, i0):
                    c0, c1 = base["conflict"][:, t], base["conflict"][:, t + 1]
                    q = nets[key].predict(X[:, t])[:, ON]
                    m_ = np.isfinite(c0) & np.isfinite(c1) & (c0 == 0) & np.isfinite(q)
                    zs.append(lg(np.clip(q[m_], 1e-5, 1 - 1e-5)))
                    ys.append(c1[m_])
                zs, ys = np.concatenate(zs), np.concatenate(ys)
                st = np.digitize(zs, cuts)
                za = lg(np.clip(pr, 1e-5, 1 - 1e-5))
                sa = np.digitize(za, cuts)
                shifts = []
                for g in range(ns):
                    zg, yg = zs[st == g], ys[st == g]
                    pm = sg(zg).mean()
                    tgt = (yg.sum() + a_) / (len(yg) + a_ / pm)
                    lo, hi = -6.0, 6.0
                    for _ in range(40):
                        m = (lo + hi) / 2
                        lo, hi = (m, hi) if sg(zg + m).mean() < tgt else (lo, m)
                    shifts.append((lo + hi) / 2)
                res[v + "_shifts"] = [round(x, 2) for x in shifts]
                pa = sg(za + np.array(shifts)[sa])
            else:
                pa = align(pr, i0)
            ok = np.isfinite(act).all(1) & (act[:, 0] == 0) & np.isfinite(pa)
            any_sim = np.clip(1 - (1 - pa) ** T, 1e-4, 1 - 1e-4)
            idx = np.where(ok)[0]
            order = idx[np.argsort(pa[idx])]
            safe = order[:20]
            y, q = any_obs[idx], any_sim[idx]
            res[v] = {"brier": round(float(np.mean((q - y) ** 2)), 4),
                      "logloss": round(float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))), 4),
                      "mean": round(float(q.mean()), 3), "obs": round(float(y.mean()), 3),
                      "safe20": [round(float(any_sim[safe].sum()), 2), int(any_obs[safe].sum())],
                      "bottom_half": [round(float(any_sim[order[:len(order) // 2]].mean()), 3), round(float(any_obs[order[:len(order) // 2]].mean()), 3)]}
        out[O] = res
        print(O, json.dumps(res), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "outputs", f"safe_tail_check_H{H}_{'_'.join(variants)}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
