"""Is the year-1 alignment's shift right, and would solving it on the previous decade's own risk sets be closer?

The simulator matches each hazard's level to the previous decade's rate with one log-odds shift, solved on the
ORIGIN's risk set (countries at peace for onsets, at war for endings). After a turbulent decade the countries that
started wars are at war at the origin, so the peace set is safer than the decade's average peace-year and the shift
overshoots (the composition hypothesis, debate 2026-11-05). Three shifts per origin and hazard, from the forecast's
own simulator network at that origin (pipeline.build_sim_env, cached):
- current: solved on the origin's risk set to the previous decade's pooled rate (as simulate does);
- matched: solved on the previous decade's own yearly risk sets (each year's network predictions on that year's
  features) to the same rate: a period effect with the composition held fixed;
- oracle: solved on the next decade's yearly risk sets (actual features) to what actually happened in it.
Reported: |current - oracle| and |matched - oracle| by decade of origin. The matched shift earns a simulator test only
if it is closer in at least 3 of 4 decades. CPU only (the networks are cached; nothing is trained).
Caveat: the previous decade's rows are inside the network's training data; the mean level, not each row, is what
the shift uses.
python -P scripts/align_oracle.py [SEED] -> outputs/adopt/align_oracle.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")


def solve(z, target):
    lo, hi = -4.0, 4.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(z + mid)))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    import pickle
    from psychohistory import pipeline as PL
    from psychohistory.features import TASKS
    from psychohistory.data import LAST_OBS_YEAR
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    p, base, X, Yl = PL.prepare(False)
    cf = base["conflict"]
    k_on, k_te = TASKS.index("onset_1y"), TASKS.index("termination_1y")
    lg = lambda v: np.log(np.clip(v, 1e-5, 1 - 1e-5) / (1 - np.clip(v, 1e-5, 1 - 1e-5)))
    envs = sorted(f for f in os.listdir(os.path.join(ROOT, "outputs", "sim_env")) if f.endswith(f"_m10_s{seed}.pkl"))
    res = {}
    for f in envs:
        O = int(f.split("_")[0])
        net = pickle.load(open(os.path.join(ROOT, "outputs", "sim_env", f), "rb"))
        t0 = p.yi(O)

        def rows(y0, y1, hazard):
            """network log-odds and outcomes over the risk sets of years y0..y1-1 (outcome in the next year)"""
            zs, ys = [], []
            for y in range(y0, y1):
                t = p.yi(y)
                if t + 1 >= cf.shape[1]:
                    break
                now, nxt = cf[:, t], cf[:, t + 1]
                risk = np.isfinite(now) & np.isfinite(nxt) & ((now == 0) if hazard == "on" else (now == 1))
                if not risk.any():
                    continue
                pr = net.predict(X[:, t])[:, k_on if hazard == "on" else k_te]
                ok = risk & np.isfinite(pr)
                zs.append(lg(pr[ok]))
                ys.append((nxt[ok] == 1) if hazard == "on" else (nxt[ok] == 0))
            return (np.concatenate(zs), np.concatenate(ys).astype(float)) if zs else (np.array([]), np.array([]))

        r = {}
        for hz in ("on", "te"):
            zp, yp = rows(O - 10, O, hz)                         # previous decade (the pooled target as simulate's _rates)
            target = float(yp.mean())
            now = cf[:, t0]
            risk0 = np.isfinite(now) & ((now == 0) if hz == "on" else (now == 1))
            z0 = lg(net.predict(X[:, t0])[:, k_on if hz == "on" else k_te])[risk0]
            zf, yf = rows(O, min(O + 10, LAST_OBS_YEAR), hz)
            cur, mat = solve(z0, target), solve(zp, target)
            ora = solve(zf, float(yf.mean())) if len(yf) else np.nan
            r[hz] = {"target": round(target, 4), "future_rate": round(float(yf.mean()), 4) if len(yf) else None,
                     "current": round(cur, 3), "matched": round(mat, 3), "oracle": round(float(ora), 3),
                     "err_current": round(abs(cur - ora), 3), "err_matched": round(abs(mat - ora), 3)}
        res[O] = r
        print(O, json.dumps(r), flush=True)
    summ = {}
    for hz in ("on", "te"):
        by = {}
        for O, r in res.items():
            d = f"{(O // 10) * 10}s"
            by.setdefault(d, []).append((r[hz]["err_current"], r[hz]["err_matched"]))
        dec = {d: {"err_current": round(float(np.mean([a for a, _ in v])), 3), "err_matched": round(float(np.mean([b for _, b in v])), 3),
                   "n": len(v)} for d, v in sorted(by.items())}
        wins = sum(v["err_matched"] < v["err_current"] for v in dec.values())
        summ[hz] = {"by_decade": dec, "decades_matched_closer": wins, "of": len(dec),
                    "earns_simulator_test": wins >= 3}
    out = {"seed": seed, "by_origin": res, "summary": summ}
    os.makedirs(os.path.join(ROOT, "outputs", "adopt"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "outputs", "adopt", "align_oracle.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
