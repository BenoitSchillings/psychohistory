"""Does the interstate onset calibration rule work on history? The simulator scales the 1-year pair model's onset hazard by
observed / predicted onsets over the previous decade (calib_recent), drifting to the 1946-origin ratio (calib_long,
out-of-sample) with a 10-year half-life (0.93 a year). For each origin, the 1-year model is fitted on years before it,
both ratios are computed from pre-origin data only, and each rule's predicted number of onsets over the next 10 years
(pairs at peace, not both democracies; features as realised) is compared with what happened.
Rules: none (ratio 1), recent, long, drift (0.93^j), fast drift (0.80^j).
python -P scripts/interstate_calib_test.py -> outputs/interstate_calib_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import interstate as I

ROOT = os.path.join(os.path.dirname(__file__), "..")
RULES = {"none": lambda r, l, j: 1.0, "recent": lambda r, l, j: r, "long": lambda r, l, j: l,
         "drift": lambda r, l, j: np.exp(np.log(l) + (np.log(r) - np.log(l)) * 0.93 ** j),
         "fast_drift": lambda r, l, j: np.exp(np.log(l) + (np.log(r) - np.log(l)) * 0.80 ** j)}


def main():
    df = I.build(False)
    X = lambda s: s[I.FEATS_ON].to_numpy(float)
    risk = lambda s: s[(s.ongoing == 0) & ~(s.dem_min > 0.6) & s.interstate_1y.notna()]
    out = {}
    for O in list(range(1975, 2011, 5)) + [2014]:
        t1 = df.interstate_1y.notna() & (df.year + 1 <= O) & (df.ongoing == 0)
        m = I.DyadModel().fit(X(df[t1]), df.loc[t1, "interstate_1y"].to_numpy())
        rec = risk(df[(df.year >= O - 10) & (df.year < O)])
        r_recent = (rec.interstate_1y.sum() + 0.5) / max(m.predict(X(rec)).sum(), 1e-6)
        num = den = 0.0
        for lo_y in range(1946, O, 10):                      # out-of-sample long-run ratio, as in sim_inputs
            te = t1 & (df.year >= lo_y) & (df.year < min(lo_y + 10, O))
            if not te.any():
                continue
            tr_ = t1 & ~((df.year >= lo_y - 1) & (df.year < lo_y + 11))
            mb = I.DyadModel().fit(X(df[tr_]), df.loc[tr_, "interstate_1y"].to_numpy())
            sub = risk(df[te])
            num += sub.interstate_1y.sum()
            den += mb.predict(X(sub)).sum()
        r_long = (num + 0.5) / max(den, 1e-6)
        res = {"calib_recent": round(float(r_recent), 3), "calib_long": round(float(r_long), 3)}
        act, pred = 0.0, {k: 0.0 for k in RULES}
        for j in range(10):
            sub = risk(df[df.year == O + j])
            if not len(sub):
                continue
            p = m.predict(X(sub))
            act += sub.interstate_1y.sum()
            for k, f in RULES.items():
                pred[k] += float(np.sum(1 - (1 - np.clip(p, 0, 1)) ** f(r_recent, r_long, j)))   # ratio on the hazard
        res["actual"] = float(act)
        res["pred"] = {k: round(v, 2) for k, v in pred.items()}
        out[O] = res
        print(O, json.dumps(res), flush=True)
    # summary: mean absolute log error of the 10-year count, and Poisson deviance
    summ = {}
    for k in RULES:
        le = [abs(np.log((v["pred"][k] + 0.5) / (v["actual"] + 0.5))) for v in out.values()]
        dev = [2 * (v["actual"] * np.log((v["actual"] + 1e-9) / v["pred"][k]) - (v["actual"] - v["pred"][k])) for v in out.values()]
        summ[k] = {"mean_abs_log_err": round(float(np.mean(le)), 3), "poisson_dev": round(float(np.sum(dev)), 2),
                   "dev_origins_1975_2005": round(float(np.mean([abs(np.log((v["pred"][k] + 0.5) / (v["actual"] + 0.5)))
                                                                for o, v in out.items() if o <= 2005])), 3),
                   "test_origins_2010_2014": round(float(np.mean([abs(np.log((v["pred"][k] + 0.5) / (v["actual"] + 0.5)))
                                                                 for o, v in out.items() if o >= 2010])), 3)}
    print(json.dumps(summ, indent=1))
    json.dump({"by_origin": out, "summary": summ}, open(os.path.join(ROOT, "outputs", "interstate_calib_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
