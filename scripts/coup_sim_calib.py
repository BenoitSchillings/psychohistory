"""The simulator's country coup risks (debate 4 item 2; 10 November 2026). Since version 55 the five-year coup model is
recalibrated (pipeline.COUP_RECAL), but the simulator's coup counts are aligned separately: its 2030 coup chances
(Guinea 44%, Madagascar 48%) sit well above the five-year model's (24%, 23%). The world level is pinned by the year-1
alignment, so the question is the spread across countries: the slope of the simulator's coup log-odds.

Step 1 (diagnosis, report only): from the baseline hindcasts (TAG_s0..s2, 16 origins 1970-2009, the stored
coup_p5 / coup_y5 per country: the simulated chance of a coup attempt within five years and the outcome), calibration by
band, and on the same country-origins the five-year ensemble's coup_5y backtest forecast (origins it covers).
Step 2 (the slope, leave one decade out): convert each p5 to an annual hazard h = 1 - (1 - p5)^(1/5); for each decade D
of origins fit y5 ~ 1 - (1 - sigmoid(a + b logit h))^5 by maximum likelihood on the other three decades; COUP_SLOPE for
D is b. Written to outputs/coup_slope_lodo.json for world_hindcast --consts.
Pre-registered decision (10 November 2026, before any variant run): the variant (COUP_SLOPE by decade, 3 seeds) is
adopted for February if, against the baseline, (i) country five-year coup Brier AND log loss improve, pooled over the 16
origins and 3 seeds, (ii) Brier improves in at least 3 of the 4 decades, and (iii) scripts/adopt_score.py shows no
guardrail breach and a composite no worse than minus the noise floor.
python -P scripts/coup_sim_calib.py fit BASE            -> diagnosis + outputs/coup_slope_lodo.json
python -P scripts/coup_sim_calib.py score BASE VARIANT  -> outputs/coup_sim_calib_BASE_VARIANT.json"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = os.path.join(os.path.dirname(__file__), "..")
HC = os.path.join(ROOT, "outputs", "world_hindcast")
lg = lambda q: np.log(np.clip(q, 1e-5, 1 - 1e-5) / (1 - np.clip(q, 1e-5, 1 - 1e-5)))
sg = lambda z: 1 / (1 + np.exp(-z))


def load(tag):
    rows = []
    for f in sorted(glob.glob(os.path.join(HC, f"{tag}_s[0-9]_*.npz"))):
        a = np.load(f, allow_pickle=True)
        if "coup_p5" not in a.files:
            continue
        seed = int(os.path.basename(f).split("_s")[1].split("_")[0])
        for c, p_, y_ in zip(a["coup_codes"], a["coup_p5"], a["coup_y5"]):
            rows.append((str(c), int(a["origin"]), seed, float(p_), float(y_)))
    d = pd.DataFrame(rows, columns=["code", "origin", "seed", "p5", "y5"])
    d["decade"] = (d.origin // 10 * 10).astype(str)
    return d


def sc(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(np.mean((p - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_ab(p5, y5):
    z = lg(1 - (1 - np.clip(p5, 1e-4, 1 - 1e-4)) ** 0.2)
    def nll(t):
        q = np.clip(1 - (1 - sg(t[0] + t[1] * z)) ** 5, 1e-6, 1 - 1e-6)
        return -np.sum(y5 * np.log(q) + (1 - y5) * np.log(1 - q))
    r = minimize(nll, [0.0, 1.0], method="Nelder-Mead")
    return [float(r.x[0]), float(r.x[1])]


def main():
    mode, base = sys.argv[1], sys.argv[2]
    d = load(base)
    if mode == "fit":
        m = d.groupby(["code", "origin", "decade"]).agg(p5=("p5", "mean"), y5=("y5", "first")).reset_index()
        res = {"rows": int(len(m)), "events": int(m.y5.sum())}
        bands = [0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.01]
        m["band"] = pd.cut(m.p5, bands, right=False)
        res["calibration"] = {str(k): [int(len(g)), round(float(g.p5.mean()), 3), round(float(g.y5.mean()), 3)]
                              for k, g in m.groupby("band", observed=True)}
        res["brier_logloss_sim"] = [round(v, 5) for v in sc(m.p5.values, m.y5.values)]
        # the five-year ensemble on the same country-origins (backtest rows: year = origin)
        bt = pd.read_parquet(os.path.join(ROOT, "outputs", "backtest_predictions.parquet"))
        bt = bt[(bt.task == "coup_5y") & (bt.model == "Seldon ensemble")][["code", "year", "p", "y"]]
        j = m.merge(bt, left_on=["code", "origin"], right_on=["code", "year"], how="inner")
        if len(j):
            res["same_rows_vs_ensemble"] = {"rows": int(len(j)), "sim": [round(v, 5) for v in sc(j.p5.values, j.y5.values)],
                                            "ensemble": [round(v, 5) for v in sc(j.p.values, j.y.values)],
                                            "label_agreement": round(float((j.y == j.y5).mean()), 3)}
        res["slope_all"] = fit_ab(m.p5.values, m.y5.values)
        lodo = {D: fit_ab(m[m.decade != D].p5.values, m[m.decade != D].y5.values) for D in sorted(m.decade.unique())}
        res["lodo_ab"] = lodo
        json.dump({D: round(v[1], 3) for D, v in lodo.items()}, open(os.path.join(ROOT, "outputs", "coup_slope_lodo.json"), "w"))
        print(json.dumps(res, indent=1, default=str))
        json.dump(res, open(os.path.join(ROOT, "outputs", f"coup_sim_calib_{base}.json"), "w"), indent=1, default=str)
        return
    var = sys.argv[3]
    v = load(var)
    j = d.merge(v, on=["code", "origin", "seed", "decade"], suffixes=("_b", "_v"))
    res = {"rows": int(len(j)), "pooled": {"base": [round(x, 5) for x in sc(j.p5_b.values, j.y5_b.values)],
                                            "variant": [round(x, 5) for x in sc(j.p5_v.values, j.y5_b.values)]}, "decades": {}}
    wins = 0
    for D, g in j.groupby("decade"):
        b_, v_ = sc(g.p5_b.values, g.y5_b.values)[0], sc(g.p5_v.values, g.y5_b.values)[0]
        res["decades"][D] = [round(b_, 5), round(v_, 5)]
        wins += v_ < b_
    P = res["pooled"]
    res["i_pooled"] = bool(P["variant"][0] < P["base"][0] and P["variant"][1] < P["base"][1])
    res["ii_decades"] = int(wins)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT, "outputs", f"coup_sim_calib_{base}_{var}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
