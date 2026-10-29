"""Score forecasts of country-month conflict fatalities exactly as the 2023/24 VIEWS Prediction Challenge does, and rank
our model among all its participants and benchmarks.

Data: the challenge replication archive (Hegre et al. 2025, JPR 62(6); data/raw/predcomp/cm: actuals and every
submission at country-month level, windows Y2018..Y2023 = calendar years forecast from data to October of the year
before, Y2024 = the true future (2024, from data to October 2023), Y2025 = January-June 2025).
Metrics (challenge code, github.com/prio-data/prediction_competition_2023): CRPS of the sample ensemble against the
observed count (properscoring/xskillscore crps_ensemble, equal weights), the binned ignorance score (bins 0, 1-2, 3-5,
6-10, 11-25, 26-50, 51-100, 101-250, 251-500, 501-1000, 1001+, each bin added once to the samples) and the 90% mean
interval score; averaged over all country-months of the window.

python scripts/predcomp_eval.py [--ours DIR]   (DIR: our submission in the same layout; default outputs/predcomp/seldon)
Writes outputs/predcomp_leaderboard.csv and .json.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
PC = os.path.join(ROOT, "data", "raw", "predcomp", "cm")
WINDOWS = ["Y2018", "Y2019", "Y2020", "Y2021", "Y2022", "Y2023", "Y2024", "Y2025"]
BINS = [0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5]   # upper-inclusive integer bins (0, 1-2, ...)


def crps_ensemble(y, X):
    """y [N], X [N, n] samples: mean |X - y| - 1/(2 n^2) sum |X_i - X_j| (properscoring's crps_ensemble)"""
    Xs = np.sort(X, 1)
    n = X.shape[1]
    w = (2 * np.arange(1, n + 1) - n - 1) / n ** 2
    return np.abs(X - y[:, None]).mean(1) - (Xs * w[None]).sum(1)


def ignorance(y, X):
    edges = np.array(BINS)
    bx = np.digitize(X, edges[1:], right=False)                       # 0..10
    by = np.digitize(y, edges[1:], right=False)
    nb = len(edges)
    counts = np.stack([(bx == k).sum(1) for k in range(nb)], 1) + 1  # each bin added once
    p = counts[np.arange(len(y)), by] / counts.sum(1)
    return -np.log2(p)


def interval(y, X, level=0.9):
    a = 1 - level
    lo, hi = np.quantile(X, a / 2, axis=1), np.quantile(X, 1 - a / 2, axis=1)
    return (hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)


def load(path):
    s = pd.read_parquet(path).reset_index()
    val = "outcome" if "outcome" in s else [c for c in s.columns if c not in ("month_id", "country_id", "draw", "index")][0]
    if "draw" not in s:
        s["draw"] = s.groupby(["month_id", "country_id"]).cumcount()
    return s.pivot_table(index=["month_id", "country_id"], columns="draw", values=val)


def score(sub_dir, w, actual):
    f = glob.glob(os.path.join(sub_dir, "cm", f"window={w}", "*.parquet"))
    if not f:
        return None
    P = load(f[0])
    A = actual.reindex(P.index).dropna()
    P = P.loc[A.index]
    if len(A) < 1000:                     # (the live window Y2024 covers July-December only)
        return None
    X = np.clip(np.nan_to_num(P.to_numpy(float)), 0, None)
    y = A.outcome.to_numpy(float)
    return {"crps": float(crps_ensemble(y, X).mean()), "ign": float(ignorance(y, X).mean()),
            "mis": float(interval(y, X).mean()), "n": int(len(y)), "draws": int(X.shape[1])}


def main():
    ours = sys.argv[sys.argv.index("--ours") + 1] if "--ours" in sys.argv else os.path.join(ROOT, "outputs", "predcomp", "seldon")
    subs = {os.path.basename(d): d for d in glob.glob(os.path.join(PC, "Final Participant Submissions", "*"))}
    subs.update({"bm_" + os.path.basename(d): d for d in glob.glob(os.path.join(PC, "Benchmark Models", "Benchmark Models", "*"))})
    for d in sorted(glob.glob(os.path.join(os.path.dirname(ours), "*"))):
        if os.path.isdir(d):
            subs["OURS_" + os.path.basename(d)] = d
    rows = []
    for w in WINDOWS:
        act = pd.read_parquet(glob.glob(os.path.join(PC, "Actuals", "cm", f"window={w}", "*.parquet"))[0])
        act = act.reset_index().set_index(["month_id", "country_id"])
        for name, d in subs.items():
            r = score(d, w, act)
            if r:
                rows.append({"model": name, "window": w, **r})
                print(f"{w} {name:40s} CRPS {r['crps']:8.2f} IGN {r['ign']:.3f} MIS {r['mis']:9.1f} (draws {r['draws']})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "outputs", "predcomp_leaderboard.csv"), index=False)
    hist = df[df.window.isin(WINDOWS[:6])]
    full = hist.groupby("model").filter(lambda g: len(g) == 6)
    summ = full.groupby("model")[["crps", "ign", "mis"]].mean().sort_values("crps")
    live = df[df.window.isin(["Y2024", "Y2025"])].pivot_table(index="model", columns="window", values="crps")
    print("\nHistorical windows 2018-2023 (mean over windows):\n", summ.round(3).to_string())
    print("\nLive windows (CRPS):\n", live.round(2).sort_values(live.columns[0]).to_string())
    json.dump({"historical_mean": summ.round(4).reset_index().to_dict("records"),
               "live": live.round(3).reset_index().to_dict("records"),
               "by_window": df.round(4).to_dict("records")},
              open(os.path.join(ROOT, "outputs", "predcomp_leaderboard.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
