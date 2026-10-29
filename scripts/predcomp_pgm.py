"""Grid level (PRIO-GRID month) of the 2023/24 VIEWS Prediction Challenge: our forecasts under the challenge protocol
and the same CRPS / ignorance / interval scores for every participant (data/raw/predcomp/pgm).

python scripts/predcomp_pgm.py run NAME      -> outputs/predcomp_pgm/NAME/pgm/window=Y..../*.parquet
python scripts/predcomp_pgm.py score         -> outputs/predcomp_pgm_leaderboard.csv / .json
"""
import gc
import glob
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from psychohistory import cm_fatal as CF, pgm_fatal as PG
sys.path.insert(0, os.path.dirname(__file__))
from predcomp_eval import crps_ensemble, ignorance, interval

ROOT = os.path.join(os.path.dirname(__file__), "..")
PC = os.path.join(ROOT, "data", "raw", "predcomp", "pgm")
WINDOWS = ["Y2018", "Y2019", "Y2020", "Y2021", "Y2022", "Y2023", "Y2024", "Y2025"]


def run(name):
    if "--nn" in sys.argv:
        PG.NN_BLEND = float(sys.argv[sys.argv.index("--nn") + 1])
    g, _, _ = PG.cells()
    windows = [(f"Y{y}", CF.mid(y - 1, 10), [CF.mid(y, m) for m in range(1, 13)], None) for y in range(2018, 2024)]
    windows.append(("LIVE", CF.mid(2024, 4), [CF.mid(2024, m) for m in range(7, 13)] + [CF.mid(2025, m) for m in range(1, 7)],
                    CF.mid(2023, 12)))                                         # GED 24.1, as in predcomp_run.py
    if "--live-only" in sys.argv:
        windows = windows[-1:]
    if "--hist-only" in sys.argv:
        windows = windows[:-1]
    for w, origin, months, ft in windows:
        t0 = time.time()
        draws, _, (nrows, _) = PG.train_and_forecast(origin, months, final_through=ft)
        groups = {"Y2024": [m for m in months if m <= CF.mid(2024, 12)], "Y2025": [m for m in months if m > CF.mid(2024, 12)]} \
            if w == "LIVE" else {w: months}
        for ww, ms in groups.items():
            n = 1000
            parts = [pd.DataFrame({"month_id": np.int32(m), "priogrid_gid": np.repeat(g.astype(np.int32), n),
                                   "draw": np.tile(np.arange(n, dtype=np.int32), len(g)), "outcome": draws[m].ravel()}) for m in ms]
            d = os.path.join(ROOT, "outputs", "predcomp_pgm", name, "pgm", f"window={ww}")
            os.makedirs(d, exist_ok=True)
            pd.concat(parts).to_parquet(os.path.join(d, f"{name}_pgm_{ww}.parquet"), index=False)
        print(f"{w}: {nrows} training rows, {time.time() - t0:.0f}s", flush=True)
        del draws
        gc.collect()


def repair(name):
    """apply the local tail bound (pgm_fatal.TAIL_K, TAIL_C0) to draws saved before it existed (an elementwise minimum,
    so the same as rerunning)"""
    g, _, _ = PG.cells()
    origins = {f"Y{y}": (CF.mid(y - 1, 10), None) for y in range(2018, 2024)}
    origins.update({"Y2024": (CF.mid(2024, 4), CF.mid(2023, 12)), "Y2025": (CF.mid(2024, 4), CF.mid(2023, 12))})
    for w, (o, ft) in origins.items():
        f = glob.glob(os.path.join(ROOT, "outputs", "predcomp_pgm", name, "pgm", f"window={w}", "*.parquet"))[0]
        b = np.maximum(PG.TAIL_C0, PG.TAIL_K * PG.local_max(PG.deaths(o, ft), o))
        bmap = pd.Series(b, index=g)
        d = pd.read_parquet(f)
        d["outcome"] = np.minimum(d.outcome.to_numpy(), bmap.reindex(d.priogrid_gid.to_numpy()).to_numpy()).astype(np.float32)
        d.to_parquet(f, index=False)
        print(w, "bounded", flush=True)


def load_matrix(path, index):
    """[len(index), draws] float32 from a submission parquet, aligned to `index` (month_id, priogrid_gid)"""
    t = pq.read_table(path)
    cols = t.column_names
    val = "outcome" if "outcome" in cols else [c for c in cols if c not in ("month_id", "priogrid_gid", "draw", "__index_level_0__")][0]
    m = t.column("month_id").to_numpy().astype(np.int64)
    c = t.column("priogrid_gid").to_numpy().astype(np.int64)
    v = np.nan_to_num(t.column(val).to_numpy().astype(np.float32))
    key = m * 1_000_000 + c
    order = np.argsort(key, kind="stable")
    key, v = key[order], v[order]
    uk, start, counts = np.unique(key, return_index=True, return_counts=True)
    n = int(np.median(counts))
    want = index[:, 0].astype(np.int64) * 1_000_000 + index[:, 1].astype(np.int64)
    pos = np.searchsorted(uk, want)
    ok = (pos < len(uk)) & (uk[np.clip(pos, 0, len(uk) - 1)] == want) & (counts[np.clip(pos, 0, len(uk) - 1)] >= n)
    out = np.zeros((len(want), n), np.float32)
    s = start[pos[ok]]
    out[ok] = v[s[:, None] + np.arange(n)[None]]
    return out, ok


def score():
    subs = {os.path.basename(d): d for d in glob.glob(os.path.join(PC, "Final Participant Submissions", "*"))}
    subs.update({"bm_" + os.path.basename(d): d for d in glob.glob(os.path.join(PC, "Benchmark Models", "Benchmark Models", "*"))})
    for d in glob.glob(os.path.join(ROOT, "outputs", "predcomp_pgm", "*")):
        subs["OURS_" + os.path.basename(d)] = d
    rows = []
    for w in WINDOWS:
        a = pq.read_table(glob.glob(os.path.join(PC, "Actuals", "pgm", f"window={w}", "*.parquet"))[0]).to_pandas().reset_index()
        idx_all = a[["month_id", "priogrid_gid"]].to_numpy()
        y_all = a.outcome.to_numpy(np.float32)
        for name, d in subs.items():
            f = glob.glob(os.path.join(d, "pgm", f"window={w}", "*.parquet"))
            if not f:
                continue
            try:
                X, ok = load_matrix(f[0], idx_all)
            except Exception as e:
                print(w, name, "unreadable", repr(e)[:80])
                continue
            if ok.sum() < 0.5 * len(ok):
                continue
            X, y = np.clip(X[ok], 0, None), y_all[ok]
            cr = np.concatenate([crps_ensemble(y[i:i + 20000], X[i:i + 20000]) for i in range(0, len(y), 20000)])
            ig = np.concatenate([ignorance(y[i:i + 20000], X[i:i + 20000]) for i in range(0, len(y), 20000)])
            mi = np.concatenate([interval(y[i:i + 20000], X[i:i + 20000]) for i in range(0, len(y), 20000)])
            rows.append({"model": name, "window": w, "crps": float(cr.mean()), "ign": float(ig.mean()), "mis": float(mi.mean()),
                         "n": int(ok.sum())})
            print(f"{w} {name:32s} CRPS {cr.mean():.4f} IGN {ig.mean():.4f} MIS {mi.mean():.3f} n {ok.sum()}", flush=True)
            del X
            gc.collect()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "outputs", "predcomp_pgm_leaderboard.csv"), index=False)
    hist = df[df.window.isin(WINDOWS[:6])].groupby("model").filter(lambda g: len(g) == 6)
    summ = hist.groupby("model")[["crps", "ign", "mis"]].mean().sort_values("crps")
    live = df[df.window.isin(["Y2024", "Y2025"])].pivot_table(index="model", columns="window", values="crps")
    print("\nHistorical 2018-2023:\n", summ.round(4).to_string(), "\n\nLive:\n", live.round(4).to_string())
    json.dump({"historical_mean": summ.round(5).reset_index().to_dict("records"), "live": live.round(5).reset_index().to_dict("records"),
               "by_window": df.round(5).to_dict("records")}, open(os.path.join(ROOT, "outputs", "predcomp_pgm_leaderboard.json"), "w"), indent=1)


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "repair": lambda: repair(sys.argv[2]), "score": score}[sys.argv[1]]()
