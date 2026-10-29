"""Head-to-head with every archived ViEWS production run (fatalities001/002/003, December 2021 .. December 2024
origins), country-month, 1-12 months ahead, scored on final UCDP GED 26.1 (through December 2025).

For each ViEWS run (named by its last month of data, the origin), our models are trained only on data that existed
then: final GED up to the last year released by the origin (each GED release comes out around June), UCDP Candidate
monthly releases after it. Retrained every run.
- Binary target (ViEWS's dichotomous output): at least 25 state-based deaths in the country-month. Ours: share of
  draws >= 25 from the fatality model (cm_fatal) and a dedicated classifier (monthly.py-style), averaged.
- Fatalities: ViEWS's point forecast of log(1 + deaths) (main_mean_ln, or sc_cm_sb_main for fatalities001) against our
  mean of log(1 + draws); mean squared error on the log scale (ViEWS's own evaluation metric).
python scripts/views_archive_benchmark.py [--step 3]  -> outputs/views_archive_benchmark.json
"""
import glob
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from psychohistory import cm_fatal as CF

ROOT = os.path.join(os.path.dirname(__file__), "..")
ARCH = os.path.join(ROOT, "data", "raw", "outcomes", "views", "archive")
LAST_EVAL = CF.mid(2025, 12)


def views_run(path):
    d = json.load(open(path))["data"]
    df = pd.DataFrame(d)
    if "main_dich" in df:
        df["p25"], df["ln"] = df.main_dich, df.main_mean_ln
    else:
        df["p25"], df["ln"] = df.sc_cm_sb_dich_main, df.sc_cm_sb_main
    return df[["country_id", "month_id", "p25", "ln"]]


def final_through(origin):
    """last month of final GED released by the origin (release for year Y around June of Y + 1)"""
    y, m = 1980 + (origin - 1) // 12, (origin - 1) % 12 + 1
    return CF.mid(y - 1 if m >= 7 else y - 2, 12)


def classifier(D, F, origin, horizons, start=CF.mid(1995, 1)):
    Xs, ys = [], []
    for h in horizons:
        ts = list(range(start, origin - h + 1))
        X, _, _ = CF.design(F, ts, [h])
        Xs.append(X)
        ys.append(np.concatenate([(D[:, t + h] >= 25).astype(float) for t in ts]))
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, verbose=-1)
    return m.fit(np.concatenate(Xs), np.concatenate(ys))


def main():
    step = int(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else 1
    runs = sorted(glob.glob(os.path.join(ARCH, "fatalities*_t01.json")))
    cs = CF.countries()
    vid2i = {v: i for i, v in enumerate(cs.views_country_id)}
    truth = CF.deaths(LAST_EVAL)                      # final GED 26.1
    rows = []
    done = 0
    for k, path in enumerate(runs):
        V = views_run(path)
        origin = int(V.month_id.min()) - 1
        if origin + 1 > LAST_EVAL or k % step:
            continue
        t0 = time.time()
        ft = final_through(origin)
        D = CF.deaths(origin, ft)
        F = CF.features(D)
        months = [m for m in range(origin + 1, min(origin + 12, LAST_EVAL) + 1)]
        draws, _ = CF.train_and_forecast(origin, months, final_through=ft)
        clf = classifier(D, F, origin, [m - origin for m in months])
        for m in months:
            Xt, _, _ = CF.design(F, [origin], [m - origin])
            pc = clf.predict_proba(Xt)[:, 1]
            dr = draws[m]
            sub = V[V.month_id == m]
            for cid, p25, ln in zip(sub.country_id, sub.p25, sub.ln):
                i = vid2i.get(int(cid))
                if i is None:
                    continue
                y = truth[i, m]
                rows.append({"run": os.path.basename(path)[:-5], "origin": origin, "month_id": m, "h": m - origin,
                             "country_id": int(cid), "y": y, "y25": float(y >= 25), "views_p25": p25, "views_ln": ln,
                             "ours_p25_draws": float((dr[i] >= 25).mean()), "ours_p25_clf": float(pc[i]),
                             "ours_ln": float(np.log1p(dr[i]).mean())})
        done += 1
        print(f"{os.path.basename(path)} origin {origin} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df["ours_p25"] = (df.ours_p25_draws + df.ours_p25_clf) / 2
    df.to_parquet(os.path.join(ROOT, "outputs", "views_archive_benchmark.parquet"))
    out = summarize(df)
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "outputs", "views_archive_benchmark.json"), "w"), indent=1)


def summarize(df):
    def block(d):
        y, ly = d.y25.to_numpy(), np.log1p(d.y.to_numpy())
        r = {"n": int(len(d)), "events": int(y.sum()), "origins": int(d.origin.nunique())}
        for nm, col in (("views", "views_p25"), ("ours", "ours_p25"), ("ours_draws", "ours_p25_draws"), ("ours_clf", "ours_p25_clf")):
            p = np.clip(d[col].to_numpy(float), 1e-4, 1 - 1e-4)
            r[nm] = {"brier": float(np.mean((p - y) ** 2)), "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                     "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None}
        r["mse_ln"] = {"views": float(np.mean((d.views_ln - ly) ** 2)), "ours": float(np.mean((d.ours_ln - ly) ** 2))}
        # paired bootstrap over origins (forecast dates) of the Brier and log-MSE differences (ours - views)
        rng = np.random.default_rng(0)
        o = d.origin.to_numpy()
        uo = np.unique(o)
        db = (np.clip(d.ours_p25, 0, 1) - y) ** 2 - (d.views_p25 - y) ** 2
        dm = (d.ours_ln - ly) ** 2 - (d.views_ln - ly) ** 2
        g = pd.DataFrame({"o": o, "db": db, "dm": dm}).groupby("o").agg(["sum", "count"])
        bs_b, bs_m = [], []
        for _ in range(2000):
            s = rng.choice(len(uo), len(uo))
            gg = g.iloc[s]
            bs_b.append(gg[("db", "sum")].sum() / gg[("db", "count")].sum())
            bs_m.append(gg[("dm", "sum")].sum() / gg[("dm", "count")].sum())
        r["diff_brier_90ci"] = [float(np.quantile(bs_b, .05)), float(np.quantile(bs_b, .95))]
        r["diff_mse_ln_90ci"] = [float(np.quantile(bs_m, .05)), float(np.quantile(bs_m, .95))]
        return r
    out = {"all": block(df)}
    for gen in ("fatalities001", "fatalities002", "fatalities003"):
        s = df[df.run.str.startswith(gen)]
        if len(s):
            out[gen] = block(s)
    for h in (1, 3, 6, 12):
        out[f"h{h}"] = block(df[df.h == h])
    return out


if __name__ == "__main__":
    main()
