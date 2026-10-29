"""Do shared rivers and water disputes predict interstate conflict beyond the existing pair model?

python scripts/water_dyad_test.py [--rebuild]
Rolling-origin backtests of the 5-year models (UCDP interstate 1975-2015 origins; fatal MIDs and uses of force
1950-2005) and the simulator's 1-year onset model (origins 2000-2024), with and without the water features, plus a
pair-clustered bootstrap of the Brier-score difference. Writes outputs/water_dyad_test.json.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from psychohistory import interstate as I
from psychohistory import water
from psychohistory.data import ROOT

CACHE = ROOT / "data" / "interstate_df.parquet"


def load(rebuild=False):
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    df = I.build(verbose=True)
    df.to_parquet(CACHE)
    return df


def boot(bt0, bt1, col="avg", n=1000, seed=0):
    """pair-clustered bootstrap of Brier(base) - Brier(water): positive = water helps."""
    d = bt0[["a", "b"]].copy()
    d["g"] = (bt0[col] - bt0.y) ** 2 - (bt1[col] - bt1.y) ** 2
    s = d.groupby(["a", "b"]).g.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    k = len(s)
    out = []
    for _ in range(n):
        i = rng.integers(0, k, k)
        out.append(s["sum"].to_numpy()[i].sum() / s["count"].to_numpy()[i].sum())
    return float(np.mean(d.g)), [float(np.quantile(out, .05)), float(np.quantile(out, .95))]


def run(df, target, origins, feats, label):
    res = {}
    bts = {}
    for name, f in (("base", feats), ("water", feats + water.FEATS)):
        r, bt = I.backtest(df, target, origins, f)
        res[name] = {m: {k: round(v, 4) if isinstance(v, float) else v for k, v in r[m].items()} for m in ("logit", "gbm", "avg")}
        bts[name] = bt
    # attach pair ids for clustering (backtest rows are in the order of df[te] per origin)
    ids = []
    for T in origins:
        te = (df.year >= T) & (df.year < T + 5) & df[target].notna()
        tr = (df.year + (5 if target.endswith("5y") else 1) <= T) & df[target].notna()
        if df.loc[tr, target].sum() < 10 or df.loc[te, target].sum() < 1:
            continue
        ids.append(df.loc[te, ["a", "b"]])
    ids = pd.concat(ids).reset_index(drop=True)
    b0, b1 = bts["base"].reset_index(drop=True), bts["water"].reset_index(drop=True)
    b0[["a", "b"]] = ids
    res["brier_gain"] = {c: boot(b0, b1, c) for c in ("avg", "logit")}
    print(label, json.dumps(res, indent=1), flush=True)
    return res


def coefs(df, target, feats):
    tr = df[target].notna()
    X = df.loc[tr, feats + water.FEATS].to_numpy(float)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
    Z = np.nan_to_num(np.clip((X - mu) / sd, -6, 6))
    lr = LogisticRegression(C=0.3, max_iter=4000).fit(Z, df.loc[tr, target])
    return {f: round(float(c), 3) for f, c in zip(feats + water.FEATS, lr.coef_[0]) if f in water.FEATS}


def main():
    df = load("--rebuild" in sys.argv)
    df = water.add_features(df)
    print("pairs sharing a basin:", int(df[df.year == 2024].shared_basin.sum()), "of", int((df.year == 2024).sum()))
    # descriptive: raw onset rates by shared basin, among contiguous pairs (rivers mostly cross land borders)
    d = df[(df.year >= 1946) & (df.contig_land == 1) & df.interstate_5y.notna()]
    desc = d.groupby("shared_basin").interstate_5y.agg(["mean", "size"]).round(4).to_dict()
    print("contiguous pairs, P(interstate conflict in 5y) by shared basin:", desc)
    out = {"descriptive_contiguous": desc}
    out["interstate_5y"] = run(df, "interstate_5y", range(1975, 2020, 5), I.FEATS, "UCDP 5y")
    out["fatal_mid_5y"] = run(df, "fatal_mid_5y", range(1950, 2006, 5), I.FEATS, "fatal MID 5y")
    out["force_5y"] = run(df, "force_5y", range(1950, 2006, 5), I.FEATS, "force 5y")
    d1 = df[df.ongoing == 0]
    out["onset_1y"] = run(d1, "interstate_1y", range(2000, 2025, 5), I.FEATS_ON, "1y onset (simulator)")
    out["coefs"] = {t: coefs(df if t != "interstate_1y" else d1, t, I.FEATS if t != "interstate_1y" else I.FEATS_ON)
                    for t in ("interstate_5y", "fatal_mid_5y", "force_5y", "interstate_1y")}
    print("standardised logit coefficients:", json.dumps(out["coefs"], indent=1))
    json.dump(out, open(ROOT / "outputs" / "water_dyad_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
