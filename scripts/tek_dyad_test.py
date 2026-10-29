"""Transborder ethnic kin (EPR-TEK, 1946-2021): does a pair where a group rules in one state and is excluded or
discriminated in the other predict interstate conflict beyond the existing pair model? Rolling-origin backtests and
pair-clustered bootstrap, as in water_dyad_test.py / ocean_test.py.
python -P scripts/tek_dyad_test.py  -> outputs/tek_dyad_test.json
"""
import importlib.util
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

from psychohistory import interstate as I
from psychohistory.data import RAW, ROOT

FEATS = ["tek_ruling_excluded", "tek_shared"]
spec = importlib.util.spec_from_file_location("wdt", os.path.join(os.path.dirname(__file__), "water_dyad_test.py"))
wdt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wdt)

t = pd.read_csv(RAW / "structure" / "epr_tek_dyad_year.csv")
t = t.set_index(["iso3_a", "iso3_b", "year"])
df = wdt.load()
key = [(min(a, b), max(a, b), min(y, 2021)) for a, b, y in zip(df.a, df.b, df.year)]    # held at 2021 after
df["tek_ruling_excluded"] = [float(t.kin_ruling_vs_excluded.get(k, 0) or 0) for k in key]
df["tek_shared"] = [float((t.n_shared_tek_groups.get(k, 0) or 0) > 0) for k in key]
print("pairs ruling-vs-excluded in 2020:", int(df[df.year == 2020].tek_ruling_excluded.sum()), flush=True)


def run(dd, target, origins, feats, label):
    res, bts = {}, {}
    for name, f in (("base", feats), ("tek", feats + FEATS)):
        r, bt = I.backtest(dd, target, origins, f)
        res[name] = {m: {k: round(v, 4) if isinstance(v, float) else v for k, v in r[m].items()} for m in ("logit", "gbm", "avg")}
        bts[name] = bt
    ids = []
    for T in origins:
        te = (dd.year >= T) & (dd.year < T + 5) & dd[target].notna()
        tr = (dd.year + (5 if target.endswith("5y") else 1) <= T) & dd[target].notna()
        if dd.loc[tr, target].sum() < 10 or dd.loc[te, target].sum() < 1:
            continue
        ids.append(dd.loc[te, ["a", "b"]])
    ids = pd.concat(ids).reset_index(drop=True)
    b0, b1 = bts["base"].reset_index(drop=True), bts["tek"].reset_index(drop=True)
    b0[["a", "b"]] = ids
    res["brier_gain"] = {c: wdt.boot(b0, b1, c) for c in ("avg", "logit")}
    print(label, json.dumps({k: res[k] for k in ("brier_gain",)}), res["base"]["avg"], res["tek"]["avg"], flush=True)
    return res


out = {"descriptive": df[(df.year >= 1946) & df.interstate_5y.notna()].groupby("tek_ruling_excluded").interstate_5y.agg(["mean", "size"]).round(4).to_dict()}
print(out, flush=True)
out["interstate_5y"] = run(df, "interstate_5y", range(1975, 2020, 5), I.FEATS, "UCDP 5y")
out["fatal_mid_5y"] = run(df, "fatal_mid_5y", range(1950, 2006, 5), I.FEATS, "fatal MID 5y")
out["force_5y"] = run(df, "force_5y", range(1950, 2006, 5), I.FEATS, "force 5y")
out["onset_1y"] = run(df[df.ongoing == 0], "interstate_1y", range(2000, 2025, 5), I.FEATS_ON, "1y onset")
json.dump(out, open(ROOT / "outputs" / "tek_dyad_test.json", "w"), indent=1)
