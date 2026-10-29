"""Narratives at the scale where event signals live: does rhetoric (verbal conflict in machine-coded events: accusations,
demands, threats, rejections) improve the monthly conflict forecasts of scripts/monthly_nowcast.py?
Target: >= 25 state-based battle deaths in the country in a month (UCDP GED 26.1), 1-12 months ahead.
Base features: the nowcast's fighting history, structural variables and the horizon. Added: verbal conflict as a share
of all events (3-month mean), its surge (log of the 3-month mean against the previous 12 months), and the same for
material conflict, so rhetoric is compared with what the event data already add. ICEWS 1995-2022, POLECAT 2023-2024
(shares and surges are within-source, so the splice matters little).
Rolling test years: each year Y trained on months to December Y-1 (from 1996, both variants), forecasts issued from
December Y-1 for its 12 months. Development Y = 2003-2011, test Y = 2013-2023 (odd years). Brier and log-loss, all
country-months and those without fighting in the previous 12 months.
python -P scripts/monthly_rhetoric_test.py -> outputs/new_data/monthly_rhetoric.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import lightgbm as lgb
import numpy as np
import pandas as pd

from psychohistory.data import get_panel
from psychohistory.features import base_arrays

O = os.path.join(ROOT, "data", "raw", "outcomes")
EV = os.path.join(ROOT, "data", "raw", "events")
Y0 = 1989


def roll(a, n, f=np.nanmean):
    out = np.full(a.shape, np.nan)
    for t in range(n - 1, a.shape[1]):
        out[:, t] = f(a[:, t - n + 1:t + 1], axis=1)
    return out


def main():
    vw = pd.read_csv(os.path.join(O, "views_2025_cm.csv"))
    gw2iso = dict(zip(vw.gwcode, vw.iso3))
    ged = pd.read_csv(os.path.join(O, "ucdp", "ged261-csv", "GEDEvent_v26_1.csv"),
                      usecols=["type_of_violence", "country_id", "date_start", "best"])
    ged = ged[ged.type_of_violence == 1]
    ged["iso3"] = ged.country_id.map(gw2iso)
    ged = ged.dropna(subset=["iso3"])
    d = pd.to_datetime(ged.date_start)
    ged["m"] = (d.dt.year - Y0) * 12 + d.dt.month - 1
    codes = sorted(vw.iso3.unique())
    ci = {c: i for i, c in enumerate(codes)}
    M = (2025 - Y0 + 1) * 12
    D = np.zeros((len(codes), M))
    for (c, m), v in ged.groupby(["iso3", "m"]).best.sum().items():
        if c in ci and m < M:
            D[ci[c], m] = v
    ld, hit = np.log1p(D), (D >= 25).astype(float)
    feats = {"ld": ld, "ld3": roll(ld, 3), "ld12": roll(ld, 12), "max12": roll(ld, 12, np.nanmax), "hit24": roll(hit, 24)}
    since = np.full(D.shape, 240.0)
    run = np.full(len(codes), 240.0)
    for t in range(M):
        run = np.where(hit[:, t] > 0, 0, np.minimum(run + 1, 240))
        since[:, t] = run
    feats["since"] = np.log1p(since)
    p = get_panel()
    b = base_arrays(p)
    for name, arr in [("lg", b["lg"]), ("lpop", b["lpop"]), ("dem", b["dem"]), ("conf", b["conflict"])]:
        a = np.full(D.shape, np.nan)
        for c, i in ci.items():
            if c in p.codes:
                row = arr[p.codes.index(c)]
                for t in range(M):
                    y = Y0 + t // 12 - 1
                    if p.years[0] <= y <= 2024:
                        a[i, t] = row[p.yi(y)]
        feats[name] = a
    # event data, country-month
    ev = pd.concat([pd.read_csv(os.path.join(EV, "icews_cm.csv")).query("year <= 2022"),
                    pd.read_csv(os.path.join(EV, "polecat_cm.csv")).query("year >= 2023")])
    ev["m"] = (ev.year - Y0) * 12 + ev.month - 1
    E = {k: np.full(D.shape, np.nan) for k in ("n_events", "n_verbal_conflict", "n_material_conflict")}
    for k in E:
        for c, m, v in zip(ev.iso3, ev.m, ev[k]):
            if c in ci and 0 <= m < M:
                E[k][ci[c], m] = v
    share = lambda k: E[k] / np.maximum(E["n_events"], 1)
    surge = lambda k: np.log1p(roll(E[k], 3)) - np.log1p(np.concatenate([np.full((len(codes), 3), np.nan), roll(E[k], 12)[:, :-3]], 1))
    rhet = {"verbal_share3": roll(share("n_verbal_conflict"), 3), "verbal_surge": surge("n_verbal_conflict"),
            "material_share3": roll(share("n_material_conflict"), 3), "material_surge": surge("n_material_conflict")}
    variants = {"base": list(feats), "material": list(feats) + ["material_share3", "material_surge"],
                "rhetoric": list(feats) + list(rhet)}
    allf = {**feats, **rhet}
    first = (1996 - Y0) * 12 + 3
    res = {}
    for Y in range(2003, 2024, 2):
        last_train = (Y - Y0) * 12 - 1                         # December Y-1
        for v, F in variants.items():
            Xs, ys = [], []
            for h in range(1, 13):
                for t in range(first, last_train - h + 1):
                    Xs.append(np.stack([allf[f][:, t] for f in F] + [np.full(len(codes), h)], 1)); ys.append(hit[:, t + h])
            m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, reg_lambda=10, verbose=-1).fit(np.concatenate(Xs), np.concatenate(ys))
            for h in range(1, 13):
                pr = m.predict_proba(np.stack([allf[f][:, last_train] for f in F] + [np.full(len(codes), h)], 1))[:, 1]
                y = hit[:, last_train + h]
                new = hit[:, last_train - 11:last_train + 1].max(1) == 0
                res.setdefault(v, []).append(pd.DataFrame({"Y": Y, "h": h, "p": pr, "y": y, "new": new}))
        print("year", Y, flush=True)
    out = {}
    for v, parts in res.items():
        df = pd.concat(parts)
        for split, sel in (("dev", df.Y <= 2011), ("test", df.Y >= 2013)):
            for sub, s2 in (("all", sel), ("new", sel & df.new)):
                x = df[s2]
                pc = x.p.clip(1e-4, 1 - 1e-4)
                out[f"{v}:{split}:{sub}"] = {"brier": round(float(np.mean((x.p - x.y) ** 2)), 6),
                                             "logloss": round(float(-np.mean(x.y * np.log(pc) + (1 - x.y) * np.log(1 - pc))), 5),
                                             "events": int(x.y.sum()), "n": len(x)}
    for k, v in out.items():
        print(k, v)
    json.dump(out, open(os.path.join(ROOT, "outputs", "new_data", "monthly_rhetoric.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
