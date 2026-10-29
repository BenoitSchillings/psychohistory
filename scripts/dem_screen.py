"""Which variables would improve the simulator's democracy equation (simulate.DEM_FEATURES, a ridge model)? The same
screen as scripts/growth_screen.py with the electoral democracy index in place of log GDP per person:
  g1   next-year change net of each year's world mean;  g10  mean yearly change over the next 10 years
Development origins 1975-1995, test 2000-2014.
python -P scripts/dem_screen.py [combos] -> outputs/dem_screen[_combos].json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn.linear_model import Ridge

from psychohistory.data import ROOT
from psychohistory.features import FEATURES
from psychohistory.pipeline import prepare
from psychohistory.simulate import DEM_FEATURES as GROWTH_FEATURES

DEV, TEST = (1975, 1980, 1985, 1990, 1995), (2000, 2005, 2010, 2014)


def main():
    p, base, X, Yl = prepare()
    X = X.cpu().numpy() if hasattr(X, "cpu") else np.asarray(X)
    names = list(FEATURES)
    cands = [c for c in FEATURES if c not in GROWTH_FEATURES]
    if "combos" in sys.argv:
        cands = [["freexp_change_5y", "neopatrimonialism"], ["freexp_change_5y", "csrepress_change_5y"],
                 ["freexp_change_5y", "neopatrimonialism", "csrepress_change_5y", "cso_participation"],
                 ["freexp_change_5y", "neopatrimonialism", "csrepress_change_5y", "cso_participation", "diag_acc",
                  "civil_liberties", "legit_personalist", "religious_repression"]]
    if "invest" in sys.argv:
        import pandas as pd
        d = pd.read_csv(os.path.join(ROOT, "data", "raw", "newdata", "pwt", "wdi_invest_cy.csv"))
        piv = lambda c: pd.DataFrame(d.pivot_table(index="iso3", columns="year", values=c).reindex(
            index=list(p.codes), columns=list(p.years)).to_numpy(float)).T.ffill(limit=3).T.to_numpy(float)
        gi, fi, fdi = np.clip(piv("NE.GDI.TOTL.ZS"), 0, 80), np.clip(piv("NE.GDI.FTOT.ZS"), 0, 80), np.clip(piv("BX.KLT.DINV.WD.GD.ZS"), -20, 40)
        lag5 = np.concatenate([np.full((gi.shape[0], 5), np.nan), gi[:, :-5]], 1)
        new = {"investment": gi, "fixed_investment": fi, "fdi_inflow": fdi, "investment_change_5y": gi - lag5}
        X = np.concatenate([X, np.stack(list(new.values()), -1)], -1)
        names += list(new)
        cands = list(new) + [list(new)[:2]]
    lg = base["dem"]
    C, T, F = X.shape
    yrs = np.asarray(p.years)
    g1 = np.full((C, T), np.nan)
    g1[:, :-1] = lg[:, 1:] - lg[:, :-1]
    g1 = np.clip(g1, -0.5, 0.5)
    g10 = np.full((C, T), np.nan)
    for t in range(T - 1):
        e = min(t + 10, T - 1)
        g10[:, t] = (lg[:, e] - lg[:, t]) / (e - t)
    last = int(np.max(yrs[np.isfinite(lg).any(0)]))
    cy = np.broadcast_to(yrs[None], (C, T))

    def score(cols, origins, target):
        ci = [names.index(c) for c in cols]
        se, n = 0.0, 0
        for O in origins:
            h = 1 if target == "g1" else 10
            y = g1 if target == "g1" else g10
            tr = (cy >= 1955) & (cy + h <= O) & np.isfinite(y)
            te = (cy >= O) & (cy < O + 5) & (cy + 1 <= last) & np.isfinite(y)
            if target == "g10":
                te = (cy == O) & np.isfinite(y)
            Xtr, Xte = X[tr][:, ci], X[te][:, ci]
            mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-6
            z = lambda A: np.nan_to_num(np.clip((A - mu) / sd, -6, 6))
            ytr, yte = y[tr], y[te]
            if target == "g1":           # net of each year's world mean
                for yy in np.unique(cy[tr]):
                    s = cy[tr] == yy
                    ytr[s] -= ytr[s].mean()
                for yy in np.unique(cy[te]):
                    s = cy[te] == yy
                    yte[s] -= yte[s].mean()
            sq = [cols.index("democracy")]
            Ztr, Zte = z(Xtr), z(Xte)
            Ztr, Zte = np.concatenate([Ztr, Ztr[:, sq] ** 2], 1), np.concatenate([Zte, Zte[:, sq] ** 2], 1)
            m = Ridge(alpha=10.0).fit(Ztr, ytr)
            r = yte - m.predict(Zte)
            if target == "g10":
                r = r - r.mean()          # level of world growth is not the question
            se += float(np.sum(r ** 2)); n += len(r)
        return se / n
    res = {}
    base_s = {f"{t}:{s}": score(GROWTH_FEATURES, o, t) for t in ("g1", "g10") for s, o in (("dev", DEV), ("test", TEST))}
    res["base"] = base_s
    print("base", base_s, flush=True)
    for c in cands:
        add = c if isinstance(c, list) else [c]
        c = "+".join(add)
        sc = {k: score(GROWTH_FEATURES + add, o, k.split(":")[0]) for k, o in
              (("g1:dev", DEV), ("g1:test", TEST), ("g10:dev", DEV), ("g10:test", TEST))}
        rel = {k: round(100 * (v / base_s[k] - 1), 2) for k, v in sc.items()}
        res[c] = rel
    ok = {c: v for c, v in res.items() if c != "base" and all(x < 0 for x in v.values())}
    print("improve all four (% change in MSE):")
    for c, v in sorted(ok.items(), key=lambda kv: kv[1]["g10:dev"]):
        print(f"  {c:28s} {v}")
    best10 = sorted(((c, v) for c, v in res.items() if c != "base"), key=lambda kv: kv[1]["g10:dev"])[:12]
    print("best on 10-year growth (dev):")
    for c, v in best10:
        print(f"  {c:28s} {v}")
    json.dump(res, open(os.path.join(ROOT, "outputs", "dem_screen_combos.json" if "combos" in sys.argv else "dem_screen.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
