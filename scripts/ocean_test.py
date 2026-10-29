"""Ocean food and overfishing, tested on history before anything goes into the simulator.

Country level (1965-2023): does fish dependence (fish share of protein supply, FAO food balances via OWID), or a fall
in national capture-fishery catch (5-year log change, FAO/World Bank via OWID) in fish-dependent countries, predict
civil-conflict onset or an economic collapse (GDP per person falling more than 5%) the next year? Logit with year
effects, clustered by country, plus an out-of-sample check (fit to 2000, score 2001-2023).
Pair level: do international fisheries conflicts (Spijkers et al. 2019, International Fishery Conflict Database,
1974-2016; events from seizures of boats to deadly clashes) predict interstate conflict, fatal MIDs or uses of force
beyond the existing pair model? Rolling-origin backtests as in water_dyad_test.py, origins inside the IFCD coverage.
python -P scripts/ocean_test.py  -> outputs/ocean_test.json
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import statsmodels.api as sm

from psychohistory.data import RAW, ROOT, get_panel
from psychohistory.features import base_arrays
from psychohistory import interstate as I

OC = RAW / "ocean"
FISH_FEATS = ["fish_conf_ever", "fish_conf_10y", "fish_conf_max_int_10y"]


def owid(name):
    d = pd.read_csv(OC / f"{name}.csv")
    return d.rename(columns={d.columns[3]: "v"}).dropna(subset=["code"]).set_index(["code", "year"]).v


def country_frame():
    """country-year: fish share of protein, catch change, outcomes and controls"""
    p = get_panel()
    b = base_arrays(p)
    Y = list(p.years)
    fish = owid("fish-and-seafood-consumption-per-capita")              # kg live weight per person per year
    prot = owid("daily-per-capita-protein-supply")                       # g protein per person per day
    catch = owid("capture-fishery-production")                           # tonnes
    rows = []
    for i, c in enumerate(p.codes):
        for t in range(1965, 2024):
            ti = Y.index(t)
            if not (np.isfinite(b["lg"][i, ti]) and np.isfinite(b["lg"][i, ti + 1])):
                continue
            f, pr = fish.get((c, t)), prot.get((c, t))
            k0, k1 = catch.get((c, t - 5)), catch.get((c, t))
            rows.append({
                "c": c, "year": t,
                # about 0.1 g protein per g of live-weight fish (edible share x protein content)
                "fish_dep": np.nan if f is None or pr is None or pr <= 0 else min(f * 100 / 365 / pr, 0.6),
                "catch_chg5": np.nan if not k0 or not k1 or k0 < 1000 else float(np.clip(np.log(k1 / k0), -2, 2)),
                "onset": float(b["conflict"][i, ti + 1] > 0 and b["conflict"][i, ti] == 0) if b["conflict"][i, ti] == 0 else np.nan,
                "collapse": float(b["lg"][i, ti + 1] - b["lg"][i, ti] < -0.05),
                "lg": b["lg"][i, ti], "lpop": b["lpop"][i, ti], "conflict_now": float(b["conflict"][i, ti] > 0),
                "dem": b["dem"][i, ti]})
    d = pd.DataFrame(rows)
    d["decline_x_dep"] = d.fish_dep * np.minimum(d.catch_chg5, 0)        # falls in catch, weighted by dependence
    return d


def country_tests(d):
    out = {}
    for outcome in ("onset", "collapse"):
        res = {}
        for name, cols in (("dependence", ["fish_dep"]), ("dependence+decline", ["fish_dep", "catch_chg5", "decline_x_dep"])):
            ctrl = ["lg", "lpop", "dem"] + ([] if outcome == "onset" else ["conflict_now"])
            sub = d.dropna(subset=[outcome] + cols + ctrl)
            sub = sub[sub.groupby("year")[outcome].transform("max") > 0]   # years with no event drop out under year effects
            fe = pd.get_dummies(sub.year, prefix="y", drop_first=True).astype(float)
            X = pd.concat([sub[cols + ctrl], fe], axis=1).assign(const=1.0)
            X = X.loc[:, X.std() > 0].assign(const=1.0)
            try:
                m = sm.Logit(sub[outcome], X).fit(disp=0, maxiter=300, cov_type="cluster",
                                                  cov_kwds={"groups": sub.c.astype("category").cat.codes})
                res[name] = {k: [round(float(m.params[k]), 3), round(float(m.bse[k]), 3)] for k in cols}
            except Exception as e:
                res[name] = repr(e)[:120]
            tr, te = sub.year <= 2000, sub.year >= 2001
            for tag, cc in (("without", ctrl), ("with", cols + ctrl)):
                Z = sm.add_constant(sub[cc])
                mo = sm.Logit(sub[outcome][tr], Z[tr]).fit(disp=0, maxiter=300)
                res[f"{name} oos_brier_{tag}"] = round(float(np.mean((mo.predict(Z[te]) - sub[outcome][te]) ** 2)), 5)
            res[f"{name} n"] = int(len(sub))
            res[f"{name} events"] = int(sub[outcome].sum())
        out[outcome] = res
        print(outcome, json.dumps(res, indent=1), flush=True)
    return out


def ifcd_pairs():
    """dyad-year table of fishery-conflict events: count and max intensity"""
    x = pd.read_excel(OC / "ifcd" / "ifcd.xlsx")
    x = x[x["Country code"].astype(str).str.len() == 3]                  # drops "EU" as an actor
    rows = []
    for eid, g in x.groupby("Event ID"):
        cs = sorted(set(g["Country code"]))
        yr = int(g["Event date (year)"].iloc[0])
        it = int(g["Intensity lvl of action type (Spijkers et al. 2018)"].max())
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                rows.append((cs[i], cs[j], yr, it))
    e = pd.DataFrame(rows, columns=["a", "b", "year", "intensity"])
    return e.groupby(["a", "b", "year"]).intensity.agg(["size", "max"]).reset_index()


def add_fish_features(df):
    e = ifcd_pairs()
    idx = {}
    for r in e.itertuples():
        idx.setdefault((r.a, r.b), []).append((r.year, r.size, r.max))
    ever, n10, m10 = [], [], []
    for a, b, y in zip(df.a, df.b, df.year):
        ev = idx.get((a, b)) or idx.get((b, a)) or []
        past = [(yy, n, m) for yy, n, m in ev if yy <= y]
        rec = [(n, m) for yy, n, m in past if yy > y - 10]
        ever.append(float(bool(past)))
        n10.append(np.log1p(sum(n for n, _ in rec)))
        m10.append(max([m for _, m in rec], default=0))
    df = df.copy()
    df["fish_conf_ever"], df["fish_conf_10y"], df["fish_conf_max_int_10y"] = ever, n10, m10
    return df


def pair_tests():
    sys.argv = [a for a in sys.argv if a != "--rebuild"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("wdt", os.path.join(os.path.dirname(__file__), "water_dyad_test.py"))
    wdt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wdt)
    df = add_fish_features(wdt.load())
    print("pairs with a fishery conflict by 2016:", int(df[df.year == 2016].fish_conf_ever.sum()), flush=True)
    d = df[(df.year >= 1975) & (df.year <= 2016) & df.interstate_5y.notna()]
    out = {"descriptive": d.groupby("fish_conf_ever")[["interstate_5y", "fatal_mid_5y", "force_5y"]].mean().round(4).to_dict()}
    print(out, flush=True)

    def run(dd, target, origins, feats, label):
        res, bts = {}, {}
        for name, f in (("base", feats), ("fish", feats + FISH_FEATS)):
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
        b0, b1 = bts["base"].reset_index(drop=True), bts["fish"].reset_index(drop=True)
        b0[["a", "b"]] = ids
        res["brier_gain"] = {c: wdt.boot(b0, b1, c) for c in ("avg", "logit")}
        print(label, json.dumps(res, indent=1), flush=True)
        return res

    # IFCD starts in 1974: origins from 1985 so every origin has at least 10 years of event history
    out["interstate_5y"] = run(df, "interstate_5y", range(1985, 2016, 5), I.FEATS, "UCDP 5y")
    out["fatal_mid_5y"] = run(df, "fatal_mid_5y", range(1985, 2006, 5), I.FEATS, "fatal MID 5y")
    out["force_5y"] = run(df, "force_5y", range(1985, 2006, 5), I.FEATS, "force 5y")
    d1 = df[(df.ongoing == 0) & (df.year <= 2017)]
    out["onset_1y"] = run(d1, "interstate_1y", range(1990, 2016, 5), I.FEATS_ON, "1y onset")
    return out


def main():
    d = country_frame()
    print("most fish-dependent 2019:", d[d.year == 2019].nlargest(12, "fish_dep")[["c", "fish_dep"]].round(2).values.tolist(), flush=True)
    out = {"country": country_tests(d)}
    if "--country-only" not in sys.argv:
        out["pair"] = pair_tests()
    json.dump(out, open(ROOT / "outputs" / "ocean_test.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
