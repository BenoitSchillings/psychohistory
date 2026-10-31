"""Screen new data sources against the risk models' out-of-sample errors (the residual_screen.py test): for each task and
candidate z, logit(y ~ logit(p) + z) on the tuned GBM's out-of-sample predictions p (z standardised, errors clustered by
country). Development: origins 1975-2005; confirmation: the 2010/2015 lockbox. A candidate passes if |t| > 3.3 on
development and the lockbox coefficient has the same sign with p < 0.05. A passing candidate still has to improve the
backtests before it enters a model.

python -P scripts/candidate_screen.py SOURCE [SOURCE ...] -> outputs/candidate_screen_<SOURCES>.json
SOURCE: gdd (IMF Global Debt Database: private, household and corporate debt), plad (leaders' ethnic groups),
labour (Penn World Table labour share)"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import statsmodels.api as sm

from psychohistory.data import RAW, ROOT
from psychohistory.evaluate import evaluate, DEV_ORIGINS, LOCK_ORIGINS
from psychohistory.features import TASKS
from psychohistory.pipeline import prepare
from psychohistory import models as M
from exp import Wrap, TUNED_GBM


def _grid(p, by_code_year):
    """{code: {year: value}} -> [C, Y]"""
    a = np.full((len(p.codes), len(p.years)), np.nan)
    yi = {int(y): j for j, y in enumerate(p.years)}
    for i, c in enumerate(p.codes):
        for y, v in by_code_year.get(c, {}).items():
            if int(y) in yi and v is not None:
                a[i, yi[int(y)]] = float(v)
    return a


def _change(a, k):
    return a - np.concatenate([np.full((a.shape[0], k), np.nan), a[:, :-k]], 1)


def gdd(p):
    out = {}
    for k in ("PVD_LS", "HH_LS", "NFC_LS"):
        a = _grid(p, json.load(open(RAW / "finance_gdd" / f"{k}.json"))["values"][k])
        out[f"gdd_{k}"] = a
        out[f"gdd_{k}_chg5"] = _change(a, 5)          # a credit boom: the rise in debt / GDP over five years
        out[f"gdd_{k}_chg3"] = _change(a, 3)
    return out


def plad(p):
    """PLAD v7 (CC0; leaders 1989-2023): an ethnic change at the top (the new effective leader's group differs from the
    predecessor's) in the past 3 years, and the population share (EPR) of the leader's group"""
    import pandas as pd
    d = pd.read_stata(RAW / "politics" / "plad" / "PLAD_April_2024.tab", convert_categoricals=False)
    d = d.dropna(subset=["startyear"]).sort_values(["idacr", "startdate", "startyear"])
    epr = pd.read_csv(RAW / "EPR-2021.csv")
    cw = pd.read_csv(RAW / "politics" / "cow_to_iso3.csv")
    g2i = dict(zip(cw.ccode, cw.iso3))
    epr["iso3"] = epr.gwid.map(g2i)
    size = {}
    for r in epr.itertuples():
        for y in range(int(r._3), int(r.to) + 1):
            size[(r.iso3, str(r.group).strip().lower(), y)] = r.size
    yi = {int(y): j for j, y in enumerate(p.years)}
    ci = {c: i for i, c in enumerate(p.codes)}
    change = np.full((len(p.codes), len(p.years)), np.nan)
    gsize = np.full((len(p.codes), len(p.years)), np.nan)
    for c, g in d.groupby("idacr"):
        if c not in ci:
            continue
        i = ci[c]
        rows = g.to_dict("records")
        for y in range(1989, 2024):
            if y not in yi:
                continue
            cur = [r for r in rows if r["startyear"] <= y <= (r["endyear"] if np.isfinite(r["endyear"]) else 2023)]
            if not cur:
                continue
            k = rows.index(cur[-1])
            eth = str(cur[-1]["ethnicity_geoepr1"] or "").strip().lower()
            gsize[i, yi[y]] = size.get((c, eth, y), np.nan)
            # a change of group among leaders who took office in the past 3 years (y-2..y)
            ch = 0.0
            for kk in range(1, len(rows)):
                if y - 2 <= rows[kk]["startyear"] <= y:
                    a_, b_ = str(rows[kk - 1]["ethnicity_geoepr1"] or ""), str(rows[kk]["ethnicity_geoepr1"] or "")
                    if a_ and b_ and a_ != b_:
                        ch = 1.0
            change[i, yi[y]] = ch
    return {"plad_eth_change3": change, "plad_leader_group_size": gsize, "plad_leader_minority": (gsize < 0.5).astype(float) * np.where(np.isfinite(gsize), 1, np.nan)}


def labour(p):
    """Penn World Table 10.01 labour share of GDP (to 2019, carried 3 years) and its 5-year change: a falling labour share
    is the measurable trace of rents moving to capital and to the firms that own it"""
    import pandas as pd
    d = pd.read_csv(RAW / "newdata" / "pwt" / "pwt1001_cy.csv", usecols=["countrycode", "year", "labsh"])
    a = pd.DataFrame(d.pivot_table(index="countrycode", columns="year", values="labsh").reindex(
        index=list(p.codes), columns=list(p.years)).to_numpy(float)).T.ffill(limit=3).T.to_numpy(float)
    return {"labour_share": a, "labour_share_chg5": _change(a, 5)}


SOURCES = {"gdd": gdd, "plad": plad, "labour": labour}


def main():
    names = sys.argv[1:]
    p, base, X, Yl = prepare()
    cand = {}
    for s in names:
        cand.update(SOURCES[s](p))
    print(len(cand), "candidates", flush=True)
    fac = lambda: Wrap(M.GBMModel(TUNED_GBM))
    yi = {int(y): i for i, y in enumerate(p.years)}
    res = {}
    from psychohistory.backtest import flatten
    Xf, Yf, ci, year = flatten(p, X, Yl)
    for split, origins in (("dev", DEV_ORIGINS), ("lock", LOCK_ORIGINS)):
        _, _, preds = evaluate(p, X, Yl, fac, origins)
        Y, P, C, Yr = [], [], [], []
        for T, y_, pr, _, c_ in preds:
            te = (year >= T) & (year < T + 5)
            Y.append(y_), P.append(pr), C.append(c_), Yr.append(year[te])
        Y, P, C, Yr = map(np.concatenate, (Y, P, C, Yr))
        ti = np.array([yi[int(y)] for y in Yr])
        for k, task in enumerate(TASKS):
            for name, arr in cand.items():
                z = arr[C, ti]
                y, pr = Y[:, k], np.clip(P[:, k], 1e-4, 1 - 1e-4)
                ok = np.isfinite(y) & np.isfinite(z)
                if ok.sum() < 500 or y[ok].sum() < 20 or np.std(z[ok]) == 0:
                    continue
                Z = np.column_stack([np.log(pr[ok] / (1 - pr[ok])), (z[ok] - z[ok].mean()) / z[ok].std()])
                try:
                    m = sm.Logit(y[ok], sm.add_constant(Z)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": C[ok]})
                except Exception:
                    continue
                res.setdefault(f"{task}:{name}", {})[split] = {"coef": round(float(m.params[2]), 3), "t": round(float(m.tvalues[2]), 2),
                                                              "p": float(m.pvalues[2]), "n": int(ok.sum()), "events": int(y[ok].sum())}
        print(split, "done", flush=True)
    passed = {k: v for k, v in res.items() if "lock" in v and "dev" in v and abs(v["dev"]["t"]) > 3.3
              and np.sign(v["dev"]["coef"]) == np.sign(v["lock"]["coef"]) and v["lock"]["p"] < 0.05}
    dev_hits = sum(abs(v["dev"]["t"]) > 3.3 for v in res.values() if "dev" in v)
    print(f"{len(res)} tests, {dev_hits} pass development, {len(passed)} confirmed on the lockbox")
    for k, v in sorted(res.items(), key=lambda kv: -abs(kv[1].get("dev", {"t": 0})["t"]))[:25]:
        d, l = v.get("dev", {}), v.get("lock", {})
        print(f"  {'*' if k in passed else ' '} {k:45s} dev {d.get('coef', np.nan):+.3f} (t {d.get('t', np.nan):+.1f}, n {d.get('n', 0)})"
              f"  lock {l.get('coef', np.nan):+.3f} (p {l.get('p', np.nan):.3f})")
    json.dump({"tests": res, "passed": sorted(passed), "dev_hits": dev_hits},
              open(os.path.join(ROOT, "outputs", f"candidate_screen_{'_'.join(names)}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
