"""Revenge and revanchism: do pairs where one side lost a war or a serious dispute to the other fight again more than
the model's existing memory (years since the last conflict, ever in conflict, rivalry, territorial threat) predicts?

Pair features, built only from events that ended by the year in question:
- war_lost: a COW interstate war between the two ended with one a winner and the other a loser (Correlates of War,
  dyadic, 1816-2007), decayed with half-life HL years since it ended
- mid_lost: a militarized dispute between them (GML MID 2.x, 1816-2010) at hostility level 4 or 5 ended in victory
  for one side or a yield by the other, decayed the same way
- humiliation: either of the two
Dyad backtests (interstate.backtest, logit + GBM) for the 5-year UCDP outcome and the 1-year onset model that drives
the simulator (pairs at peace, FEATS_ON); origins 1975-1995 dev, 2000-2015 test.
python -P scripts/revenge_dyad_test.py [HL] -> outputs/new_data/revenge_dyad.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

from psychohistory import interstate as I
from psychohistory.data import ROOT

NEW = ["war_lost", "mid_lost"]


def events(codes):
    """[(iso_a, iso_b, end_year, kind)] with a clear loser"""
    c2i = I._ccode_to_iso(set(codes))
    out = []
    w = I._rda("cow_war_inter")
    last = w.groupby(["warnum", "ccode1", "ccode2"]).agg(o1=("outcome1", "last"), o2=("outcome2", "last")).reset_index()
    # the outcome is the war's, so it is known only when the whole war ends (not when this pair stopped fighting)
    last["year"] = last.warnum.map(w.groupby("warnum").year.max())
    for r in last.itertuples():
        if {int(r.o1), int(r.o2)} == {1, 2} and int(r.ccode1) in c2i and int(r.ccode2) in c2i:
            out.append((c2i[int(r.ccode1)], c2i[int(r.ccode2)], int(r.year), "war_lost"))
    d = I._rda("gml_mid_disps")
    dd = I._rda("gml_mid_ddydisps")
    dis = d[d.outcome.isin([1, 2, 3, 4]) & (d.hostlev >= 4)].set_index("dispnum")
    end = dd.groupby("dispnum").year.max()
    # only pairs that themselves used force against each other (dyadic hostility 4-5)
    pairs = dd[dd.dispnum.isin(dis.index) & (dd.sidea1 != dd.sidea2) & (dd.hostlev >= 4)][["dispnum", "ccode1", "ccode2"]].drop_duplicates()
    for r in pairs.itertuples():
        if int(r.ccode1) in c2i and int(r.ccode2) in c2i:
            out.append((c2i[int(r.ccode1)], c2i[int(r.ccode2)], int(end[r.dispnum]), "mid_lost"))
    return out


def add(df, hl):
    ev = events(set(df.a) | set(df.b))
    by = {}
    for a, b, y, k in ev:
        by.setdefault((min(a, b), max(a, b), k), []).append(y)
    lam = np.log(2) / hl
    for k in NEW:
        v = np.zeros(len(df))
        for i, (a, b, y) in enumerate(zip(df.a.to_numpy(), df.b.to_numpy(), df.year.to_numpy())):
            ys = by.get((min(a, b), max(a, b), k))
            if ys:
                past = [y - e for e in ys if e < y]      # strictly before, like the other memory features
                if past:
                    v[i] = np.exp(-lam * min(past))
        df[k] = v
    df["humiliation"] = np.maximum(df.war_lost, df.mid_lost)
    return df


def main():
    hl = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    df = add(I.build(False), hl)
    print({k: float((df[k] > 0).mean()) for k in NEW + ["humiliation"]}, flush=True)
    res = {"half_life": hl}
    for target, feats0 in (("interstate_5y", I.FEATS), ("interstate_1y", I.FEATS_ON)):
        d = df if target == "interstate_5y" else df[df.ongoing == 0]
        for split, origins in (("dev", range(1975, 1996, 5)), ("test", range(2000, 2016, 5))):
            for name, feats in (("base", feats0), ("revenge", feats0 + NEW), ("humiliation", feats0 + ["humiliation"])):
                out, _ = I.backtest(d, target, origins, feats)
                res[f"{target}:{split}:{name}"] = {m: {k: round(v, 4) for k, v in out[m].items() if k in ("auc", "bss", "events")}
                                                   for m in ("logit", "gbm", "avg")}
                print(target, split, name, res[f"{target}:{split}:{name}"]["avg"], "logit",
                      res[f"{target}:{split}:{name}"]["logit"], flush=True)
    # association with controls (logit on all pairs at peace 1946-2015, the 1-year onset), for scale
    d = df[(df.ongoing == 0) & (df.year >= 1946) & (df.year <= 2015)].dropna(subset=I.FEATS_ON + ["interstate_1y"])
    import statsmodels.api as sm
    Xc = sm.add_constant(d[I.FEATS_ON + ["humiliation"]].astype(float))
    try:
        m = sm.Logit(d.interstate_1y.astype(float), Xc).fit(disp=0, cov_type="cluster",
                                                            cov_kwds={"groups": pd.factorize(d.a + d.b)[0]})
        res["logit_humiliation"] = [float(m.params["humiliation"]), float(m.bse["humiliation"])]
        print("logit humiliation", res["logit_humiliation"])
    except Exception as e:
        print("logit failed", e)
    os.makedirs(os.path.join(ROOT, "outputs", "new_data"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "outputs", "new_data", f"revenge_dyad_hl{int(hl)}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
