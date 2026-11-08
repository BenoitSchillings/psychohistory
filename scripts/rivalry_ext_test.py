"""Rivalries after 2010 (reader feedback and state-of-the-art survey, 7 November 2026): the Thompson-Dreyer strategic
rivalries in the pair model stop with their data; rivalries active then are carried forward, but none can start
after it, and some salient pairs are not coded at all (US-Iran, Russia-Ukraine). A mechanical extension from the event
data: a pair counts as rivals in a year if the 5-year mean of its threat and force-posture events (ev_threat3, log)
up to that year is at or above a threshold tau. tau is chosen on 1995-2010 only, as the value whose classification
best reproduces the coded rivalries among relevant pairs (F1); no outcome is used. rival_x = coded rival or event rival.

Pre-registered (7 November 2026, before the first run): the 1-year onset model that drives the simulator (pairs at
peace, FEATS_ON with rival_x in place of rival), origins 2011, 2015, 2019 (each scored on its next five years), against
the same model with the coded rivals. Adopt for the February release if the Brier skill of the averaged model improves
and its AUC does not fall.
python -P scripts/rivalry_ext_test.py -> outputs/rivalry_ext_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import interstate as I


def main():
    df = I.build(False).sort_values(["a", "b", "year"])
    g = df.groupby(["a", "b"]).ev_threat3
    df["thr5"] = g.transform(lambda s: s.rolling(5, min_periods=3).mean())
    cal = df[(df.year >= 1999) & (df.year <= 2010) & df.thr5.notna()]
    best = None
    for tau in np.arange(1.0, 6.01, 0.1):
        pr = cal.thr5 >= tau
        tp = (pr & (cal.rival == 1)).sum(); fp = (pr & (cal.rival == 0)).sum(); fn = (~pr & (cal.rival == 1)).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if best is None or f1 > best[1]:
            best = (round(float(tau), 2), float(f1), int(tp), int(fp), int(fn))
    tau = best[0]
    print("tau", best, flush=True)
    df["rival_x"] = np.maximum(df.rival, (df.thr5 >= tau).astype(float).where(df.thr5.notna(), 0.0))
    new = df[(df.year == 2025) & (df.rival_x == 1) & (df.rival == 0)][["a", "b"]]
    print("new rivals in 2025:", len(new), [f"{a}-{b}" for a, b in zip(new.a, new.b)][:40], flush=True)
    d = df[df.ongoing == 0]
    feats0 = I.FEATS_ON
    feats1 = [("rival_x" if f == "rival" else f) for f in feats0]
    res = {"tau": best, "new_rivals_2025": [f"{a}-{b}" for a, b in zip(new.a, new.b)]}
    for name, feats in (("coded", feats0), ("extended", feats1)):
        out, _ = I.backtest(d, "interstate_1y", (2011, 2015, 2019), feats)
        res[name] = {m: {k: round(v, 4) for k, v in out[m].items()} for m in ("logit", "gbm", "avg")}
        print(name, res[name]["avg"], flush=True)
    a0, a1 = res["coded"]["avg"], res["extended"]["avg"]
    res["adopt"] = bool(a1["bss"] > a0["bss"] and a1["auc"] >= a0["auc"])
    print("adopt:", res["adopt"])
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "rivalry_ext_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
