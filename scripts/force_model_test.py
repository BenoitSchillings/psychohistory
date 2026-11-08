"""Forecast what the crowd questions ask: any use of force between two states (state of the art survey, 7 November 2026:
most of the crowd's lead comes from a target mismatch; 13% of the use-of-force questions resolved yes while the
benchmark model's 25-death label averages 2%).

Label: a pair-year with 20 or more machine-coded fighting events between the two (ICEWS 1995-2017, POLECAT 2018-2024,
GDELT 2025, rescaled to ICEWS levels as in interstate._add_events). On 1995-2010 this threshold matches the rate of
militarised uses of force in the hand-coded MID data (1.4% against 1.35%), with precision 0.39 and recall 0.42 against them.
Model: the pair model (interstate.DyadModel) on the 1-year features with events (FEATS_1Y), predicting the label next
year; for each question, refitted on years whose outcome was known before the question opened, its one-year
probability converted to the question's window at a constant hazard.
Pre-registered comparison (7 November 2026, before the first run): Brier on the matched use-of-force questions one week
after opening, against the current benchmark model (UCDP labels, 0.130) and the crowd (0.100), and the logged 10%
logarithmic pool. Adopted for the benchmark only if its Brier is below 0.130.
python -P scripts/force_model_test.py -> outputs/force_model_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
import pandas as pd

from psychohistory import interstate as I
from psychohistory.data import RAW
from crowd_benchmark import FORCE, SKIP_SUB, stated_deadline

K = 20


def fight_counts():
    f = RAW / "events"
    ic = pd.read_csv(f / "icews_dy.csv")
    pc = pd.read_csv(f / "polecat_dy.csv")
    ic, pc = ic[ic.year <= 2017], pc[pc.year >= 2018].copy()
    pc["n_fight"] = pc["n_fight"] / I.EV_RATIO["n_fight"]
    pc.loc[pc.year == 2024, "n_fight"] *= 365 / 181
    parts = [ic, pc]
    gd, gs = f / "gdelt_dy.csv", f / "gdelt_splice.json"
    if gd.exists() and gs.exists():
        rr = json.load(open(gs))["polecat_over_gdelt"]
        g = pd.read_csv(gd)
        g = g[g.year == 2025].copy()
        g["n_fight"] = g["n_fight"] * rr["n_fight"] / I.EV_RATIO["n_fight"]
        parts.append(g)
    ev = pd.concat(parts)[["iso3_a", "iso3_b", "year", "n_fight"]].rename(columns={"iso3_a": "a", "iso3_b": "b"})
    return ev


def main():
    df = I.build(False)
    ev = fight_counts()
    df = df.merge(ev, on=["a", "b", "year"], how="left")
    df["n_fight"] = np.where((df.year >= 1995) & (df.year <= 2025), df.n_fight.fillna(0), np.nan)
    df["force_ev"] = np.where(df.n_fight.notna(), (df.n_fight >= K).astype(float), np.nan)
    df = df.sort_values(["a", "b", "year"])
    df["force_ev_1y"] = df.groupby(["a", "b"]).force_ev.shift(-1)
    q = pd.read_csv(os.path.join(ROOT, "data", "raw", "benchmarks", "crowd_questions.csv"), parse_dates=["open_date", "close_date"])
    q = q[q.resolution.notna() & (q.polarity == 1)]
    sel = q[(q.category == "interstate") & ~q.subtype.isin(SKIP_SUB) & q.iso3_a.notna() & q.iso3_b.notna()
            & q.title.str.contains(FORCE)].copy()
    feats = I.FEATS_1Y
    fits = {}

    def model_for(y0):
        if y0 not in fits:
            tr = df.force_ev_1y.notna() & (df.year + 1 <= y0) & (df.year >= 1995)
            fits[y0] = I.DyadModel().fit(df.loc[tr, feats].to_numpy(float), df.loc[tr, "force_ev_1y"].to_numpy())
        return fits[y0]
    rows = []
    for _, r in sel.iterrows():
        a, b = sorted((r.iso3_a, r.iso3_b))
        y0 = r.open_date.year - 1
        x = df[(df.a == a) & (df.b == b) & (df.year == y0)]
        if x.empty:
            continue
        p1 = float(model_for(y0).predict(x[feats].to_numpy(float))[0])
        start = r.open_date + pd.Timedelta(days=7)
        end = stated_deadline(r.title)
        if end is None:
            if r.resolution == 1:
                continue
            end = r.close_date
        end = max(end, r.close_date)
        if not np.isfinite(r.p_open_7d):
            continue
        days = max((end - start).days, 1)
        rows.append({"qid": r.qid, "pair": f"{a}-{b}", "y": float(r.resolution), "crowd": float(r.p_open_7d),
                     "model": float(1 - (1 - np.clip(p1, 0, 0.999)) ** (days / 365)), "p1": p1, "year": int(r.open_date.year)})
    R = pd.DataFrame(rows)
    y, c, m = R.y.to_numpy(), R.crowd.to_numpy(), R.model.to_numpy()
    lg = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
    pool = 1 / (1 + np.exp(-(0.9 * lg(c) + 0.1 * lg(m))))
    br = lambda p: round(float(np.mean((p - y) ** 2)), 4)
    rng = np.random.default_rng(0)
    up = R.pair.unique()
    idx = {u: np.where(R.pair.to_numpy() == u)[0] for u in up}
    d = []
    for _ in range(2000):
        s = np.concatenate([idx[u] for u in rng.choice(up, len(up))])
        d.append(np.mean((m[s] - y[s]) ** 2) - np.mean((c[s] - y[s]) ** 2))
    old = json.load(open(os.path.join(ROOT, "outputs", "crowd_benchmark.json")))["open"]
    out = {"n": int(len(R)), "events": int(y.sum()), "mean_model": round(float(m.mean()), 4), "mean_crowd": round(float(c.mean()), 4),
           "brier_model_force": br(m), "brier_crowd": br(c), "brier_pool_0.1": br(pool),
           "brier_model_ucdp_benchmark": round(old["brier_model"], 4),
           "model_minus_crowd_90ci": [round(float(x), 4) for x in np.percentile(d, [5, 95])],
           "adopt_for_benchmark": bool(br(m) < 0.130), "questions": R.round(4).to_dict("records")}
    print(json.dumps({k: v for k, v in out.items() if k != "questions"}, indent=1))
    json.dump(out, open(os.path.join(ROOT, "outputs", "force_model_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
