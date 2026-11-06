"""Head-to-head with crowd forecasters (Metaculus, Good Judgment Open, INFER) on the questions the model can answer.

Questions: data/raw/benchmarks/crowd_questions.csv (resolved, built from public research datasets).
Only questions whose event matches a model quantity are scored:
- use of force between two named states (strikes, attacks, lethal confrontations, invasions, armed conflict):
  the dyad use-of-force model (GML MID hostility 4-5, interstate.force_5y), with features from the year before
  the question opened (the MID labels end in 2010, so every vintage trains on the same 1816-2005 outcomes).
Not scored: coups (only five single-country questions, none resolved yes), leader exits, ceasefires, territory,
elections, currency and recession questions -- the model forecasts none of these directly.
Five-year probabilities are turned into the question's window (from the crowd snapshot to the close date) assuming
a constant hazard. The crowd is taken one week after the question opened (p_open_7d) and at the midpoint.

--labels ucdp: the model trained on UCDP/PRIO interstate conflict (labels through 2024, refitted for each question's
year on windows complete before it) -> outputs/crowd_benchmark_ucdp.json. Pre-registered 2026-11-05, before running:
the benchmark keeps whichever label set has the lower leave-one-pair-out Brier on the resolved questions, and the
logged pool weight stays 0.1 for every question resolving from now on, whatever the comparison shows.
Result (2026-11-05, one week after opening, 63 questions): Brier UCDP 0.130, MID 0.132, crowd 0.100; leave-one-pair-out
blend 0.104 / 0.111. UCDP labels are the default since. Either label is rarer than the questions' events (13% of
questions resolved yes; the model's mean is 2% with UCDP, 8% with MID): the stale MID labels were not why the crowd
wins, the mismatch between "use of force" and a 25-death conflict is a larger part.

Scores: Brier of model and crowd on the same questions, the mean log-score, and a leave-one-out logistic blend of
the two (does the model add anything to the crowd?). Standard errors cluster by country pair.
"""
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from psychohistory import interstate as I

B = os.path.join(ROOT, "data", "raw", "benchmarks")
FORCE = re.compile(r"strike|attack|lethal|confrontation|invade|invasion|armed conflict|military|clash|\bwar\b|deaths|fire|shoot|bomb", re.I)
SKIP_SUB = {"russia_ukraine_territory", "ceasefire", "nuclear_use"}


MON = "January|February|March|April|May|June|July|August|September|October|November|December"
_DMY = re.compile(rf"(\d{{1,2}}) ({MON}),? (\d{{4}})")
_MDY = re.compile(rf"({MON}) (\d{{1,2}}),? (\d{{4}})")
_BEFORE_Y = re.compile(r"before (\d{4})\b(?! ?[-/])")
_IN_Y = re.compile(rf"(?:in|during|by the end of) (?:({MON}) )?(\d{{4}})\b", re.I)
_GROUP_Y = re.compile(r"(?:dates|years)\?? \((\d{4})\)")         # Metaculus groups: "before these dates? (2025)"


def stated_deadline(title):
    """the deadline written in the question (an early YES closes the market on the event date, which must not shorten
    the model's window): the last explicit date, 'before 2024' (= 1 January), 'in 2020' / 'in December 2017'
    (= end of that year / month); None if the title states none"""
    t = str(title)
    ds = [pd.Timestamp(f"{d} {m} {y}") for d, m, y in _DMY.findall(t)] + [pd.Timestamp(f"{d} {m} {y}") for m, d, y in _MDY.findall(t)]
    if ds:
        return max(ds)
    m = _BEFORE_Y.search(t) or _GROUP_Y.search(t)
    if m:
        return pd.Timestamp(f"{m.group(1)}-01-01")
    m = _IN_Y.search(t)
    if m:
        mon, y = m.groups()
        return (pd.Timestamp(f"1 {mon} {y}") + pd.offsets.MonthEnd(0)) if mon else pd.Timestamp(f"{y}-12-31")
    return None


def window_prob(p5, start, end):
    days = max((pd.Timestamp(end) - pd.Timestamp(start)).days, 1)
    return 1 - (1 - np.clip(p5, 0, 0.999)) ** (days / 1826)


def main():
    q = pd.read_csv(os.path.join(B, "crowd_questions.csv"), parse_dates=["open_date", "close_date"])
    q = q[q.resolution.notna() & (q.polarity == 1)]
    sel = q[(q.category == "interstate") & ~q.subtype.isin(SKIP_SUB) & q.iso3_a.notna() & q.iso3_b.notna()
            & q.title.str.contains(FORCE)].copy()
    df = I.build(False)
    labels = sys.argv[sys.argv.index("--labels") + 1] if "--labels" in sys.argv else "ucdp"   # (mid until 2026-11-05)
    if labels == "mid":
        tr = df.force_5y.notna()
        m_mid = I.DyadModel().fit(df.loc[tr, I.FEATS].to_numpy(float), df.loc[tr, "force_5y"].to_numpy())   # labels end 2010
        model_for = lambda y0: m_mid
    else:
        # UCDP/PRIO interstate conflict (25+ battle deaths in a year) within five years, labels through 2024; each
        # question's model is trained only on five-year windows complete by the year before it opened (debate
        # 2026-11-05: the MID labels end in 2010, five years before the first question). A higher bar than "use of
        # force" (strikes need not kill 25), so the two are reported side by side.
        fits = {}

        def model_for(y0):
            if y0 not in fits:
                tr = df.interstate_5y.notna() & (df.year + 5 <= y0)
                fits[y0] = I.DyadModel().fit(df.loc[tr, I.FEATS].to_numpy(float), df.loc[tr, "interstate_5y"].to_numpy())
            return fits[y0]
    rows, skipped, conts = [], 0, {}
    for _, r in sel.iterrows():
        a, b = sorted((r.iso3_a, r.iso3_b))
        y0 = r.open_date.year - 1                         # data for the year before the question opened
        x = df[(df.a == a) & (df.b == b) & (df.year == y0)]
        if x.empty:
            continue
        p5 = float(model_for(y0).predict(x[I.FEATS].to_numpy(float))[0])
        # a war already in progress: next year's continuation is a floor (as in interstate.run)
        run_y = int(np.clip(np.nan_to_num(x.conflict_run_years.iloc[0]), 0, 3))
        if run_y > 0:
            cont = conts.setdefault(y0, I.continuation_rates(df, y0))          # transitions known by then
            p5 = max(p5, cont.get(run_y, 0.5))
        start = r.open_date + pd.Timedelta(days=7)
        end = stated_deadline(r.title)
        if end is None:
            if r.resolution == 1:          # closed early on the event: the real deadline is unknown, leave it out
                skipped += 1
                continue
            end = r.close_date
        end = max(end, r.close_date)
        for snap, col in (("open", "p_open_7d"), ("mid", "p_midpoint")):
            if np.isfinite(r[col]):
                s_ = start if snap == "open" else r.open_date + (r.close_date - r.open_date) / 2   # the crowd's midpoint
                rows.append({"snap": snap, "source": r.source, "qid": r.qid, "title": r.title, "pair": f"{a}-{b}",
                             "y": float(r.resolution), "crowd": float(r[col]), "model": float(window_prob(p5, s_, end)),
                             "model_5y": p5, "days": (end - s_).days, "year": int(r.open_date.year)})
    R = pd.DataFrame(rows)
    out = {"labels": labels, "n_questions_total": int(len(q)), "n_matched": int(sel.shape[0]), "n_skipped_no_deadline": skipped, "note": __doc__.strip().split("\n\n")[0]}
    lo = lambda p: np.log(np.clip(p, 0.005, 0.995) / (1 - np.clip(p, 0.005, 0.995)))
    for snap, g in R.groupby("snap"):
        y, c, mo = g.y.to_numpy(), g.crowd.to_numpy(), g.model.to_numpy()
        # leave-one-pair-out logistic blend of crowd and model log-odds
        bl = np.zeros(len(g))
        for pr in g.pair.unique():
            te = (g.pair == pr).to_numpy()
            if (~te).sum() < 10 or len(np.unique(y[~te])) < 2:
                bl[te] = c[te]
                continue
            lr = LogisticRegression(C=1.0).fit(np.c_[lo(c[~te]), lo(mo[~te])], y[~te])
            bl[te] = lr.predict_proba(np.c_[lo(c[te]), lo(mo[te])])[:, 1]
        brier = lambda p: float(np.mean((p - y) ** 2))
        logs = lambda p: float(np.mean(y * np.log(np.clip(p, .005, .995)) + (1 - y) * np.log(np.clip(1 - p, .005, .995))))
        # pair-clustered bootstrap of the Brier difference (model - crowd) and (blend - crowd)
        rng = np.random.default_rng(0)
        pairs = g.pair.to_numpy()
        up = np.unique(pairs)
        idx = {u: np.where(pairs == u)[0] for u in up}
        dm, db = [], []
        for _ in range(2000):
            s = np.concatenate([idx[u] for u in rng.choice(up, len(up))])
            dm.append(np.mean((mo[s] - y[s]) ** 2) - np.mean((c[s] - y[s]) ** 2))
            db.append(np.mean((bl[s] - y[s]) ** 2) - np.mean((c[s] - y[s]) ** 2))
        out[snap] = {"n": int(len(g)), "pairs": int(len(up)), "events": int(y.sum()), "base_rate": float(y.mean()),
                     "brier_crowd": brier(c), "brier_model": brier(mo), "brier_blend_loo": brier(bl),
                     "brier_base_rate": brier(np.full(len(y), y.mean())),
                     "log_crowd": logs(c), "log_model": logs(mo),
                     "brier_diff_model_minus_crowd_90ci": np.percentile(dm, [5, 95]).round(4).tolist(),
                     "brier_diff_blend_minus_crowd_90ci": np.percentile(db, [5, 95]).round(4).tolist(),
                     "mean_crowd": float(c.mean()), "mean_model": float(mo.mean())}
    out["questions"] = R[R.snap == "open"].sort_values("year").round(4).to_dict("records")
    json.dump(out, open(os.path.join(ROOT, "outputs", "crowd_benchmark.json" if labels == "ucdp" else f"crowd_benchmark_{labels}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "questions"}, indent=1))


if __name__ == "__main__":
    main()
