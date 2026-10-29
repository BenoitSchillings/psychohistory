"""Should interstate continuation depend on the pair, not only on run length? The production table (P(still active
next year) by run length 1 / 2 / 3+, recency-weighted with interstate.CONT_HALFLIFE = 25) against a logistic model
with the same run-length cells, the same recency weights, and a few pair features the simulator knows each year
(L2-penalised, standardised). Same protocol as scripts/interstate_cont_test.py: at each origin O fitted on
dyad-years before O, scored on every continuation outcome in O..O+9; dev origins 1970-1998, test 2000-2014 (step 2).
python -P scripts/interstate_cont_model_test.py -> outputs/interstate_cont_model_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.linear_model import LogisticRegression

from psychohistory import interstate as I
from interstate_cont_test import table, cells

ROOT = os.path.join(os.path.dirname(__file__), "..")
H = I.CONT_HALFLIFE
SETS = {"war": ["war_now"],
        "geo": ["war_now", "contig_land", "log_cap_dist", "rival"],
        "power": ["war_now", "one_major", "major_pair", "abs_power_ratio", "one_nuclear", "both_nuclear"],
        "regime": ["war_now", "dem_min", "dem_gap", "civil_war_any"],
        "all": ["war_now", "contig_land", "log_cap_dist", "rival", "one_major", "major_pair", "abs_power_ratio",
                "one_nuclear", "both_nuclear", "dem_min", "dem_gap", "civil_war_any", "conflict_past_10y"]}
CS = (0.03, 0.3)


def design(d, feats):
    r = d.conflict_run_years.to_numpy()
    runs = np.stack([(r == 2), (r >= 3)], -1).astype(float)
    return np.concatenate([runs, d[feats].to_numpy(float)], -1)


def score(y, p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)), np.sum((y - p) ** 2)


def main():
    df = I.build(False)
    d = df[df.ongoing == 1].copy()
    nxt = df.set_index(["a", "b", "year"]).ev_ucdp
    d["next"] = [nxt.get((a, b, y + 1), np.nan) for a, b, y in zip(d.a, d.b, d.year)]
    d = d[d["next"].notna()]
    res = {}
    for split, origins in (("dev", range(1970, 1999, 2)), ("test", range(2000, 2015, 2))):
        acc = {}
        for O in origins:
            tr, ev = d[d.year < O], d[(d.year >= O) & (d.year < O + 10) & (d.year + 1 <= 2024)]
            if not len(ev):
                continue
            y = ev["next"].astype(float).to_numpy()
            tab = table(tr, O, H, 0)
            p = np.zeros(len(ev))
            for r_, m in cells(ev).items():
                p[m.to_numpy()] = tab[r_]
            acc.setdefault("table", []).append((*score(y, p), len(y)))
            w = 0.5 ** ((O - tr.year.to_numpy()) / H)
            for name, feats in SETS.items():
                Xtr, Xev = design(tr, feats), design(ev, feats)
                mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-6
                z = lambda A: np.nan_to_num(np.clip((A - mu) / sd, -6, 6))
                for C in CS:
                    m = LogisticRegression(C=C, max_iter=4000).fit(z(Xtr), tr["next"].astype(int), sample_weight=w)
                    acc.setdefault(f"{name},C={C}", []).append((*score(y, m.predict_proba(z(Xev))[:, 1]), len(y)))
        res[split] = {k: {"logloss": round(sum(a for a, _, _ in v) / sum(n for _, _, n in v), 4),
                          "brier": round(sum(b for _, b, _ in v) / sum(n for _, _, n in v), 4),
                          "n": sum(n for _, _, n in v)} for k, v in acc.items()}
        print(split, json.dumps(res[split], indent=0), flush=True)
    json.dump(res, open(os.path.join(ROOT, "outputs", "interstate_cont_model_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
