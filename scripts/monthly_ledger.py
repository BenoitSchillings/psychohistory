"""Prospective monthly conflict forecasts: issue, log with today's date, and score whatever has matured.

python scripts/monthly_ledger.py issue   # train on all events to the last observed month, forecast the next 12
python scripts/monthly_ledger.py score   # score every logged forecast whose month is now observed
python scripts/monthly_ledger.py summary YEAR MONTH   # outputs/ledger_summary.json for the ledger issued with data to then
python scripts/monthly_ledger.py civilian YEAR MONTH  # non-state and one-sided violence (UCDP types 2, 3), 12 months
Ledger: outputs/ledger/monthly_<issue month>.csv (one row per country x target month), never overwritten.
"""
import datetime as dt
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd

from psychohistory import monthly as MO
from psychohistory.data import get_panel

LEDGER = os.path.join(ROOT, "outputs", "ledger")
LAST = MO.last_observed()      # last month in the newest UCDP candidate file


def issue():
    os.makedirs(LEDGER, exist_ok=True)
    gw2iso, codes = MO.gw_map()
    D = MO.deaths(codes, gw2iso, *LAST)
    F, hit = MO.features(D, codes, get_panel())
    t = MO.midx(*LAST)
    model = MO.fit(F, hit, t)
    pred = MO.predict(model, F, t)
    rows = []
    for h, pr in pred.items():
        y, m = divmod(t + h, 12)
        for c, v in zip(codes, pr):
            rows.append({"issued": dt.date.today().isoformat(), "data_to": f"{LAST[0]}-{LAST[1]:02d}", "iso3": c,
                         "target_year": MO.M0 + y, "target_month": m + 1, "horizon": h, "p_ge25": round(float(v), 5)})
    path = os.path.join(LEDGER, f"monthly_{LAST[0]}-{LAST[1]:02d}.csv")
    if os.path.exists(path):
        print("already issued:", path)
        return
    pd.DataFrame(rows).to_csv(path, index=False)
    ann = MO.predict_annual(MO.fit_annual(F, hit, t), F, t)       # (the product of monthly forecasts overstated it)
    top = sorted(zip(codes, ann, hit[:, t - 11:t + 1].max(1)), key=lambda r: -r[1])
    print("issued", path)
    print("highest chance of a 25-death month, Oct 2026 - Sep 2027, among countries without one in the past year:",
          [(c, round(float(a), 3)) for c, a, h in top if h == 0][:10])


def summary(year, month):
    """outputs/ledger_summary.json for the ledger issued with data to year-month: each country's probability of at
    least one 25-death month in the next 12, from the direct 12-month model (monthly.fit_annual; the product of the
    monthly forecasts overstated it: scripts/monthly_annual_test.py), with the old product kept for comparison"""
    gw2iso, codes = MO.gw_map()
    D = MO.deaths(codes, gw2iso, year, month)
    F, hit = MO.features(D, codes, get_panel())
    t = MO.midx(year, month)
    direct = MO.predict_annual(MO.fit_annual(F, hit, t), F, t)
    prod = MO.annual_probability(MO.predict(MO.fit(F, hit, t), F, t))
    active = hit[:, t - 11:t + 1].max(1) > 0
    f = os.path.join(ROOT, "outputs", "ledger_summary.json")
    old = json.load(open(f)) if os.path.exists(f) else {}
    rank = lambda v, m: dict(sorted(((c, round(float(a), 4)) for c, a, k in zip(codes, v, m) if k), key=lambda r: -r[1]))
    out = {"issued": old.get("issued", dt.date.today().isoformat()), "data_to": f"{year}-{month:02d}",
           "months": old.get("months"),
           "n_rows": old.get("n_rows"), "n_active": int(active.sum()),
           "annual_method": "direct 12-month model (monthly.fit_annual), since 2026-11-06",
           "annual_new": rank(direct, ~active), "annual_active": rank(direct, active),
           "annual_new_product": rank(prod, ~active)}
    json.dump(out, open(f, "w"), indent=1)
    print({k: list(v.items())[:6] if isinstance(v, dict) else v for k, v in out.items()})


def civilian(year, month):
    """frozen 12-month forecasts of non-state and one-sided violence (UCDP types 2 and 3: militias, communal fighting,
    massacres of civilians): P(a month with 25 or more such deaths in the next 12), direct model
    (scripts/monthly_civilian_test.py: Brier skill +27% against persistence, +15% where none in the past year).
    outputs/ledger/civilian_<year>-<month>.csv, never overwritten; summary in outputs/ledger_summary.json"""
    gw2iso, codes = MO.gw_map()
    D = MO.deaths(codes, gw2iso, year, month, types=(2, 3))
    F, hit = MO.features(D, codes, get_panel())
    t = MO.midx(year, month)
    p = MO.predict_annual(MO.fit_annual(F, hit, t), F, t)
    active = hit[:, t - 11:t + 1].max(1) > 0
    path = os.path.join(LEDGER, f"civilian_{year}-{month:02d}.csv")
    if not os.path.exists(path):
        pd.DataFrame({"issued": dt.date.today().isoformat(), "data_to": f"{year}-{month:02d}", "iso3": codes,
                      "past_12m": active.astype(int), "p_12m": np.round(p, 5)}).to_csv(path, index=False)
    f = os.path.join(ROOT, "outputs", "ledger_summary.json")
    S = json.load(open(f)) if os.path.exists(f) else {}
    rank = lambda m: dict(sorted(((c, round(float(a), 4)) for c, a, k in zip(codes, p, m) if k), key=lambda r: -r[1]))
    S["civilian"] = {"issued": dt.date.today().isoformat(), "data_to": f"{year}-{month:02d}", "n_active": int(active.sum()),
                     "new": rank(~active), "active": rank(active)}
    json.dump(S, open(f, "w"), indent=1)
    print(path, list(S["civilian"]["new"].items())[:8])


def score():
    """outcomes: the newest candidate/final event files available under data/raw/outcomes/ucdp."""
    gw2iso, codes = MO.gw_map()
    D = MO.deaths(codes, gw2iso, *LAST)
    res = {}
    for f in sorted(glob.glob(os.path.join(LEDGER, "monthly_*.csv"))):
        L = pd.read_csv(f)
        L["m"] = (L.target_year - MO.M0) * 12 + L.target_month - 1
        L = L[L.m < D.shape[1]]
        if L.empty:
            res[os.path.basename(f)] = "no matured months yet"
            continue
        ci = {c: i for i, c in enumerate(codes)}
        L["y"] = [float(D[ci[c], m] >= 25) for c, m in zip(L.iso3, L.m)]
        res[os.path.basename(f)] = {"months": int(L.m.nunique()), "n": len(L), "events": int(L.y.sum()),
                                    "brier": float(np.mean((L.p_ge25 - L.y) ** 2))}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary(int(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == "civilian":
        civilian(int(sys.argv[2]), int(sys.argv[3]))
    else:
        {"issue": issue, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "issue"]()
