"""Prospective monthly conflict forecasts: issue, log with today's date, and score whatever has matured.

python scripts/monthly_ledger.py issue   # train on all events to the last observed month, forecast the next 12
python scripts/monthly_ledger.py score   # score every logged forecast whose month is now observed
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
    ann = MO.annual_probability(pred)
    top = sorted(zip(codes, ann, hit[:, t - 11:t + 1].max(1)), key=lambda r: -r[1])
    print("issued", path)
    print("highest chance of a 25-death month, Oct 2026 - Sep 2027, among countries without one in the past year:",
          [(c, round(float(a), 3)) for c, a, h in top if h == 0][:10])


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
    {"issue": issue, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "issue"]()
