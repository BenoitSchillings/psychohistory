"""News topics for onsets in quiet countries (state-of-the-art survey, 7 November 2026): conflictforecast.org's
news-topic stocks (Mueller and Rauh 2022, data paper Mueller, Rauh and Seimon 2024; free for non-commercial use with
citation; file data/raw/conflictforecast/conflictforecast_ons_armedconf_12.csv, downloaded 8 November 2026) as features
of the direct 12-month model (monthly.fit_annual; monthly.NEWS_TOPICS). Only the raw topic stocks and token counts are
used, not that site's own forecasts (which may not be as issued). The topic model itself was estimated on the whole
corpus, a mild look-ahead the field accepts.
Pre-registered (8 November 2026, before the first run): issued each December 2017-2023, scored on the next year, as
scripts/monthly_annual_test.py; adopt for the February release if both Brier and log loss improve for countries
without a 25-death month in the past year (pooled), the hard case.
python -P scripts/monthly_topics_test.py -> outputs/monthly_topics_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import monthly as MO
from psychohistory.data import get_panel


def sc(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"brier": round(float(np.mean((p - y) ** 2)), 5), "logloss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5),
            "mean_p": round(float(p.mean()), 4), "rate": round(float(y.mean()), 4)}


def main():
    gw2iso, codes = MO.gw_map()
    D = MO.deaths(codes, gw2iso, 2025, 12)
    P = {"base": [], "topics": [], "y": [], "new": []}
    for flag, key in ((False, "base"), (True, "topics")):
        MO.NEWS_TOPICS = flag
        F, hit = MO.features(D, codes, get_panel())
        for yr in range(2017, 2024):
            t = MO.midx(yr, 12)
            P[key].append(MO.predict_annual(MO.fit_annual(F, hit, t), F, t))
            if key == "base":
                P["y"].append(hit[:, t + 1:t + 13].max(1))
                P["new"].append(hit[:, t - 11:t + 1].max(1) == 0)
            print(key, yr, flush=True)
    y, new = np.concatenate(P["y"]), np.concatenate(P["new"])
    out = {}
    for k in ("base", "topics"):
        p = np.concatenate(P[k])
        out[k] = {"all": sc(p, y), "new_only": sc(p[new], y[new])}
    out["by_year_new_brier"] = {str(2017 + i): [round(float(np.mean((P[k][i][P["new"][i]] - P["y"][i][P["new"][i]]) ** 2)), 5)
                                               for k in ("base", "topics")] for i in range(7)}
    b, t_ = out["base"]["new_only"], out["topics"]["new_only"]
    out["adopt"] = bool(t_["brier"] < b["brier"] and t_["logloss"] < b["logloss"])
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "monthly_topics_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
