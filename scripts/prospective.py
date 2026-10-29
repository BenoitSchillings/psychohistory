"""Prospective scoring, meant to run once a month (cron): fetch new data, issue new forecasts, score matured ones.

python -P scripts/prospective.py            # fetch + issue + snapshot + score
python -P scripts/prospective.py score      # score only (no network)

1. fetch: the newest cumulative UCDP Candidate file of the current year (GEDEvent_vYY_01_YY_MM.csv, released around the
   20th of the following month) into data/raw/outcomes/ucdp/candidate_live/. A newer FINAL GED release (GEDYY.1) is
   only reported: switching to it is a manual step (psychohistory/monthly.FINAL_THROUGH, cm_fatal._final).
2. issue: a 12-month forecast of P(25+ state-based battle deaths) per country from the newest month
   (scripts/monthly_ledger.py issue: one file per data month, never overwritten).
3. crowd snapshot: open Polymarket / Manifold conflict questions with the model's and the blend's probability for each,
   appended to outputs/ledger/crowd_snapshots.jsonl.
4. score: every matured monthly forecast against the newest data, with two baselines issued from the same data month
   (persistence: share of 25+ months in the past 12; climatology: the country's share since 1989), and every resolved
   market question (model vs crowd vs blend, first snapshot of each question). The LLM-panel ledger
   (outputs/llm_ledger.jsonl) covers 2025-2029 and resolves in 2030.
5. annual: log the published simulator forecast once per model version and score each completed calendar year
   (scripts/annual_ledger.py).
Writes outputs/prospective_scorecard.json and appends one line per run to outputs/ledger/score_history.jsonl.
"""
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from psychohistory import monthly as MO

LEDGER = os.path.join(ROOT, "outputs", "ledger")
LIVE = MO.O / "ucdp" / "candidate_live"
UCDP = "https://ucdp.uu.se/downloads/candidateged/"
UA = {"User-Agent": "psychohistory-research/1.0"}


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _exists(url):
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except urllib.error.HTTPError:
        return False


def fetch():
    """download the newest cumulative candidate file not yet on disk; returns a log dict"""
    log = {}
    today = dt.date.today()
    have = MO.candidate_files()
    for y in sorted({today.year, (today - dt.timedelta(days=60)).year}):
        if y <= MO.FINAL_THROUGH:
            continue
        for mo in range(12, 0, -1):
            name = f"GEDEvent_v{y % 100}_01_{y % 100}_{mo:02d}.csv"
            if y in have and have[y][1] >= mo:
                break
            if _exists(UCDP + name):
                LIVE.mkdir(parents=True, exist_ok=True)
                data = _get(UCDP + name, timeout=300)
                if not data.startswith(b"id,") and b"type_of_violence" not in data[:2000]:
                    log[name] = "unexpected content, not saved"
                    break
                (LIVE / name).write_bytes(data)
                log[name] = f"downloaded {len(data) // 1024} kB"
                break
    nxt = MO.FINAL_THROUGH + 1
    final_url = f"https://ucdp.uu.se/downloads/ged/ged{(nxt + 1) % 100}1-csv.zip"
    log["new_final_ged"] = final_url if _exists(final_url) else None
    return log


def issue():
    r = subprocess.run([sys.executable, "-P", os.path.join(ROOT, "scripts", "monthly_ledger.py"), "issue"],
                       capture_output=True, text=True)
    return (r.stdout.strip().splitlines() or [r.stderr.strip()[-300:]])[0]


def crowd_snapshot():
    from psychohistory import crowd
    fc = json.load(open(os.path.join(ROOT, "outputs", "forecast.json")))
    qs = crowd.fetch()
    m, _ = crowd.match(qs, fc)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    os.makedirs(LEDGER, exist_ok=True)
    with open(os.path.join(LEDGER, "crowd_snapshots.jsonl"), "a") as f:
        for r in m:
            f.write(json.dumps({"snap": now, **{k: r.get(k) for k in ("source", "id", "question", "close", "kind",
                                                                       "countries", "p", "model", "blend", "url")}}) + "\n")
    return len(m)


def _resolution(source, mid_):
    """1.0 / 0.0 when resolved YES / NO, None otherwise"""
    try:
        if source == "polymarket":
            d = json.loads(_get(f"https://gamma-api.polymarket.com/markets/{mid_}"))
            if not d.get("closed"):
                return None
            pr = d.get("outcomePrices")
            pr = json.loads(pr) if isinstance(pr, str) else pr
            if pr and max(float(x) for x in pr) > 0.99:
                return float(float(pr[0]) > 0.99)
            return None
        if source == "manifold":
            d = json.loads(_get(f"https://api.manifold.markets/v0/market/{mid_}"))
            if d.get("isResolved") and d.get("resolution") in ("YES", "NO"):
                return float(d["resolution"] == "YES")
    except Exception:
        return None
    return None


def score_crowd(online=True):
    f = os.path.join(LEDGER, "crowd_snapshots.jsonl")
    if not os.path.exists(f):
        return {"questions": 0}
    S = pd.DataFrame([json.loads(l) for l in open(f)])
    first = S.sort_values("snap").groupby(["source", "id"], as_index=False).first()
    cache_f = os.path.join(LEDGER, "crowd_resolutions.json")
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    for s, i, close in zip(first.source, first.id, first.close):
        k = f"{s}:{i}"
        if k in cache or not online:
            continue
        if close and pd.Timestamp(close) > pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=400):
            continue
        r = _resolution(s, i)
        if r is not None:
            cache[k] = r
    json.dump(cache, open(cache_f, "w"), indent=1)
    first["y"] = [cache.get(f"{s}:{i}") for s, i in zip(first.source, first.id)]
    res = first.dropna(subset=["y"])
    out = {"questions": int(len(first)), "resolved": int(len(res))}
    if len(res):
        # the pool is recomputed from the logged crowd and model numbers (snapshots before 2026-10-28 stored an older,
        # fitted blend), so every question is scored with the same fixed rule
        from psychohistory.crowd import POOL_W
        lg = lambda v: np.log(np.clip(v, 0.005, 0.995) / (1 - np.clip(v, 0.005, 0.995)))
        res = res.assign(pool=1 / (1 + np.exp(-((1 - POOL_W) * lg(res.p.astype(float)) + POOL_W * lg(res.model.astype(float))))))
        for k in ("model", "p", "pool"):
            out[f"brier_{'crowd' if k == 'p' else k}"] = round(float(np.mean((res[k].astype(float) - res.y) ** 2)), 4)
        out["resolved_yes"] = int(res.y.sum())
    return out


def score_monthly():
    gw2iso, codes = MO.gw_map()
    ly, lm = MO.last_observed()
    D = MO.deaths(codes, gw2iso, ly, lm)
    hit = (D >= 25).astype(float)
    ci = {c: i for i, c in enumerate(codes)}
    out = {"data_to": f"{ly}-{lm:02d}", "ledgers": {}}
    allrows = []
    for f in sorted(glob.glob(os.path.join(LEDGER, "monthly_*.csv"))):
        L = pd.read_csv(f)
        dy, dm = map(int, str(L.data_to.iloc[0]).split("-"))
        t0 = MO.midx(dy, dm)
        L["m"] = (L.target_year - MO.M0) * 12 + L.target_month - 1
        L = L[(L.m < D.shape[1]) & L.iso3.isin(ci)]
        if L.empty:
            out["ledgers"][os.path.basename(f)] = "no matured months yet"
            continue
        idx = L.iso3.map(ci).to_numpy()
        L["y"] = hit[idx, L.m.to_numpy()]
        L["persist"] = [hit[i, t0 - 11:t0 + 1].mean() for i in idx]
        L["clim"] = [hit[i, :t0 + 1].mean() for i in idx]
        allrows.append(L)
        out["ledgers"][os.path.basename(f)] = _metrics(L)
    if allrows:
        out["all"] = _metrics(pd.concat(allrows))
    return out


def _metrics(L):
    from sklearn.metrics import roc_auc_score
    b = lambda p: float(np.mean((p - L.y) ** 2))
    r = {"months": int(L.m.nunique()), "n": int(len(L)), "events": int(L.y.sum()),
         "brier_model": round(b(L.p_ge25), 5), "brier_persistence": round(b(L.persist), 5), "brier_climatology": round(b(L.clim), 5)}
    r["skill_vs_persistence"] = round(1 - r["brier_model"] / r["brier_persistence"], 4) if r["brier_persistence"] else None
    if 0 < L.y.sum() < len(L):
        r["auc_model"] = round(float(roc_auc_score(L.y, L.p_ge25)), 4)
        r["auc_persistence"] = round(float(roc_auc_score(L.y, L.persist)), 4)
    new = L[L.persist == 0]                       # countries with no 25-death month in the year before issue
    if len(new) and 0 < new.y.sum() < len(new):
        r["new_cases"] = {"n": int(len(new)), "events": int(new.y.sum()), "auc_model": round(float(roc_auc_score(new.y, new.p_ge25)), 4),
                          "brier_model": round(float(np.mean((new.p_ge25 - new.y) ** 2)), 5)}
    return r


def main():
    online = not (len(sys.argv) > 1 and sys.argv[1] == "score")
    run = {"run": dt.datetime.now().isoformat(timespec="seconds")}
    if online:
        try:
            run["fetch"] = fetch()
        except Exception as e:
            run["fetch"] = f"failed: {e}"
        run["issue"] = issue()
        try:
            run["crowd_snapshot"] = crowd_snapshot()
        except Exception as e:
            run["crowd_snapshot"] = f"failed: {e}"
    run["monthly"] = score_monthly()
    run["crowd"] = score_crowd(online)
    try:
        import annual_ledger as AL
        run["annual"] = {"issue": AL.issue() if online else "not issued (score only)", "score": AL.score()}
    except Exception as e:
        run["annual"] = f"failed: {e}"
    run["llm_panel"] = "window 2025-2029: resolves in 2030"
    json.dump(run, open(os.path.join(ROOT, "outputs", "prospective_scorecard.json"), "w"), indent=1)
    with open(os.path.join(LEDGER, "score_history.jsonl"), "a") as f:
        f.write(json.dumps(run) + "\n")
    print(json.dumps(run, indent=1))


if __name__ == "__main__":
    main()
