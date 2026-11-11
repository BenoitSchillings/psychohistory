"""Prospective ledger for the simulator: freeze each published forecast version and score it year by year.

python -P scripts/annual_ledger.py issue    # log the current outputs/forecast.json (once per model version)
python -P scripts/annual_ledger.py score    # score every logged version against each completed calendar year

issue: a model version is the SHA-256 of the code (psychohistory/*.py, ensemble_config.json) and of the forecast
itself; each new version is saved once as outputs/ledger/annual_<origin>_<hash12>.json, never overwritten, with
 - per country: P(in conflict) for each year 2026-2030 (the simulator's conflict path), P(a coup by 2030),
   P(a new conflict by 2030), and the 5-year risk models' civil-conflict and coup probabilities;
 - per pair: the 5-year interstate probabilities of the 40 pairs with the highest risk;
 - per year: world quantiles (5/25/50/75/95) of the share of countries in conflict, coups and new conflicts.
and a line in outputs/ledger/model_versions.jsonl.
score: for every calendar year that is complete in the UCDP data (final GED or a December candidate file), the
per-country P(in conflict) against "25 or more state-based battle deaths in the year" (UCDP GED events located in the
country: a proxy for the training label, the UCDP/PRIO conflict list, which is published later), with the same-proxy
persistence baseline (in conflict the year before or not, each with its rate over the previous ten years; the old
fixed 0.9 / 0.1 is also reported) and the world share against the fan.
Canonical versions: one per published release, named before any of its outcomes is known
(python -P scripts/annual_ledger.py canonical VERSION "release label"); the score lists them first. Other logged
versions are scored too but are not the release's record (debate 2026-11-05: five versions were frozen in a week).
Coups and new conflicts need the end of 2030 (or a coup list for each year) and are scored then.
"""
import datetime as dt
import glob
import hashlib
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import numpy as np

LEDGER = os.path.join(ROOT, "outputs", "ledger")
FC = os.path.join(ROOT, "outputs", "forecast.json")
YEARS = 5                      # 2026-2030


def version():
    h = hashlib.sha256()
    files = sorted(glob.glob(os.path.join(ROOT, "psychohistory", "*.py"))) + [os.path.join(ROOT, "psychohistory", "ensemble_config.json"), FC]
    for f in files:
        h.update(os.path.basename(f).encode())
        h.update(open(f, "rb").read())
    return h.hexdigest()[:12]


def issue():
    os.makedirs(LEDGER, exist_ok=True)
    d = json.load(open(FC))
    v = version()
    origin = int(d["meta"]["origin"])
    f = os.path.join(LEDGER, f"annual_{origin}_{v}.json")
    if os.path.exists(f):
        return f"already logged: {os.path.basename(f)}"
    yrs = d["world"]["years"][:YEARS]
    num = lambda x: None if x is None else round(float(x), 5)
    countries = {c["code"]: {"p_conflict_year": [num(x) for x in c["conflict_path"][:YEARS]], "in_conflict_2025": c["in_conflict"],
                             "p_coup_by_2030": num(c.get("p_coup_by_2030")), "p_onset_by_2030": num(c.get("p_onset_by_2030")),
                             "coup_5y": num(c.get("coup_5y")), "civil_conflict_5y": num(c.get("civil_conflict_5y"))}
                 for c in d["countries"]}
    pairs = [{k: r[k] for k in ("a", "b", "p_conflict_5y", "p_lo", "p_hi", "ongoing", "observed") if k in r}
             for r in d["interstate"]["top"][:40]]
    world = {k: [[num(q[i]) for q in d["world"][k]] for i in range(YEARS)] for k in ("world_conflict_q", "coups_q", "onsets_q")}
    out = {"version": v, "issued": dt.date.today().isoformat(), "origin": origin, "years": yrs,
           "n_scenarios": d["meta"].get("n_scenarios"), "countries": countries, "pairs": pairs, "world": world}
    json.dump(out, open(f, "w"), indent=1)
    with open(os.path.join(LEDGER, "model_versions.jsonl"), "a") as g:
        g.write(json.dumps({"version": v, "issued": out["issued"], "origin": origin, "file": os.path.basename(f)}) + "\n")
    return f"logged {os.path.basename(f)}"


def canonical_versions():
    """{version: release label} from outputs/ledger/canonical.jsonl"""
    f = os.path.join(LEDGER, "canonical.jsonl")
    if not os.path.exists(f):
        return {}
    return {r["version"]: r["release"] for r in map(json.loads, open(f)) if r.get("version")}


def mark_canonical(version, release):
    vs = [json.loads(l)["version"] for l in open(os.path.join(LEDGER, "model_versions.jsonl"))]
    hit = [v for v in vs if v.startswith(version)]
    if len(hit) != 1:
        return f"version {version} matches {len(hit)} logged versions"
    if hit[0] in canonical_versions():
        return f"{hit[0]} is already canonical"
    with open(os.path.join(LEDGER, "canonical.jsonl"), "a") as g:
        g.write(json.dumps({"version": hit[0], "release": release, "named": dt.date.today().isoformat()}) + "\n")
    return f"{hit[0]} is the canonical record of {release}"


def completed_years():
    """calendar years whose twelve months are all in the UCDP data"""
    from psychohistory import monthly as MO
    ly, lm = MO.last_observed()
    return MO, (ly if lm == 12 else ly - 1)


def score():
    MO, last_full = completed_years()
    gw2iso, codes = MO.gw_map()
    D = MO.deaths(codes, gw2iso, last_full, 12)
    ci = {c: i for i, c in enumerate(codes)}
    yearly = lambda y: D[:, MO.midx(y, 1):MO.midx(y, 12) + 1].sum(1)
    canon = canonical_versions()
    out = {"complete_through": last_full, "canonical": canon, "versions": {}}
    for f in sorted(glob.glob(os.path.join(LEDGER, "annual_*.json"))):
        L = json.load(open(f))
        res = {}
        for j, y in enumerate(L["years"]):
            if y > last_full:
                break
            obs, prev = yearly(y), yearly(y - 1)
            rows = [(c, v["p_conflict_year"][j]) for c, v in L["countries"].items() if c in ci and v["p_conflict_year"][j] is not None]
            p = np.array([r[1] for r in rows])
            yv = np.array([float(obs[ci[c]] >= 25) for c, _ in rows])
            # persistence with its two rates estimated on the same proxy over the ten years before (the fixed 0.9 / 0.1
            # gave every country at peace a 10% chance of war, about four times the rate of new wars: a weak baseline
            # that flattered the model; debate 2026-11-05). The fixed version is kept for continuity.
            st = np.stack([yearly(y_) >= 25 for y_ in range(y - 11, y)], 1)            # [C, 11] years y-11..y-1
            was, now_ = st[:, :-1], st[:, 1:]
            p_stay = float((was & now_).sum() / max(was.sum(), 1))
            p_new = float((~was & now_).sum() / max((~was).sum(), 1))
            pers = np.array([p_stay if prev[ci[c]] >= 25 else p_new for c, _ in rows])
            pers_fixed = np.array([0.9 if prev[ci[c]] >= 25 else 0.1 for c, _ in rows])
            share = float(yv.mean())
            q = L["world"]["world_conflict_q"][j]
            # a year is blind only if the forecast was issued before it began (audit 2026-11-10: the versions issued in
            # November 2026 had 2026 data to September in hand, so their 2026 score is not a forecast)
            res[str(y)] = {"blind": bool(y > int(L["issued"][:4])), "n": len(rows), "in_conflict": int(yv.sum()), "brier_model": round(float(np.mean((p - yv) ** 2)), 5),
                           "brier_persistence": round(float(np.mean((pers - yv) ** 2)), 5),
                           "persistence_rates": [round(p_stay, 4), round(p_new, 4)],
                           "brier_persistence_fixed": round(float(np.mean((pers_fixed - yv) ** 2)), 5),
                           "share_observed": round(share, 4),
                           "share_fan_5_95": [q[0], q[4]], "share_in_90": bool(q[0] <= share <= q[4])}
        res = res or None
        out["versions"][L["version"]] = {"canonical": canon.get(L["version"]), "years": res} if res else \
            {"canonical": canon.get(L["version"]), "years": f"first year ({L['years'][0]}) not complete yet (data through {last_full})"}
        out["versions"][L["version"]]["first_blind_year"] = int(L["issued"][:4]) + 1
    return out


if __name__ == "__main__":
    r = issue() if sys.argv[1] == "issue" else mark_canonical(sys.argv[2], sys.argv[3]) if sys.argv[1] == "canonical" \
        else score()
    print(json.dumps(r, indent=1) if isinstance(r, dict) else r)
