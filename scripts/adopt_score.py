"""The adopt rule (docs/adopt_rule.md, fixed 5 November 2026): score a simulator change against the baseline on the
full-system hindcasts in the forecast's configuration, with leave-one-decade-out decisions and the measured noise.

Inputs: scripts/world_hindcast.py runs of the baseline and the variant, tags BASE_s<seed> and VAR_s<seed> (one per
ensemble refit, world_hindcast --ens-seed), at the same origins. For each statistic the score is the CRPS of the
simulated worlds against what happened (the per-country conflict Brier score for "country_brier"), averaged over the
seeds. The composite is the weighted mean of the variant's relative change per statistic (negative = better).

Decision (all must hold):
 1. the composite improves in at least 3 of the 4 decades of origins (1970s, 1980s, 1990s, 2000s);
 2. the pooled composite improves by more than the noise floor (twice the standard deviation of the composite
    between ensemble refits of the baseline itself, outputs/adopt/noise_floor.json; scripts/adopt_score.py --floor);
 3. no statistic with weight is worse by more than 3% pooled (guardrail), and the safe-country war score is not worse
    by more than its own noise floor (non-regression guardrail, added 6 November 2026);
 4. the configuration matches the forecast's (scripts/config_parity.py; the variant's own change aside).
Statistics whose noise floor exceeds 10% carry no weight (listed in the noise-floor file).

python -P scripts/adopt_score.py BASE VAR [--seeds 0 1 2] [--origins ...]
python -P scripts/adopt_score.py --floor BASE [--seeds 0 1 2] [--origins ...]
   (the noise floor: each refit of BASE scored against the mean of the others, as if it were a variant)"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
HC = os.path.join(ROOT, "outputs", "world_hindcast")
OUT = os.path.join(ROOT, "outputs", "adopt")
# fixed before any decision under this rule (5 November 2026). What the report publishes: each country's war odds,
# the world's war level, new wars and coups, world growth, how far countries drift apart, democracy, wars between
# states, oil. safe20_wars and gdp_falls are diagnostics (counts of rare events), not decision statistics.
WEIGHTS = {"country_brier": 0.25, "conflict_share": 0.15, "onsets": 0.10, "coups": 0.10, "gdp_growth": 0.10,
           "gdp_dispersion": 0.10, "dem_change": 0.05, "interstate": 0.05, "oil_change": 0.05, "gdp_vol": 0.05}
GUARD = 0.03
ORIGINS = [1970, 1973, 1976, 1979, 1980, 1983, 1986, 1989, 1990, 1993, 1996, 1999, 2000, 2003, 2006, 2009]


def crps(x, y):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float(np.mean(np.abs(x - y)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2)


def scores(tag, O):
    z = np.load(os.path.join(HC, f"{tag}_{O}.npz"))
    r = {str(k): crps(z["sim"][:, j], float(z["obs"][j])) for j, k in enumerate(z["names"])}
    r["country_brier"] = json.loads(str(z["country"]))["conflict_brier"]
    return r


def table(tag, seeds, origins):
    """{origin: {stat: score averaged over seeds}}"""
    out = {}
    for O in origins:
        per = [scores(f"{tag}_s{s}", O) for s in seeds]
        out[O] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    return out


def decade(O):
    return f"{(O // 10) * 10}s"


def composite(A, B, origins, weights):
    """relative change of B against A per statistic (pooled over the origins given), and the weighted composite"""
    rel = {k: float(np.mean([B[o][k] for o in origins]) / max(np.mean([A[o][k] for o in origins]), 1e-12) - 1)
           for k in weights}
    tot = sum(weights.values())
    return rel, sum(w * rel[k] for k, w in weights.items()) / tot


def active_weights():
    f = os.path.join(OUT, "noise_floor.json")
    if not os.path.exists(f):
        return dict(WEIGHTS), None
    nf = json.load(open(f))
    w = {k: v for k, v in WEIGHTS.items() if k not in nf.get("dropped", [])}
    return w, nf


def decide(base, var, seeds, origins):
    w, nf = active_weights()
    A, B = table(base, seeds, origins), table(var, seeds, origins)
    rel, comp = composite(A, B, origins, w)
    by_dec = {}
    for d in sorted({decade(o) for o in origins}):
        os_ = [o for o in origins if decade(o) == d]
        by_dec[d] = composite(A, B, os_, w)[1]
    better_dec = sum(v < 0 for v in by_dec.values())
    floor = nf["composite_floor"] if nf else None
    guard = {k: v for k, v in rel.items() if v > GUARD}
    # non-regression guardrail on the safe-country tail (second debate, 6 November 2026): the 15-year score of new
    # wars among the 20 safest countries may not get worse by more than its noise between refits
    if "safe20_wars" in A[origins[0]]:
        s20 = float(np.mean([B[o]["safe20_wars"] for o in origins]) / max(np.mean([A[o]["safe20_wars"] for o in origins]), 1e-12) - 1)
        s20_floor = (nf or {}).get("safe20_floor")
        if s20_floor is not None and s20 > s20_floor:
            guard["safe20_wars"] = s20
    ok = better_dec >= 3 and (floor is not None and comp < -floor) and not guard
    res = {"base": base, "variant": var, "seeds": seeds, "origins": origins, "weights": w,
           "relative_change": {k: round(v, 4) for k, v in rel.items()}, "composite": round(comp, 4),
           "composite_by_decade": {k: round(v, 4) for k, v in by_dec.items()}, "decades_better": better_dec,
           "noise_floor": floor, "guardrail_breaches": guard, "adopt": bool(ok),
           "verdict": "adopt" if ok else ("no noise floor measured yet" if floor is None else "reject")}
    # diagnostics, outside the decision
    diag = {}
    for k in ("safe20_wars", "gdp_falls"):
        if k in A[origins[0]]:
            diag[k] = round(float(np.mean([B[o][k] for o in origins]) / max(np.mean([A[o][k] for o in origins]), 1e-12) - 1), 4)
    res["diagnostics"] = diag
    return res


def floor(base, seeds, origins):
    """each refit scored against the mean of the other refits' scores, as if it were a variant: the spread of the
    composite between refits is the change a do-nothing variant shows"""
    T = {s: table(base, [s], origins) for s in seeds}
    comps, per_stat = [], {k: [] for k in WEIGHTS}
    for s in seeds:
        others = [t for t in seeds if t != s]
        M = {o: {k: float(np.mean([T[t][o][k] for t in others])) for k in T[s][o]} for o in origins}
        rel, c = composite(M, T[s], origins, WEIGHTS)
        comps.append(c)
        for k in WEIGHTS:
            per_stat[k].append(rel[k])
    sd = {k: float(np.std(v, ddof=1)) for k, v in per_stat.items()}
    # a single statistic's floor: twice its spread between refits; above 10% it cannot decide anything
    dropped = [k for k, v in sd.items() if 2 * v > 0.10]
    w = {k: v for k, v in WEIGHTS.items() if k not in dropped}
    comps_w = []
    for s in seeds:
        others = [t for t in seeds if t != s]
        M = {o: {k: float(np.mean([T[t][o][k] for t in others])) for k in T[s][o]} for o in origins}
        comps_w.append(composite(M, T[s], origins, w)[1])
    s20 = []
    for s in seeds:
        others = [t for t in seeds if t != s]
        a_ = np.mean([np.mean([T[t][o]["safe20_wars"] for t in others]) for o in origins])
        s20.append(np.mean([T[s][o]["safe20_wars"] for o in origins]) / max(a_, 1e-12) - 1)
    res = {"base": base, "seeds": seeds, "origins": origins, "safe20_floor": round(2 * float(np.std(s20, ddof=1)), 4),
           "stat_floor": {k: round(2 * v, 4) for k, v in sd.items()}, "dropped": dropped,
           "composite_floor": round(2 * float(np.std(comps_w, ddof=1)), 4),
           "note": "floor = 2 x sd between ensemble refits of the baseline (each against the mean of the others)"}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "noise_floor.json"), "w"), indent=1)
    return res


def main():
    a = sys.argv[1:]
    seeds = [0, 1, 2]
    origins = ORIGINS
    if "--seeds" in a:
        i = a.index("--seeds")
        j = next((k for k in range(i + 1, len(a)) if a[k].startswith("--")), len(a))
        seeds = [int(x) for x in a[i + 1:j]]
        a = a[:i] + a[j:]
    if "--origins" in a:
        i = a.index("--origins")
        j = next((k for k in range(i + 1, len(a)) if a[k].startswith("--")), len(a))
        origins = [int(x) for x in a[i + 1:j]]
        a = a[:i] + a[j:]
    if a[0] == "--floor":
        r = floor(a[1], seeds, origins)
    else:
        r = decide(a[0], a[1], seeds, origins)
        os.makedirs(OUT, exist_ok=True)
        json.dump(r, open(os.path.join(OUT, f"{a[1]}_vs_{a[0]}.json"), "w"), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
