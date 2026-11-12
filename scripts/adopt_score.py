"""The adopt rule (docs/adopt_rule.md; this form since 12 November 2026): score a simulator change against the baseline
on the full-system hindcasts in the forecast's configuration.

Inputs: scripts/world_hindcast.py runs of the baseline and the variant on the same ensemble refit (tags BASE_s0 and
VAR_s0), one origin for every year 1970-2009, 15 years each. For each statistic the score is the CRPS of the simulated
worlds against what happened (the per-country conflict Brier score for "country_brier"). The composite is the weighted
mean of the variant's relative change per statistic (negative = better).

Decision (all must hold):
 1. the 90% interval of the composite lies below zero. The interval comes from resampling the origins in blocks of ten
    consecutive years (moving blocks, wrapped; 2,000 resamples, fixed seed): neighbouring origins share most of their
    15-year window, so single origins are not independent draws;
 2. no weighted statistic is worse by more than 3% pooled (guardrail), and the safe-country war score is not worse
    with a 90% interval above zero;
 3. the configuration matches the forecast's (scripts/config_parity.py; the variant's own change aside).
The composite by decade and on any further refits given with --seeds are reported, not decided on.
Until 11 November 2026 the rule used three refits, 16 origins, a noise floor (twice the spread of the baseline between
refits) and "better in 3 of 4 decades"; why it changed is in docs/adopt_rule.md. --floor still measures that spread.

python -P scripts/adopt_score.py BASE VAR [--seeds 0] [--origins ...]
python -P scripts/adopt_score.py --floor BASE [--seeds 0 1 2] [--origins ...]"""
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
ORIGINS = list(range(1970, 2010))
ORIGINS_OLD = [1970, 1973, 1976, 1979, 1980, 1983, 1986, 1989, 1990, 1993, 1996, 1999, 2000, 2003, 2006, 2009]   # until 11 November 2026
BLOCK, N_BOOT = 10, 2000


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


def active_weights(base=None):
    """the weights and the noise floor of this baseline (audit 2026-11-10: one shared noise_floor.json was used for every
    baseline, whichever it had been measured on); the shared file is read only if it was measured on `base`"""
    f = os.path.join(OUT, f"noise_floor_{base}.json")
    if not os.path.exists(f):
        f = os.path.join(OUT, "noise_floor.json")
    if not os.path.exists(f):
        return dict(WEIGHTS), None
    nf = json.load(open(f))
    if base is not None and nf.get("base") != base:
        return dict(WEIGHTS), None
    w = {k: v for k, v in WEIGHTS.items() if k not in nf.get("dropped", [])}
    return w, nf


def block_resamples(origins, rng, n=N_BOOT, block=BLOCK):
    """moving-block resamples of the origins (sorted, wrapped): blocks of `block` consecutive origins"""
    o = sorted(origins)
    m = len(o)
    step = max(1, round(block * m / (o[-1] - o[0] + 1)))        # origins per `block` years
    for _ in range(n):
        starts = rng.integers(0, m, int(np.ceil(m / step)))
        yield [o[(s_ + k) % m] for s_ in starts for k in range(step)][:m]


def decide(base, var, seeds, origins):
    w = dict(WEIGHTS)
    A, B = table(base, seeds, origins), table(var, seeds, origins)
    rel, comp = composite(A, B, origins, w)
    by_dec = {}
    for d in sorted({decade(o) for o in origins}):
        os_ = [o for o in origins if decade(o) == d]
        by_dec[d] = composite(A, B, os_, w)[1]
    s20 = lambda oo: float(np.mean([B[o]["safe20_wars"] for o in oo]) / max(np.mean([A[o]["safe20_wars"] for o in oo]), 1e-12) - 1)
    rng = np.random.default_rng(20261112)
    boot, boot20 = [], []
    for oo in block_resamples(origins, rng):
        boot.append(composite(A, B, oo, w)[1])
        boot20.append(s20(oo))
    lo, hi = (float(x) for x in np.percentile(boot, [5, 95]))
    guard = {k: v for k, v in rel.items() if v > GUARD}
    if float(np.percentile(boot20, 5)) > 0:
        guard["safe20_wars"] = s20(origins)
    ok = hi < 0 and not guard
    res = {"base": base, "variant": var, "seeds": seeds, "origins": origins, "weights": w,
           "relative_change": {k: round(v, 4) for k, v in rel.items()}, "composite": round(comp, 4),
           "composite_interval_90": [round(lo, 4), round(hi, 4)],
           "composite_by_decade": {k: round(v, 4) for k, v in by_dec.items()},
           "guardrail_breaches": guard, "adopt": bool(ok), "verdict": "adopt" if ok else "reject"}
    if len(seeds) > 1:
        res["composite_by_refit"] = {s_: round(composite(table(base, [s_], origins), table(var, [s_], origins), origins, w)[1], 4) for s_ in seeds}
    diag = {"safe20_wars": round(s20(origins), 4)}
    if "gdp_falls" in A[origins[0]]:
        diag["gdp_falls"] = round(float(np.mean([B[o]["gdp_falls"] for o in origins]) / max(np.mean([A[o]["gdp_falls"] for o in origins]), 1e-12) - 1), 4)
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
    json.dump(res, open(os.path.join(OUT, f"noise_floor_{base}.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(OUT, "noise_floor.json"), "w"), indent=1)        # (the latest, for reference)
    return res


def main():
    a = sys.argv[1:]
    seeds = [0, 1, 2] if a and a[0] == "--floor" else [0]
    origins = ORIGINS_OLD if a and a[0] == "--floor" else ORIGINS
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
