"""The most disastrous simulated futures and why they are possible (root-cause analysis of the forecast's tails).

python -P scripts/disaster_rca.py sim      headline worlds (uncertain AI, SSP2-4.5, climate damage x1, every
                                           assumption sampled; 2 x 4,000 worlds) -> outputs/disaster/worlds.npz,
                                           event logs of the worst worlds -> outputs/disaster/logs.json
python -P scripts/disaster_rca.py knock    the same worlds (same seeds) with one mechanism switched off at a time
                                           -> outputs/disaster/knockouts.json
python -P scripts/disaster_rca.py analyze  -> outputs/disaster/rca.json (printed summary)

Disasters, per world (each the worst 1% of the 8,000 worlds):
- economic: lowest world GDP per person in 2065
- deaths: most war and nuclear deaths over 2026-2065 (battle deaths x (1 + civilian multiplier) + nuclear deaths)
- civil war: highest mean share of countries at war 2026-2065
- autocracy: lowest population-weighted democracy in 2065
- composite: highest mean of the four percentile ranks
Root causes, three ways:
1. assumptions: which sampled assumptions separate disaster worlds from the rest (AUC of each assumption)
2. events: how much likelier each shock (great-power war, nuclear use, pandemic, bubble, oil spike, ...) is in them,
   and which came first
3. mechanisms: the same 8,000 worlds rerun with one mechanism switched off; the share of the disaster tail (worlds
   beyond the baseline's 1% threshold) that disappears. Random draws diverge after the first change, so this is a
   comparison of tail frequencies, not of matched worlds.
For the economic tail, the 2065 world GDP gap to the median world is also split into underlying growth, war and shock
damage, climate damage and population mix, and by country."""
import dataclasses
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

OUTD = os.path.join(os.path.dirname(__file__), "..", "outputs", "disaster")
SEEDS = (300, 301)
N = 4000
TAIL = 0.01


def setup():
    from psychohistory import pipeline as PL, interstate as INTER
    from psychohistory.backtest import flatten, train_mask, ALL_FAMILIES
    from psychohistory.models import SeldonEnsemble, load_combine
    from psychohistory.features import TASKS
    from psychohistory.simulate import fit_transitions
    from psychohistory.data import LAST_OBS_YEAR
    p, base, X, Yl = PL.prepare(False)
    Xf, Yf, ci, year = flatten(p, X, Yl)
    Ytr, tr = train_mask(Yf, year, LAST_OBS_YEAR, first_year=1860)
    # as the forecast (pipeline.step_forecast), with 3 ensemble members instead of 10
    ens = SeldonEnsemble(load_combine(), members=3, families=ALL_FAMILIES).fit(Xf[tr], Ytr[tr], ci[tr], year[tr])
    X_now = X[:, p.yi(LAST_OBS_YEAR)]
    parts = ens.predict_families(X_now)
    net = ens.fam["N"][0]
    net.nets = [n for m in ens.fam["N"] for n in m.nets]
    lgf = lambda v: np.log(np.clip(v, 1e-4, 1 - 1e-4) / (1 - np.clip(v, 1e-4, 1 - 1e-4)))
    fam_l = np.stack([lgf(v) for v in parts.values()])
    net.struct_sd = np.nan_to_num(np.nanstd(fam_l, 0)[:, [TASKS.index(t) for t in ("onset_1y", "termination_1y", "coup_1y")]])
    PL.attach_ensemble(net, ens, X[:, p.yi(LAST_OBS_YEAR)])     # (structural inputs + ensemble central offset)
    growth, ddem = fit_transitions(p, X, base, LAST_OBS_YEAR)
    dy = INTER.sim_inputs(INTER.run(verbose=False, backtests=False))
    clim = PL.climate_summary(p, base, LAST_OBS_YEAR, 2065)["ssp245"]["inputs"]
    return p, base, net, growth, ddem, dy, clim, LAST_OBS_YEAR


def run(env, seed, cfg=None, chronicle=0, record=True, **kw):
    from psychohistory.simulate import simulate
    from psychohistory import world as WM
    p, base, net, growth, ddem, dy, clim, O = env
    cfg = cfg or WM.WorldConfig(ai=WM.AI_SCENARIOS["uncertain"], sample_params=True)
    kw.setdefault("climate_damage", 1.0)
    return simulate(p, base, net, growth, ddem, O, 2065, n_scen=N, clim=clim, world=cfg, dyads=dy, seed=seed,
                    record_paths=N if record else 0, chronicle=chronicle, **kw)


def per_world(o):
    """per-world summary series [S, T] and scalars"""
    w = o["world"]
    s = lambda k: w[k].reshape(w[k].shape[0], -1, w[k].shape[-1]).sum(1) if k in w else None   # [S, T]
    civ = w["params"].get("civ_mult", 1.5) if "params" in w else 1.5
    bd = s("battle_deaths")
    deaths = (bd if bd is not None else np.zeros_like(o["world_gdp"])) * (1 + np.reshape(civ, (-1, 1)))
    if s("nuclear_deaths") is not None:
        deaths = deaths + s("nuclear_deaths")
    d = {"world_gdp": o["world_gdp"], "world_conflict": o["world_conflict"], "world_dem": o["world_dem"],
         "deaths": deaths, "onsets": o["onsets"], "coups": o["coups"]}
    for k in ("great_power_war", "great_power_war_cow", "nuclear_use", "nuclear_large", "pandemic", "eng_pandemic",
              "bubble", "solar", "volcano", "oil", "wars", "sanctioned", "nuclear_states", "excess_deaths", "food_price"):
        if s(k) is not None:
            d[k] = s(k)
    return {k: np.asarray(v, np.float32) for k, v in d.items()}


def scores(d):
    """the disaster measures per world (higher = worse)"""
    return {"economic": -np.log(d["world_gdp"][:, -1]), "deaths": d["deaths"].sum(1),
            "civil_war": d["world_conflict"].mean(1), "autocracy": -d["world_dem"][:, -1]}


def rank01(a):
    return np.argsort(np.argsort(a)) / (len(a) - 1)


def event_log(o, wi, p, base, O):
    """compact yearly log of world wi (chronicle entries)"""
    log = []
    oil0 = base["oil_price"][0][p.yi(O)]
    nm = lambda c: p.names[p.codes.index(c)]
    for yr in o["chronicle"]:
        e = {"year": yr["year"], "world_gdp": round(yr["world_gdp"][wi]), "conflict_share": round(yr["conflict_share"][wi], 3)}
        for k in ("onset", "coup", "dem_drop", "slump"):
            if yr[k][wi]:
                e[k] = [nm(c) for c in yr[k][wi]]
        if yr.get("new_wars") and yr["new_wars"][wi]:
            e["new_interstate"] = [f"{nm(a)} vs {nm(b)}" + (" (war)" if wr else "") for a, b, wr in yr["new_wars"][wi]]
        if yr.get("pandemic") and yr["pandemic"][wi] > 0:
            e["pandemic"] = True
        if yr.get("oil") is not None:
            e["oil_index"] = round(float(100 * np.exp(yr["oil"][wi] - oil0)))
        log.append(e)
    return log


def cmd_sim():
    os.makedirs(OUTD, exist_ok=True)
    env = setup()
    p, base, O = env[0], env[1], env[7]
    D, P, paths, logs_raw = {}, {}, {}, []
    for seed in SEEDS:
        o = run(env, seed, chronicle=N)
        d = per_world(o)
        for k, v in d.items():
            D.setdefault(k, []).append(v)
        for k, v in o["world"]["params"].items():
            P.setdefault(k, []).append(np.asarray(v, np.float32))
        for k, v in o["paths"].items():
            paths.setdefault(k, []).append(v.astype(np.float32))
        # keep the event logs of each chunk's worst 40 worlds by the composite and by each measure
        sc = scores(d)
        comp = np.mean([rank01(v) for v in sc.values()], 0)
        keep = set(np.argsort(-comp)[:40].tolist())
        for v in sc.values():
            keep |= set(np.argsort(-v)[:15].tolist())
        for wi in sorted(keep):
            logs_raw.append({"world": int(wi + N * SEEDS.index(seed)), "events": event_log(o, wi, p, base, O)})
        del o
        print("chunk", seed, "done", flush=True)
    cat = lambda dd: {k: np.concatenate(v) for k, v in dd.items()}
    D, P, paths = cat(D), cat(P), cat(paths)
    np.savez_compressed(os.path.join(OUTD, "worlds.npz"), **{f"d_{k}": v for k, v in D.items()},
                        **{f"p_{k}": v for k, v in P.items()},
                        **{f"x_{k}": v for k, v in paths.items()}, codes=np.array(p.codes), names=np.array(p.names),
                        years=np.arange(O + 1, 2066))
    json.dump(logs_raw, open(os.path.join(OUTD, "logs.json"), "w"))
    print("saved", {k: v.shape for k, v in D.items()})


KNOCKOUTS = {
    "no_interstate_wars": dict(cfg=dict(interstate=False)),
    "no_nuclear_use": dict(cfg=dict(nuc_use_both=0.0, nuc_use_one=0.0)),
    "no_pandemics": dict(cfg=dict(pandemic_p=0.0, tech_force={"pathogens": None}), ai=dict(eng_pandemic=0.0)),   # (and no pathogen tools)
    "no_climate_damage": dict(kw=dict(climate_damage=0.0)),
    "no_ai": dict(ai_none=True),
    "no_bubbles": dict(cfg=dict(bubble_p=0.0)),
    "no_alliance_cascades": dict(cfg=dict(cascade_q=0.0, cascade_hop2=0.0, quasi_USA_TWN=0.0, quasi_USA_ISR=0.0,
                                          quasi_RUS_IRN=0.0, quasi_CHN_PAK=0.0)),
    "no_war_spillover": dict(cfg=dict(war_spillover=0.0)),
    "no_financial_contagion": dict(cfg=dict(fin_contagion=0.0)),
    "no_country_frailty": dict(kw=dict(frailty_sd=0.0)),
    "no_world_mood": dict(kw=dict(mood_sd=(0.0, 0.0))),
    "no_tail_catastrophes": dict(cfg=dict(solar_p=0.0, volc_p=0.0)),
    # climate (2026-10-29): tails and channels off (a parameter set to its default would still be sampled, hence 1e-9)
    "no_climate_tails": dict(cfg=dict(amoc_p=1e-9, slr_ice_p=0.0, clim_tail_df=0.0, harvest_amp=1e-9, clim_conflict=1e-9,
                                      clim_migr=1e-9)),
    "smallest_climate_damage": dict(cfg=dict(dmg_family=0.0)),
}


def knock_cfg(spec):
    from psychohistory import world as WM
    ai = WM.AI_SCENARIOS["none"] if spec.get("ai_none") else WM.AI_SCENARIOS["uncertain"]
    if spec.get("ai"):
        ai = dataclasses.replace(ai, **{k: v for k, v in spec["ai"].items() if hasattr(ai, k)})
    c = {k: v for k, v in spec.get("cfg", {}).items() if hasattr(WM.WorldConfig(), k)}
    missing = set(spec.get("cfg", {})) - set(c)
    if missing:
        print("  (not WorldConfig fields:", missing, ")")
    return WM.WorldConfig(ai=ai, sample_params=True, **c)


def cmd_knock():
    env = setup()
    base_d = np.load(os.path.join(OUTD, "worlds.npz"))
    sc0 = scores({k[2:]: base_d[k] for k in base_d.files if k.startswith("d_")})
    thr = {k: np.quantile(v, 1 - TAIL) for k, v in sc0.items()}
    out = {"thresholds": {k: float(v) for k, v in thr.items()}, "runs": {}}
    f = os.path.join(OUTD, "knockouts.json")
    if os.path.exists(f):
        out["runs"] = json.load(open(f))["runs"]
    for name, spec in KNOCKOUTS.items():
        if name in out["runs"]:
            continue
        sc = {k: [] for k in sc0}
        for seed in SEEDS:
            o = run(env, seed, cfg=knock_cfg(spec), record=False, **spec.get("kw", {}))
            for k, v in scores(per_world(o)).items():
                sc[k].append(v)
            del o
        sc = {k: np.concatenate(v) for k, v in sc.items()}
        out["runs"][name] = {k: {"tail_share": float((sc[k] > thr[k]).mean()), "median": float(np.median(sc[k])),
                                 "q99": float(np.quantile(sc[k], 0.99))} for k in sc}
        print(name, {k: round(v["tail_share"] / TAIL, 2) for k, v in out["runs"][name].items()}, flush=True)
        json.dump(out, open(f, "w"), indent=1)


def origin_values(codes):
    """real oil price (log) and world GDP per person (population-weighted, same countries) at the 2025 origin"""
    from psychohistory.data import get_panel, LAST_OBS_YEAR
    from psychohistory.features import base_arrays
    p = get_panel()
    b = base_arrays(p)
    t = p.yi(LAST_OBS_YEAR)
    ix = [p.codes.index(c) for c in codes]
    lg, lp = b["lg"][ix, t], b["lpop"][ix, t]
    ok = np.isfinite(lg) & np.isfinite(lp)
    return float(b["oil_price"][0][t]), float((np.exp(lg[ok]) * np.exp(lp[ok])).sum() / np.exp(lp[ok]).sum())


def auc(x, y):
    """P(x in disaster > x elsewhere), ties half"""
    from scipy.stats import rankdata
    r = rankdata(x)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cmd_analyze():
    z = np.load(os.path.join(OUTD, "worlds.npz"))
    D = {k[2:]: z[k] for k in z.files if k.startswith("d_")}
    P = {k[2:]: z[k].ravel() for k in z.files if k.startswith("p_")}
    X = {k[2:]: z[k] for k in z.files if k.startswith("x_")}
    codes, names, years = list(z["codes"]), list(z["names"]), z["years"]
    S = len(D["world_gdp"])
    sc = scores(D)
    sc["composite"] = np.mean([rank01(v) for v in sc.values()], 0)
    rng = np.random.default_rng(0)
    res = {"n_worlds": S, "years": [int(years[0]), int(years[-1])], "sets": {}}
    # event indicators per world (first year, or nan)
    first = lambda a: np.where(a.any(1), years[np.argmax(a, 1)], np.nan)
    # (the world layer records pandemic = natural or engineered, nuclear_use = any use including a large exchange:
    # split them so each world counts once, the more specific event listed first for the "first shock" tie-break)
    Z0 = np.zeros_like(D["onsets"])
    eng = D.get("eng_pandemic", Z0) > 0
    large = D.get("nuclear_large", Z0) > 0
    ev = {"great-power war": D["great_power_war"] > 0, "large nuclear exchange": large,
          "limited nuclear use": (D.get("nuclear_use", Z0) > 0) & ~large,
          "engineered pandemic": eng, "natural pandemic": (D.get("pandemic", Z0) > 0) & ~eng,
          "AI bubble burst": D.get("bubble", Z0) > 0, "solar superstorm": D.get("solar", Z0) > 0,
          "VEI 6+ eruption": D.get("volcano", Z0) > 0}
    # baselines at the 2025 origin (the first simulated year is 2026)
    oil0, wg0 = origin_values(codes)
    if "oil" in D:
        ev["oil price doubles"] = D["oil"] - oil0 > np.log(2)
    # a year with at least 2 sd more new civil wars than the world's mean (an onset wave)
    on = D["onsets"]
    ev["civil-war onset wave"] = on > np.median(on) + 2 * on.std()
    lw = np.log(D["world_gdp"])
    ev["world GDP falls 5%+ in a year"] = np.diff(lw, axis=1, prepend=np.full((len(lw), 1), np.log(wg0))) < -0.05
    ev_first = {k: first(v) for k, v in ev.items()}
    ev_any = {k: v.any(1) for k, v in ev.items()}
    params = {k: v for k, v in P.items() if np.std(v) > 0}
    for name, s in sc.items():
        cut = np.quantile(s, 1 - TAIL)
        y = s >= cut
        r = {"n": int(y.sum())}
        # 1. assumptions; null AUC spread by permutation
        null = np.array([abs(auc(params["climate_damage"], rng.permutation(y)) - 0.5) for _ in range(200)])
        r["auc_null_95"] = float(np.quantile(null, 0.95))
        a_ = {k: auc(v, y) for k, v in params.items()}
        r["assumptions"] = [{"name": k, "auc": round(float(a), 3), "median_disaster": float(np.median(params[k][y])),
                             "median_all": float(np.median(params[k])),
                             "q10": float(np.quantile(params[k], .1)), "q90": float(np.quantile(params[k], .9))}
                            for k, a in sorted(a_.items(), key=lambda kv: -abs(kv[1] - 0.5))[:10]]
        # 2. events
        r["events"] = [{"event": k, "share_disaster": round(float(v[y].mean()), 3), "share_all": round(float(v.mean()), 3),
                        "ratio": round(float(v[y].mean() / max(v.mean(), 1e-9)), 2),
                        "median_first_year": float(np.nanmedian(ev_first[k][y])) if v[y].any() else None}
                       for k, v in ev_any.items()]
        # which listed shock came first in each disaster world
        fy = np.stack([ev_first[k] for k in ev if k not in ("world GDP falls 5%+ in a year",)], 1)
        kn = [k for k in ev if k not in ("world GDP falls 5%+ in a year",)]
        firsts = {}
        for i in np.where(y)[0]:
            if np.isfinite(fy[i]).any():
                j = int(np.nanargmin(fy[i]))
                firsts[kn[j]] = firsts.get(kn[j], 0) + 1
            else:
                firsts["none of these"] = firsts.get("none of these", 0) + 1
        r["first_shock"] = dict(sorted(firsts.items(), key=lambda kv: -kv[1]))
        r["outcomes_disaster"] = {"world_gdp_2065": float(np.median(D["world_gdp"][y, -1])),
                                  "world_gdp_2026": float(np.median(D["world_gdp"][y, 0])),
                                  "deaths_m": float(np.median(D["deaths"][y].sum(1)) / 1e6),
                                  "conflict_mean": float(np.median(D["world_conflict"][y].mean(1))),
                                  "dem_2065": float(np.median(D["world_dem"][y, -1]))}
        res["sets"][name] = r
    res["outcomes_all"] = {"world_gdp_2065": float(np.median(D["world_gdp"][:, -1])), "world_gdp_2026": float(np.median(D["world_gdp"][:, 0])),
                           "deaths_m": float(np.median(D["deaths"].sum(1)) / 1e6), "deaths_m_q99": float(np.quantile(D["deaths"].sum(1), .99) / 1e6),
                           "conflict_mean": float(np.median(D["world_conflict"].mean(1))), "dem_2065": float(np.median(D["world_dem"][:, -1])),
                           "p_gdp_2065_below_2026": float((D["world_gdp"][:, -1] < D["world_gdp"][:, 0]).mean()),
                           "world_gdp_2065_q01": float(np.quantile(D["world_gdp"][:, -1], .01))}
    # economic tail: decomposition of the 2065 gap to the middle worlds (40-60th percentile)
    g = D["world_gdp"][:, -1]
    y = g <= np.quantile(g, TAIL)
    mid = (g >= np.quantile(g, .4)) & (g <= np.quantile(g, .6))
    gdp, pop, wd, cd = (np.nan_to_num(X[k][:, :, -1]) for k in ("gdp", "pop", "war_dmg", "clim_dmg"))
    ok = np.isfinite(X["gdp"][:, :, -1]) & np.isfinite(X["pop"][:, :, -1])
    pop = np.where(ok, pop, 0)
    wsum = lambda v, m: ((v * pop)[m].sum(1) / pop[m].sum(1))
    lw = lambda gg, m: np.log(wsum(gg, m)).mean()
    under = gdp * np.exp(-wd - cd)
    tot = lw(gdp, y) - lw(gdp, mid)
    # sequential: underlying growth (no damages), + war/shock damage, + climate damage, with each set's own population;
    # population mix = the gap left when the disaster worlds' GDP is weighted with the middle worlds' mean population
    pop_mid = pop[mid].mean(0, keepdims=True)
    mix = np.log(((gdp * pop)[y].sum(1) / pop[y].sum(1))).mean() - np.log(((gdp[y] * pop_mid).sum(1) / pop_mid.sum())).mean()
    res["economic_decomposition"] = {
        "total_log_gap": float(tot),
        "underlying_growth": float(lw(under, y) - lw(under, mid)),
        "war_and_shock_damage": float((lw(gdp * np.exp(-cd), y) - lw(under, y)) - (lw(gdp * np.exp(-cd), mid) - lw(under, mid))),
        "climate_damage": float((lw(gdp, y) - lw(gdp * np.exp(-cd), y)) - (lw(gdp, mid) - lw(gdp * np.exp(-cd), mid))),
        "population_mix_part": float(mix - (np.log(((gdp * pop)[mid].sum(1) / pop[mid].sum(1))).mean()
                                            - np.log(((gdp[mid] * pop_mid).sum(1) / pop_mid.sum())).mean()))}
    # by country: contribution to the gap in world GDP per person (2065), middle-world population weights
    wsh = pop_mid[0] / pop_mid.sum()
    contrib = wsh * (np.median(np.where(ok[y], gdp[y], np.nan), 0) - np.median(np.where(ok[mid], gdp[mid], np.nan), 0))
    contrib = np.nan_to_num(contrib)
    gap = float(np.median(g[y]) - np.median(g[mid]))
    top = np.argsort(contrib)[:12]
    res["economic_by_country"] = {"gap_usd": gap, "countries": [
        {"code": codes[i], "name": names[i], "contribution_usd": round(float(contrib[i])),
         "gdp_disaster": round(float(np.nanmedian(np.where(ok[y, i], gdp[y, i], np.nan)))),
         "gdp_middle": round(float(np.nanmedian(np.where(ok[mid, i], gdp[mid, i], np.nan)))),
         "war_damage_disaster": round(float(np.median(wd[y, i])), 3), "war_damage_middle": round(float(np.median(wd[mid, i])), 3),
         "conflict_share_disaster": round(float(np.nanmean(X["conflict"][y, i])), 3),
         "conflict_share_middle": round(float(np.nanmean(X["conflict"][mid, i])), 3)} for i in top]}
    # civil-war tail: which countries are at war in it, beyond the middle worlds
    cs = sc["civil_war"]
    yc, mc = cs >= np.quantile(cs, 1 - TAIL), (cs >= np.quantile(cs, .4)) & (cs <= np.quantile(cs, .6))
    cw = np.nanmean(X["conflict"][yc], (0, 2)) - np.nanmean(X["conflict"][mc], (0, 2))
    res["civil_war_by_country"] = [{"name": names[i], "share_disaster": round(float(np.nanmean(X["conflict"][yc, i])), 3),
                                    "share_middle": round(float(np.nanmean(X["conflict"][mc, i])), 3)} for i in np.argsort(-cw)[:12]]
    # is the civil-war tail one region / a few countries, or a broad wave? (number of countries above their middle level)
    res["civil_war_breadth"] = {"countries_up_10pp": int((cw > 0.10).sum()), "countries_up_5pp": int((cw > 0.05).sum())}
    # pick worlds to narrate: worst composite, worst in each measure (from those with logs)
    logs = {w["world"]: w["events"] for w in json.load(open(os.path.join(OUTD, "logs.json")))}
    pick = {}
    for name in ("composite", "economic", "deaths", "civil_war", "autocracy"):
        for i in np.argsort(-sc[name]):
            if int(i) in logs and int(i) not in pick.values():
                pick[name] = int(i)
                break
    res["picked"] = {}
    for name, i in pick.items():
        res["picked"][name] = {"world": i, "percentile": {k: round(float(rank01(v)[i]), 3) for k, v in sc.items()},
                               "world_gdp_2065": float(D["world_gdp"][i, -1]), "deaths_m": float(D["deaths"][i].sum() / 1e6),
                               "conflict_mean": float(D["world_conflict"][i].mean()), "dem_2065": float(D["world_dem"][i, -1]),
                               "params": {k: float(v[i]) for k, v in params.items()},
                               "param_pct": {k: round(float((v < v[i]).mean()), 3) for k, v in params.items()},
                               "events": logs[i]}
    k_ = os.path.join(OUTD, "knockouts.json")
    if os.path.exists(k_):
        res["knockouts"] = json.load(open(k_))
    json.dump(res, open(os.path.join(OUTD, "rca.json"), "w"), indent=1, default=float)
    # print a summary
    print(json.dumps(res["outcomes_all"], indent=0))
    for name, r in res["sets"].items():
        print(f"\n== {name} (worst {r['n']}): {json.dumps({k: round(v, 3) for k, v in r['outcomes_disaster'].items()})}")
        print("  assumptions (AUC, null 95%", round(r["auc_null_95"] + 0.5, 3), "):",
              ", ".join(f"{a['name']} {a['auc']}" for a in r["assumptions"][:6]))
        print("  events:", ", ".join(f"{e['event']} {e['share_disaster']:.2f}/{e['share_all']:.2f}" for e in r["events"]))
        print("  first shock:", r["first_shock"])
    print("\neconomic decomposition", res["economic_decomposition"])
    print("by country", [(c["name"], c["contribution_usd"]) for c in res["economic_by_country"]["countries"]])
    print("civil war by country", res["civil_war_by_country"], res["civil_war_breadth"])
    if "knockouts" in res:
        for n_, v in res["knockouts"]["runs"].items():
            print(f"  {n_:24s}", {k: round(x["tail_share"] / TAIL, 2) for k, x in v.items()})


if __name__ == "__main__":
    {"sim": cmd_sim, "knock": cmd_knock, "analyze": cmd_analyze}[sys.argv[1]]()
