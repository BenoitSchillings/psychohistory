"""End-to-end steps: backtest, hindcast, forecast. Each writes into outputs/."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .backtest import calibration, flatten, run_backtest, summarize, train_mask
from .data import LAST_OBS_YEAR, ROOT, get_panel
from .features import FEATURES, GROUPS, TASKS, base_arrays, labels, panel_features
from .backtest import ALL_FAMILIES, DISPLAY
from .models import DEVICE, NET_TUNED, NaiveModel, PsychoNet, SeldonEnsemble, load_combine
from .simulate import fit_transitions, simulate

OUT = ROOT / "outputs"
HEADLINE = ["civil_conflict_5y", "econ_collapse_5y", "autocratization_5y", "coup_5y"]


def prepare(refresh=False):
    p = get_panel(refresh)
    base = base_arrays(p)
    X = panel_features(p, base, DEVICE)
    return p, base, X, labels(p, base)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------

def step_backtest(p, X, Yl, origins=range(1980, 2016, 5), members=8):
    _log("rolling-origin backtest")
    bt = run_backtest(p, X, Yl, origins=origins, members=members)
    bt.to_parquet(OUT / "backtest_predictions.parquet")
    summ = summarize(bt)
    summ.to_csv(OUT / "backtest_summary.csv", index=False)
    from .backtest import summarize_split
    summarize_split(bt).to_csv(OUT / "backtest_split.csv", index=False)
    cal = pd.concat([calibration(bt, t).assign(task=t) for t in HEADLINE])
    cal["bin"] = cal["bin"].astype(str)
    cal.to_csv(OUT / "calibration.csv", index=False)
    return summ, cal, bt


def _hindcast_metrics(p, base, out, origin, end):
    conf = base["conflict"][:, p.yi(origin) + 1:p.yi(end) + 1]
    ok = np.isfinite(conf)
    g = np.exp(base["lg"][:, p.yi(end)])
    q = out["gdp_q"][:, -1]
    okg = np.isfinite(g) & np.isfinite(q).all(1)
    d = base["dem"][:, p.yi(end)]
    qd = out["dem_q"][:, -1]
    okd = np.isfinite(d) & np.isfinite(qd).all(1)
    act = np.nanmean(conf, 0)
    wq = np.quantile(out["world_conflict"], [.05, .95], 0)
    # new onsets only, among countries at peace at the origin (persistence of ongoing wars excluded)
    c0 = base["conflict"][:, p.yi(origin)]
    full = base["conflict"][:, p.yi(origin):p.yi(end) + 1]
    onset_any = ((full[:, 1:] == 1) & (full[:, :-1] == 0)).any(1)
    peace = c0 == 0
    return {
        "origin": origin, "end": end,
        "conflict_incidence_auc": float(roc_auc_score(conf[ok], out["conflict"][ok])),
        "onset_auc_peace": float(roc_auc_score(onset_any[peace], out["ever_onset"][peace, -1])),
        "world_conflict_in_90": float(((act >= wq[0]) & (act <= wq[1])).mean()),
        "gdp_in_90": float(((g >= q[:, 0]) & (g <= q[:, 4]))[okg].mean()),
        "gdp_in_50": float(((g >= q[:, 1]) & (g <= q[:, 3]))[okg].mean()),
        "dem_in_90": float(((d >= qd[:, 0]) & (d <= qd[:, 4]))[okd].mean()),
        "dem_in_50": float(((d >= qd[:, 1]) & (d <= qd[:, 3]))[okd].mean()),
    }


def step_hindcast(p, base, X, Yl, origins=(1975, 1985, 1995, 2004, 2015), horizon=20, n_scen=4000):
    """simulate from past origins with models trained only on data before them."""
    _log("simulator hindcasts")
    Xf, Yf, ci, year = flatten(p, X, Yl)
    rows, fans = [], {}
    for O in origins:
        E = min(O + horizon, LAST_OBS_YEAR)
        Ytr, tr = train_mask(Yf, year, O)
        net = PsychoNet(members=8, **NET_TUNED).fit(Xf[tr], Ytr[tr], ci[tr])
        growth, ddem = fit_transitions(p, X, base, O)
        # hindcasts must not see the realised future: demography is extrapolated from the origin
        out = simulate(p, base, net, growth, ddem, O, E, n_scen=n_scen, demography="extrapolate")
        rows.append(_hindcast_metrics(p, base, out, O, E))
        fans[O] = {"years": out["years"].tolist(),
                   "world_conflict_q": np.quantile(out["world_conflict"], [.05, .25, .5, .75, .95], 0).round(4).tolist(),
                   "world_conflict_actual": np.nanmean(base["conflict"][:, p.yi(O) + 1:p.yi(E) + 1], 0).round(4).tolist()}
        _log(f"  hindcast {O}->{E}: {rows[-1]}")
    pd.DataFrame(rows).to_csv(OUT / "hindcast_summary.csv", index=False)
    return pd.DataFrame(rows), fans


# ---------------------------------------------------------------------------

def _drivers(logit, X_now, task_k, X_train, top=3):
    """per-country contributions of feature groups to the logit's log-odds vs. the average country
    (training mean). Missing-data indicators are reported as their own group."""
    pre = logit.pre
    Z = np.nan_to_num(np.clip((X_now - pre.mu) / pre.sd, -6, 6))
    coef_all = logit.models[task_k].coef_[0]
    coef, coef_miss = coef_all[:len(FEATURES)], coef_all[len(FEATURES):]
    contrib = Z * coef
    miss = (np.isnan(X_now) - np.isnan(X_train).mean(0)) @ coef_miss
    out = []
    for c in range(len(X_now)):
        g = {name: float(sum(contrib[c, FEATURES.index(f)] for f in feats)) for name, feats in GROUPS.items()}
        g["Missing data"] = float(miss[c])
        feats = sorted(((FEATURES[j], float(contrib[c, j])) for j in range(len(FEATURES))), key=lambda kv: -kv[1])
        out.append({"groups": g, "top_up": [f for f, v in feats[:top] if v > 0]})
    return out


CLIMATE_YEARS = (2035, 2050, 2065)


def climate_summary(p, base, origin, end):
    """per-country change in climate drivers by scenario (CMIP6 median, delta-corrected)."""
    from . import climate
    from .simulate import climate_inputs
    T = end - origin
    out = {}
    for s in climate.SCENARIOS:
        cl = climate_inputs(p, base, origin, T, s)
        d = {}
        for y in CLIMATE_YEARS:
            k = y - origin
            d[y] = {"tas": cl["tas"]["median"][:, k] - cl["tas"]["base"],
                    "tas_p10": cl["tas"]["p10"][:, k] - cl["tas"]["base"],
                    "tas_p90": cl["tas"]["p90"][:, k] - cl["tas"]["base"],
                    "txx": cl["txx"]["median"][:, k] - cl["txx"]["base"],
                    "pr_pct": 100 * (cl["pr"]["median"][:, k] / cl["pr"]["base"] - 1),
                    "cdd": cl["cdd"]["median"][:, k] - cl["cdd"]["base"]}
        out[s] = {"inputs": cl, "delta": d}
    return out


def struct_inputs(ens, X_now):
    """structural-uncertainty inputs for simulate (STRUCT_MODE) from a fitted SeldonEnsemble at the origin's features:
    struct_sd [C, 3] the families' log-odds spread; struct_fam [F, C, 3] each family's log-odds offset from PsychoNet;
    struct_w [F] the ensemble's default weights for these one-year tasks (onset, termination, coup)"""
    from .models import SeldonEnsemble
    parts = ens.predict_families(X_now)
    idx3 = [TASKS.index(t) for t in ("onset_1y", "termination_1y", "coup_1y")]
    lgf = lambda v: np.log(np.clip(v, 1e-4, 1 - 1e-4) / (1 - np.clip(v, 1e-4, 1 - 1e-4)))
    fams = [f for f in SeldonEnsemble.DEFAULT["w"] if f in parts]
    w = np.array([SeldonEnsemble.DEFAULT["w"][f] for f in fams], float)
    fam_l = np.stack([lgf(parts[f]) for f in fams])
    sd = np.nan_to_num(np.nanstd(np.stack([lgf(v) for v in parts.values()]), 0)[:, idx3])
    return sd, np.nan_to_num(fam_l[:, :, idx3] - lgf(parts["N"])[None, :, idx3]), (w / w.sum()).astype(np.float32)


def ensemble_offset(ens_p3, net, X_now):
    """[C, 3] log-odds of the ensemble's one-year probabilities (onset, termination, coup) minus the network's, at the
    origin's features (clipped to +/-3); attached to the simulator's network as net.haz_offset"""
    idx3 = [TASKS.index(t) for t in ("onset_1y", "termination_1y", "coup_1y")]
    lgf = lambda v: np.log(np.clip(v, 1e-4, 1 - 1e-4) / (1 - np.clip(v, 1e-4, 1 - 1e-4)))
    return np.nan_to_num(np.clip(lgf(ens_p3) - lgf(net.predict(X_now)[:, idx3]), -3, 3))


def attach_ensemble(net, ens, X_now):
    """the simulator's network gets the structural-uncertainty inputs and the ensemble's central offset
    (simulate.ENS_OFFSET: the default four families, equal-weight probabilities, as the onset and coup tasks combine)"""
    net.struct_sd, net.struct_fam, net.struct_w = struct_inputs(ens, X_now)
    parts = ens.predict_families(X_now)
    idx3 = [TASKS.index(t) for t in ("onset_1y", "termination_1y", "coup_1y")]
    fams = [f for f in ("G", "G60", "N", "F") if f in parts]
    net.haz_offset = ensemble_offset(np.mean([parts[f][:, idx3] for f in fams], 0), net, X_now)
    return net


def attach_cached(net, fams, p3, X_now):
    """attach_ensemble from cached family probabilities (scripts/ens_cache.py: fams [F], p3 [F, C, 3] one-year onset,
    termination, coup): the same central offset, and the structural draw's family offsets taken against this network
    (the centred draw, STRUCT_MODE familyc, does not depend on the reference)"""
    from .models import SeldonEnsemble
    idx3 = [TASKS.index(t) for t in ("onset_1y", "termination_1y", "coup_1y")]
    lgf = lambda v: np.log(np.clip(v, 1e-4, 1 - 1e-4) / (1 - np.clip(v, 1e-4, 1 - 1e-4)))
    fams = list(fams)
    use = [f for f in SeldonEnsemble.DEFAULT["w"] if f in fams]
    w = np.array([SeldonEnsemble.DEFAULT["w"][f] for f in use], float)
    P = np.stack([p3[fams.index(f)] for f in use])
    pn = net.predict(X_now)[:, idx3]
    net.struct_fam = np.nan_to_num(lgf(P) - lgf(pn)[None])
    net.struct_w = (w / w.sum()).astype(np.float32)
    net.struct_sd = np.nan_to_num(np.nanstd(lgf(P), 0))
    net.haz_offset = ensemble_offset(P.mean(0), net, X_now)
    return net

def sim_net(ens, X_now):
    """the simulator's network as in the forecast: a shallow copy of the ensemble's first PsychoNet with every seed's
    members pooled (the ensemble's own members keep their nets), with attach_ensemble's inputs at the origin"""
    import copy
    net = copy.copy(ens.fam["N"][0])
    net.nets = [n for m in ens.fam["N"] for n in m.nets]
    net.ens_models = {f: ens.fam[f] for f in ("G", "G60", "N", "F") if f in ens.fam}   # (simulate.ENS_REFRESH)
    # what the simulator was built from (recorded in simulate's output config; scripts/config_parity.py)
    net.env = {"members": int(ens.members), "ens_seed": int(getattr(ens, "seed", 0)), "n_nets": len(net.nets),
               "families": sorted(f for f in ens.fam if f in ("G", "G60", "N", "F")),
               "first_year": int(min(ens.first_year.values()))}
    return attach_ensemble(net, ens, X_now)


# the simulator's network as the forecast builds it (step_forecast): the risk ensemble fitted on every row from 1860
# (each family on its own window), the PsychoNet members of both seeds pooled, the family draw's inputs attached.
# Every harness builds it with build_sim_env, so tests run the forecast's configuration (debate 2026-11-05: the
# 10-year harness ran a stand-alone 8-member network without the family draw, the full-system cache a 3-member ensemble)
SIM_ENV = {"members": 10, "families": ("G", "G60", "N", "F"), "first_year": 1860}


def align_delta(net, p, X, base, origin, window=10):
    """[3] log-odds to add to the year-1 alignment shift (onset, ending, coups) so that it is solved on the previous
    decade's own yearly risk sets instead of the origin's (simulate.ALIGN_MATCHED): the network's shift matching the
    decade's rate on that decade's rows, minus the one matching it on the origin's risk set. Data before the origin only.
    scripts/align_oracle.py: closer to the next decade's right shift for war endings in 3 of 4 decades, not for onsets"""
    from .features import TASKS
    cf = base["conflict"]
    lg = lambda v: np.log(np.clip(v, 1e-5, 1 - 1e-5) / (1 - np.clip(v, 1e-5, 1 - 1e-5)))

    def solve(z, target):
        lo, hi = -4.0, 4.0
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if np.mean(1 / (1 + np.exp(-(z + mid)))) < target else (lo, mid)
        return (lo + hi) / 2
    out = np.zeros(3)
    t0 = p.yi(origin)
    for j, (task, at_war) in enumerate((("onset_1y", False), ("termination_1y", True))):
        k = TASKS.index(task)
        zs, ys = [], []
        for y in range(origin - window, origin):
            t = p.yi(y)
            now, nxt = cf[:, t], cf[:, t + 1]
            risk = np.isfinite(now) & np.isfinite(nxt) & ((now == 1) if at_war else (now == 0))
            pr = net.predict(X[:, t])[:, k]
            ok = risk & np.isfinite(pr)
            zs.append(lg(pr[ok]))
            ys.append((nxt[ok] == 0) if at_war else (nxt[ok] == 1))
        zp, yp = np.concatenate(zs), np.concatenate(ys).astype(float)
        now = cf[:, t0]
        risk0 = np.isfinite(now) & ((now == 1) if at_war else (now == 0))
        z0 = lg(net.predict(X[:, t0])[:, k])[risk0 & np.isfinite(net.predict(X[:, t0])[:, k])]
        if len(zp) and len(z0) and 0 < yp.mean() < 1:
            out[j] = solve(zp, yp.mean()) - solve(z0, yp.mean())
    return out


def build_sim_env(p, X, Xf, Yf, ci, year, origin, members=None, seed=0, cache=True, base=None):
    """the forecast's simulator network at a past origin, trained only on data before it; cached on disk
    (outputs/sim_env/<origin>_m<members>_s<seed>.pkl). seed offsets every family's seeds (0 = the forecast's)."""
    import pickle
    from .backtest import train_mask
    from .models import SeldonEnsemble, load_combine
    members = members or SIM_ENV["members"]
    d = OUT / "sim_env"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{origin}_m{members}_s{seed}.pkl"
    if cache and f.exists():
        with open(f, "rb") as fh:
            net = pickle.load(fh)
        if base is not None and getattr(net, "align_delta", None) is None:
            net.align_delta = align_delta(net, p, X, base, origin)
        return net
    Ytr, tr = train_mask(Yf, year, origin, first_year=SIM_ENV["first_year"])
    ens = SeldonEnsemble(load_combine(), members=members, families=list(SIM_ENV["families"]), seed=seed).fit(
        Xf[tr], Ytr[tr], ci[tr], year[tr])
    net = sim_net(ens, X[:, p.yi(origin)])
    net.env["origin"] = int(origin)
    net.fam_p1 = {k: v for k, v in ens.predict_families(X[:, p.yi(origin)]).items()}   # [C, tasks] per family
    net.align_delta = align_delta(net, p, X, base, origin) if base is not None else None
    if cache:
        refresh, net.ens_models = net.ens_models, None        # the tree families are large; ENS_REFRESH is off
        with open(f, "wb") as fh:
            pickle.dump(net, fh)
        net.ens_models = refresh
    return net


def step_forecast(p, base, X, Yl, origin=LAST_OBS_YEAR, end=2065, n_scen=10000, members=10,
                  scenario="ssp245", climate_damage=1.0, scen_n=4000):
    _log(f"training on all history up to {origin}")
    Xf, Yf, ci, year = flatten(p, X, Yl)
    Ytr, tr = train_mask(Yf, year, origin, first_year=1860)
    ens = SeldonEnsemble(load_combine(), members=members, families=ALL_FAMILIES).fit(Xf[tr], Ytr[tr], ci[tr], year[tr])
    tr = tr & (year >= 1955)                      # post-war rows: base rates, naive model, driver reference
    naive = NaiveModel().fit(Xf[tr], Ytr[tr])
    X_now = X[:, p.yi(origin)]
    parts = ens.predict_families(X_now)
    risk = ens.predict(X_now)
    # uncertainty = disagreement between the model families (bracketing the combined estimate)
    fam = np.stack(list(parts.values()) + [risk])
    lo, hi = np.nanmin(fam, 0), np.nanmax(fam, 0)
    logit = ens.fam["L"][0]
    drivers = {t: _drivers(logit, X_now, TASKS.index(t), Xf[tr]) for t in HEADLINE}
    base_rate = np.nanmean(Ytr[tr], 0)

    growth, ddem = fit_transitions(p, X, base, origin)
    # the simulator's hazards come from the tuned PsychoNet with the members of every seed pooled (all seeds are fitted
    # on the same rows, so they share one input standardisation); structural uncertainty: each
    # world draws which model family is right (simulate.STRUCT_MODE = "familyc"), around the ensemble's central view
    # of onsets and coups (simulate.ENS_OFFSET, 2026-11-03)
    net = sim_net(ens, X_now)
    net.align_delta = align_delta(net, p, X, base, origin)          # (used only if simulate.ALIGN_MATCHED)
    clim = climate_summary(p, base, origin, end)
    # world-system layer: interstate wars between country pairs, food, energy, pandemics (AI off in the baseline)
    from . import interstate as INTER
    from .world import AI_SCENARIOS, WorldConfig
    _log(f"interstate module: country pairs, backtests, world-war retrodictions, {origin + 1}-{origin + 5} risks")
    r_int = INTER.run(verbose=False)
    dy = INTER.sim_inputs(r_int)
    # the headline world: near-free cognition arrives in a year each world draws from ARRIVAL_CDF (or not by 2065)
    wbase = WorldConfig(ai=AI_SCENARIOS["uncertain"], sample_params=True)
    _log(f"simulating {n_scen} worlds {origin}->{end} under {scenario}, climate damage x{climate_damage}, world system on")
    sim = simulate(p, base, net, growth, ddem, origin, end, n_scen=n_scen, clim=clim[scenario]["inputs"],
                   climate_damage=climate_damage, world=wbase, dyads=dy)
    # the configuration the forecast simulated: the reference for scripts/config_parity.py
    (OUT / "sim_config_forecast.json").write_text(json.dumps(sim["config"], indent=1, default=str))
    # the same world under every climate scenario, with and without economic damage
    q = lambda a: np.quantile(a, [.05, .25, .5, .75, .95], 0).round(4).tolist()
    scen_out = {}
    for s_ in clim:
        for dmg in (climate_damage, 0.0):
            key = f"{s_}{'' if dmg else '_nodamage'}"
            if dmg == 0.0 and s_ not in (scenario, "ssp585"):
                continue
            _log(f"  climate scenario {key}")
            o = simulate(p, base, net, growth, ddem, origin, end, n_scen=scen_n, clim=clim[s_]["inputs"],
                         climate_damage=dmg, seed=1, world=wbase, dyads=dy)
            scen_out[key] = {"world_conflict_q": q(o["world_conflict"]), "world_gdp_q": q(o["world_gdp"]),
                             "world_dem_q": q(o["world_dem"]),
                             "gdp_end_median": np.round(o["gdp_q"][:, -1, 2]), "conflict_end": o["conflict"][:, -1].round(3)}

    # AI scenarios (near-free cognition) and a 1914-style alliance stress test, against the same baseline
    def attribution(o):
        """share of the variance of each 2065 outcome explained by each sampled assumption. The assumptions are drawn
        independently (except as noted below), so each share is the R^2 of the outcome on that assumption alone; the AI arrival year enters
        through a flexible basis (years of AI before the end, its square, and "no arrival"), since its effect on
        2065 outcomes is far from linear. The robotics arrival year is drawn correlated with the AI arrival year and
        capped by it, so its share is what it adds beyond the AI year and robot_lag (incremental R^2)"""
        w = o["world"]
        P = w.get("params")
        if not P:
            return None
        end_y = int(o["years"][-1])

        def basis(k):
            v = np.asarray(P[k], float)
            if k in ("ai_arrival", "robot_arrival"):
                yrs = np.clip(end_y + 0.5 - v, 0, None)
                return np.c_[yrs, yrs ** 2, (v > end_y + 0.5).astype(float)]
            return v[:, None]
        names = [k for k in P if np.std(P[k]) > 0]
        Y = {"world_gdp_2065": np.log(o["world_gdp"][:, -1]), "conflict_2065": o["world_conflict"][:, -1],
             "great_power_war": w["great_power_war"].max(1).astype(float), "oil_2065": w["oil"][:, -1] if "oil" in w else None,
             "democracy_2065": o["world_dem"][:, -1]}
        if "nuclear_use" in w:
            Y["nuclear_use"] = w["nuclear_use"].max(1).astype(float)
        # war (>= 1,000 battle deaths a year) by the end in the pairs with live water disputes
        wn, big = w.get("watch_names", []), w.get("watch_war_deaths")
        for nm in ("India-Pakistan", "China-India", "Egypt-Ethiopia"):
            if big is not None and nm in wn:
                Y[f"war_{nm}"] = (big[:, wn.index(nm)] > 0).astype(float)
        out = {}
        for k, y in Y.items():
            if y is None or np.var(y) == 0:
                continue
            sh = {}

            def r2(*ks):
                Xb = np.c_[np.ones(len(y)), *[basis(k_) for k_ in ks]]
                return 1 - np.var(y - Xb @ np.linalg.lstsq(Xb, y, rcond=None)[0]) / np.var(y)
            for n_ in names:
                sh[n_] = max(0.0, r2(n_))
            # robot_arrival = min(own draw, AI year + robot_lag): credit it only beyond its upstream assumptions
            up = [k_ for k_ in ("ai_arrival", "arrival_sd", "robot_lag") if k_ in sh]
            if "robot_arrival" in sh and up:
                sh["robot_arrival"] = max(0.0, r2(*up, "robot_arrival") - r2(*up))
            # technology arrival years share a science-pace factor and are pulled forward by the AI year (tech.py): credit
            # each only beyond the AI arrival, so the shares do not count AI timing again (review 2026-11-02)
            if "ai_arrival" in sh:
                for n_ in [n_ for n_ in sh if n_.startswith("tech_")]:
                    sh[n_] = max(0.0, r2("ai_arrival", n_) - r2("ai_arrival"))
            out[k] = {**{n_: round(float(v), 4) for n_, v in sh.items()}, "other": round(float(max(0, 1 - sum(sh.values()))), 4)}
        return out

    from .food import fit_world_price
    food_p0 = fit_world_price(origin)["p0"]          # world real grain price index at the origin (= 100)

    def war_scale(o):
        """war-level (>= 1,000 battle deaths a year) probabilities next to any-clash ones, and battle deaths
        (psychohistory/war_size.py)"""
        w = o["world"]
        if "battle_deaths" not in w:
            return {}
        names, y0 = w.get("watch_names", []), int(o["years"][0])
        cum = w["battle_deaths"].sum(-1)
        big = w.get("watch_war_deaths")
        thr = {"100k": 1e5, "1M": 1e6, "10M": 1e7}
        return {
            "watch_war": {n: {str(y0 + int(k) - 1): float(v[i]) for k, v in w.get("watch_war_hits", {}).items()}
                          for i, n in enumerate(names)},
            "watch_bigwar": {n: {t: float((big[:, i] > x).mean()) for t, x in thr.items()} for i, n in enumerate(names)}
            if big is not None else {},
            "blocs_war": w.get("blocs_war", {}),
            "battle_deaths_q": q(w["battle_deaths"]),
            "battle_deaths_total": {"q": np.quantile(cum, [.05, .25, .5, .75, .95]).round(0).tolist(), "mean": float(cum.mean()),
                                    "p_over": {t: float((cum > x).mean()) for t, x in thr.items()}},
            "largest_war_p_over": {t: float((w["max_war_deaths"] > x).mean()) for t, x in thr.items()},
            "major_war_deaths_total_mean": float(w["war_deaths_major"].sum(-1).mean()),
        }

    def world_summary(o):
        w = o["world"]
        names = w.get("watch_names", [])
        hits = w.get("watch_hits", {})
        return {"world_conflict_q": q(o["world_conflict"]), "world_gdp_q": q(o["world_gdp"]),
                "world_dem_q": q(o["world_dem"]), "coups_q": q(o["coups"]), "active_wars_q": q(w["active"]),
                "war_level_q": q(w["wars"]),
                "great_power_war_by": np.maximum.accumulate(w["great_power_war"], axis=1).mean(0).round(4).tolist(),
                "great_power_war_cow_by": np.maximum.accumulate(w["great_power_war_cow"], axis=1).mean(0).round(4).tolist()
                if "great_power_war_cow" in w else None,
                "pandemics_mean": float(w["pandemic"].sum(-1).mean()),
                "clean_share_q": q(w["clean_share"]) if "clean_share" in w else None,
                "oil_price_q": q(100 * np.exp(w["oil"] - base["oil_price"][0][p.yi(origin)])) if "oil" in w else None,  # real, origin year = 100
                "ai_adoption": w["ai_adoption_world"].mean(0).round(4).tolist(),
                "robot_adoption": w["robot_adoption_world"].mean(0).round(4).tolist() if "robot_adoption_world" in w else None,
                "mil_autonomy": w["mil_autonomy"].mean(0).round(4).tolist() if "mil_autonomy" in w else None,
                "watch": {n: {str(int(o["years"][0]) + int(k) - 1): float(v[i]) for k, v in hits.items()}
                          for i, n in enumerate(names)},
                "blocs": w.get("blocs", {}), "gdp_end_median": np.round(o["gdp_q"][:, -1, 2]), "attribution": attribution(o),
                **war_scale(o),
                "conflict_end": o["conflict"][:, -1].round(3),
                "displaced_q": q(o["displaced"]) if "displaced" in o else None,
                # water supply vs the origin (glacier runoff and precipitation; median world), countries that move
                "water_change": {c: [round(float(w["water_change"][i, y_ - int(o["years"][0])]), 4) for y_ in (2035, 2050, 2065)
                                     if y_ - int(o["years"][0]) < w["water_change"].shape[1]]
                                 for i, c in enumerate(p.codes) if abs(w["water_change"][i, -1]) > 0.01}
                if "water_change" in w else None,
                "food_price_q": q(100 * np.exp(w["food_price"] - food_p0)) if "food_price" in w else None,
                "risks": risk_summary(w)}

    def risk_summary(w):
        """the 2026-10-26 additions (risks.py): nuclear use and spread, population, catastrophes"""
        if "nuclear_use" not in w:
            return None
        qq = lambda a: np.quantile(a, [.05, .25, .5, .75, .95]).round(4).tolist()
        nd = w["nuclear_deaths"].sum(-1)
        ne = w.get("nuclear_end")
        return {"nuclear_use_p": float(w["nuclear_use"].max(1).mean()),
                "nuclear_use_by": np.maximum.accumulate(w["nuclear_use"], axis=1).mean(0).round(4).tolist(),
                "nuclear_large_p": float(w["nuclear_large"].max(1).mean()),
                "nuclear_deaths_mean": float(nd.mean()),
                "nuclear_deaths_given_use_q": qq(nd[nd > 0]) if (nd > 0).any() else None,
                "new_nuclear_p": float((w["nuclear_states"][:, -1] > 0).mean()) if "nuclear_states" in w else None,
                "new_nuclear_by_country": {c: round(float(v), 4) for c, v in zip(w.get("codes", []), ne) if v > 0.002}
                if ne is not None else {},
                "pop_dev_q": qq(w["pop_dev"][:, -1]) if "pop_dev" in w else None,
                "excess_deaths_q": qq(w["excess_deaths"][:, -1]) if "excess_deaths" in w else None,
                "solar_p": float(w["solar"].max(1).mean()) if "solar" in w else None,
                "ai_catastrophe_p": float(w["ai_catastrophe"].max(1).mean()) if "ai_catastrophe" in w else None,
                "volcano_p": float(w["volcano"].max(1).mean()) if "volcano" in w else None,
                "sanction_country_years_mean": float(w["sanctioned"].sum(-1).mean()) if "sanctioned" in w else None}

    ai_out = {"uncertain": world_summary(sim)}
    arr = sim["world"]["params"].get("ai_arrival")
    if arr is not None:
        arr = np.asarray(arr).ravel()
        # "by year Y" = by mid-year Y + 0.5, as ARRIVAL_CDF's knots are placed
        ai_out["uncertain"]["arrival_share_by"] = {str(y): round(float((arr <= y + 0.5).mean()), 4) for y in (2027, 2030, 2035, 2040, 2050, 2065)}
    rob = sim["world"]["params"].get("robot_arrival")
    if rob is not None:
        rob = np.asarray(rob).ravel()
        ai_out["uncertain"]["robot_share_by"] = {str(y): round(float((rob <= y + 0.5).mean()), 4) for y in (2027, 2030, 2035, 2040, 2050, 2065)}
    for name in ("none", "2027", "2030", "2035"):
        _log(f"  AI scenario: {AI_SCENARIOS[name].name}")
        o = simulate(p, base, net, growth, ddem, origin, end, n_scen=scen_n, clim=clim[scenario]["inputs"],
                     climate_damage=climate_damage, seed=2, world=WorldConfig(ai=AI_SCENARIOS[name], sample_params=True), dyads=dy)
        ai_out[name] = world_summary(o)
    _log("  stress test: 1914-style alliance cascades")
    o = simulate(p, base, net, growth, ddem, origin, end, n_scen=scen_n, clim=clim[scenario]["inputs"],
                 climate_damage=climate_damage, seed=2, world=WorldConfig(cascade_q=0.5, sample_params=True), dyads=dy)
    ai_out["alliances_1914"] = world_summary(o)
    # the alliance rule before treaty scope (2026-11-05): a permanent Security Council member joins any ally's war at
    # the full rate, wherever it is (France in a US-China war over Taiwan as often as in a Russian attack on Estonia)
    _log("  scenario: alliances without treaty scope")
    o = simulate(p, base, net, growth, ddem, origin, end, n_scen=scen_n, clim=clim[scenario]["inputs"],
                 climate_damage=climate_damage, seed=2,
                 world=WorldConfig(ai=AI_SCENARIOS["uncertain"], sample_params=True, ally_out_scope=1.0), dyads=dy)
    ai_out["alliances_any_scope"] = world_summary(o)
    # era scenario: major powers fight as often, given their circumstances, as they did in 1816-1938 (scripts/era_test.py)
    _log("  scenario: great powers in the pre-1945 era")
    o = simulate(p, base, net, growth, ddem, origin, end, n_scen=scen_n, clim=clim[scenario]["inputs"],
                 climate_damage=climate_damage, seed=2,
                 world=WorldConfig(ai=AI_SCENARIOS["uncertain"], sample_params=True, major_era_mult=1.3), dyads=dy)
    ai_out["era_pre1945"] = world_summary(o)
    # labelled scenario (not the headline): wars between states last as long as in the last decade. The simulator
    # missed the 2010s rise in active interstate conflicts two- to threefold, mostly in how long they last; fitting it
    # would use the outcomes that should test it (docs/adopt_rule.md), so it is shown beside the headline instead:
    # the continuation table weighted with a 5-year half-life (25 in the headline)
    _log("  scenario: interstate wars as long as in the 2010s")
    hl0 = INTER.CONT_HALFLIFE
    try:
        INTER.CONT_HALFLIFE = 5
        dy10 = INTER.sim_inputs(INTER.run(verbose=False, origin=origin, backtests=False))
    finally:
        INTER.CONT_HALFLIFE = hl0
    o = simulate(p, base, net, growth, ddem, origin, end, n_scen=scen_n, clim=clim[scenario]["inputs"],
                 climate_damage=climate_damage, seed=2, world=WorldConfig(ai=AI_SCENARIOS["uncertain"], sample_params=True),
                 dyads=dy10)
    ai_out["interstate_2010s"] = world_summary(o)
    # water and glaciers (assumptions): the same worlds with the effects off, at their sampled values, and with
    # disputes, scarcity and the yield response at the top of their ranges (the Indus stress test)
    for key, cfg_ in (("water_off", dict(water=False)), ("water_ref", {}),
                      ("water_high", dict(water_dispute=0.5, water_scarcity_k=4.0, yield_water=0.7))):
        _log(f"  water scenario: {key}")
        o = simulate(p, base, net, growth, ddem, origin, end, n_scen=scen_n, clim=clim[scenario]["inputs"],
                     climate_damage=climate_damage, seed=4,
                     world=WorldConfig(ai=AI_SCENARIOS["uncertain"], sample_params=True, **cfg_), dyads=dy)
        ai_out[key] = world_summary(o)
    chron = chronicle_worlds(p, base, net, growth, ddem, origin, end, clim[scenario]["inputs"], climate_damage, dy)
    (OUT / "chronicles_raw.json").write_text(json.dumps(chron, default=lambda o: None))
    explorer = crisis_explorer(p, base, net, growth, ddem, origin, end, scen_n, clim[scenario]["inputs"], climate_damage, dy)
    interstate_out = {
        "backtest": r_int["backtest"], "world_wars": r_int["world_wars"], "continuation": r_int["continuation"],
        "n_pairs": r_int["n_pairs"], "base_rate": r_int["base_rate"], "blocs_direct": r_int["blocs"],
        "top": r_int["top"].to_dict("records"), "watch": r_int["watch"].to_dict("records"),
        "cascade_q": wbase.cascade_q, "war_share": wbase.war_share}
    (OUT / "crisis_explorer.json").write_text(json.dumps(explorer, default=lambda o: None))

    years = sim["years"]
    yi = lambda y: int(y - years[0])
    countries = []
    for c, code in enumerate(p.codes):
        if not np.isfinite(X_now[c, FEATURES.index("log_pop")]):
            continue
        row = {"code": code, "name": p.names[c], "region": p.region_names[p.regions[c]],
               "pop_m": float(np.exp(base["lpop"][c, p.yi(origin)]) / 1e6),
               "in_conflict": int(np.nan_to_num(base["conflict"][c, p.yi(origin)])),
               "gdp_pc": float(np.exp(base["lg"][c, p.yi(origin)])) if np.isfinite(base["lg"][c, p.yi(origin)]) else None,
               "democracy": float(base["dem"][c, p.yi(origin)]) if np.isfinite(base["dem"][c, p.yi(origin)]) else None}
        for t in HEADLINE:
            k = TASKS.index(t)
            row[t] = None if not np.isfinite(risk[c, k]) else float(risk[c, k])
            if t == "civil_conflict_5y" and row["in_conflict"]:
                row[t] = None   # onset risk is only defined for countries at peace
            row[t + "_lo"] = float(lo[c, k])
            row[t + "_hi"] = float(hi[c, k])
            row[t + "_drivers"] = drivers[t][c]["top_up"]
            row[t + "_groups"] = {g: round(v, 3) for g, v in drivers[t][c]["groups"].items()}
        for y in sorted({origin + 5, 2030, 2035, 2050, 2065}):
            if y <= end:
                row[f"p_conflict_{y}"] = float(sim["conflict"][c, yi(y)])
                row[f"p_onset_by_{y}"] = float(sim["ever_onset"][c, yi(y)])
                row[f"p_coup_by_{y}"] = float(sim["ever_coup"][c, yi(y)])
                row[f"p_collapse_by_{y}"] = float(sim["ever_collapse"][c, yi(y)])
                row[f"p_autoc_by_{y}"] = float(sim["ever_autoc"][c, yi(y)])
                row[f"gdp_{y}_q"] = [None if not np.isfinite(v) else round(float(v)) for v in sim["gdp_q"][c, yi(y)]]
                row[f"dem_{y}_q"] = [None if not np.isfinite(v) else round(float(v), 3) for v in sim["dem_q"][c, yi(y)]]
        row["conflict_path"] = sim["conflict"][c].round(3).tolist()
        h0 = p.yi(1990)
        row["hist"] = {"gdp": np.exp(base["lg"][c, h0:p.yi(origin) + 1]).round(),
                       "dem": base["dem"][c, h0:p.yi(origin) + 1].round(3),
                       "conflict": base["conflict"][c, h0:p.yi(origin) + 1]}
        row["climate"] = {s_: {str(y): {k: round(float(v[c]), 2) for k, v in clim[s_]["delta"][y].items()}
                               for y in CLIMATE_YEARS} for s_ in clim}
        row["tas_base"] = round(float(clim[scenario]["inputs"]["tas"]["base"][c]), 2)
        row["scenarios"] = {k: {"gdp_end": v["gdp_end_median"][c], "conflict_end": float(v["conflict_end"][c])}
                            for k, v in scen_out.items()}
        row["ai"] = {k: {"gdp_end": v["gdp_end_median"][c], "conflict_end": float(v["conflict_end"][c])}
                     for k, v in ai_out.items()}
        row["gdp_path_q"] = sim["gdp_q"][c].round()      # NaN (no data) -> null in JSON
        row["dem_path_q"] = sim["dem_q"][c].round(3)
        idep = sim["world"].get("import_dep")
        dep_obs = p.x.get("feat__cereal_import_dep")
        seen = np.where(np.isfinite(dep_obs[c, :p.yi(origin) + 1]))[0] if dep_obs is not None else []
        if idep is not None and idep.ndim == 2 and len(seen) and p.years[seen[-1]] >= origin - 15:
            # countries with no food balance in 15 years start from their region's median: not shown
            row["cereal_import_dep"] = {str(y): round(float(idep[c, yi(y)]), 3) for y in (origin + 1, 2035, 2050, end)}
        countries.append(row)

    q = lambda a: np.quantile(a, [.05, .25, .5, .75, .95], 0).round(4).tolist()
    peak_year = years[np.argmax(sim["world_conflict"], 1)]
    hist_years = list(range(1946, origin + 1))
    pop = np.exp(base["lpop"])
    gdp = np.exp(base["lg"])
    hist = {"years": hist_years,
            "world_conflict": [float(np.nanmean(base["conflict"][:, p.yi(y)])) for y in hist_years],
            "world_gdp": [float(np.nansum((gdp * pop)[:, p.yi(y)]) / np.nansum(np.where(np.isfinite(gdp[:, p.yi(y)]), pop[:, p.yi(y)], 0)))
                          for y in hist_years],
            "world_dem": [float(np.nansum((base["dem"] * pop)[:, p.yi(y)]) /
                                np.nansum(np.where(np.isfinite(base["dem"][:, p.yi(y)]), pop[:, p.yi(y)], 0))) for y in hist_years],
            "coups": [float(np.nansum(base["coup_attempts"][:, p.yi(y)] > 0)) if y >= 1950 else None for y in hist_years],
            "onsets": [float(np.nansum((base["conflict"][:, p.yi(y)] == 1) & (base["conflict"][:, p.yi(y) - 1] == 0)))
                       if y > 1946 else None for y in hist_years]}
    hist_max = max(hist["world_conflict"])
    world = {"years": years.tolist(),
             "world_conflict_q": q(sim["world_conflict"]), "world_gdp_q": q(sim["world_gdp"]),
             "world_dem_q": q(sim["world_dem"]), "coups_q": q(sim["coups"]), "onsets_q": q(sim["onsets"]),
             "region_conflict_q": {r: q(sim["region_conflict"][:, i]) for i, r in enumerate(p.region_names)},
             "peak_year_hist": np.bincount(peak_year - years[0], minlength=len(years)).tolist(),
             "p_exceed_hist_max": float((sim["world_conflict"].max(1) > hist_max).mean()),
             "hist_max_conflict": hist_max,
             "sample_paths": sim["world_conflict"][:12].round(4).tolist()}

    importance = {t: logit.coefficients(TASKS.index(t)) for t in HEADLINE}
    gbm_imp = {t: ens.fam["G"][0].importance(TASKS.index(t)) for t in HEADLINE}
    group_imp = {t: {g: float(sum(gbm_imp[t][f] for f in feats)) for g, feats in GROUPS.items()} for t in HEADLINE}
    meta = {"origin": origin, "end": end, "n_scenarios": n_scen, "n_countries": len(countries), "n_features": len(FEATURES),
            "base_rates": {t: float(base_rate[TASKS.index(t)]) for t in HEADLINE},
            "naive_risk_mean": {t: float(np.nanmean(naive.predict(X_now)[:, TASKS.index(t)])) for t in HEADLINE},
            "ensemble_combine": load_combine(),
            "family_risk_mean": {DISPLAY[f]: {t: float(np.nanmean(v[:, TASKS.index(t)])) for t in HEADLINE} for f, v in parts.items()},
            "scenario": scenario, "climate_damage": climate_damage,
            "growth_coefficients": {k: float(v) for k, v in growth.coefficients().items()},
            "democracy_coefficients": {k: float(v) for k, v in ddem.coefficients().items()},
            "logit_coefficients": {t: {k: float(v) for k, v in importance[t].items()} for t in HEADLINE},
            "gbm_group_importance": group_imp}
    _log("forecast done")
    scen_world = {k: {kk: v[kk] for kk in ("world_conflict_q", "world_gdp_q", "world_dem_q")} for k, v in scen_out.items()}
    from dataclasses import asdict
    ai_world = {k: {kk: vv for kk, vv in v.items() if kk not in ("gdp_end_median", "conflict_end")}
                for k, v in ai_out.items()}
    meta["ai_scenarios"] = {k: asdict(v) for k, v in AI_SCENARIOS.items()}
    meta["world_config"] = {k: v for k, v in asdict(wbase).items() if k != "ai"}
    return {"meta": meta, "world": world, "history": hist, "countries": countries, "scenarios": scen_world,
            "ai": ai_world, "interstate": interstate_out}


def _clean(o):
    """JSON-safe: numpy -> python, non-finite -> null."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


CRISES = {
    "taiwan_2027": {"name": "China-Taiwan war, 2027", "forced": [("CHN", "TWN", 2027)]},
    "taiwan_us_2027": {"name": "China-US war over Taiwan, 2027", "forced": [("CHN", "TWN", 2027), ("CHN", "USA", 2027), ("CHN", "JPN", 2027)]},
    "baltic_2028": {"name": "Russia attacks the Baltic states, 2028", "forced": [("EST", "RUS", 2028), ("LVA", "RUS", 2028), ("LTU", "RUS", 2028)]},
    # (2027, not 2026: the 2026 oil price is observed, so a hypothetical 2026 escalation could not move it)
    "gulf_2026": {"name": "The Iran war widens to Saudi Arabia in 2027", "forced": [("IRN", "ISR", 2027), ("IRN", "USA", 2027), ("IRN", "SAU", 2027)]},
    "hormuz_2027": {"name": "Gulf war with the Strait of Hormuz closed, 2027", "forced": [("IRN", "USA", 2027), ("IRN", "SAU", 2027), ("ARE", "IRN", 2027)],
                    "oil_war_jump": 0.69},   # assumption: about double the price for a year (vs +18% for a typical exporter war)
    "south_asia_2027": {"name": "India-Pakistan war, 2027", "forced": [("IND", "PAK", 2027)]},
    "korea_2028": {"name": "Second Korean war, 2028", "forced": [("KOR", "PRK", 2028), ("PRK", "USA", 2028)]},
}


def chronicle_worlds(p, base, net, growth, ddem, origin, end, clim_inputs, climate_damage, dy, n=300):
    """Event logs of a few representative simulated worlds (baseline and near-free cognition from 2030)."""
    from .world import WorldConfig, AI_SCENARIOS
    _log("chronicles of sampled worlds")
    picked = []
    for label, cfg, seed in [("baseline", WorldConfig(), 21), ("ai2030", WorldConfig(ai=AI_SCENARIOS["2030"]), 22)]:
        o = simulate(p, base, net, growth, ddem, origin, end, n_scen=n, clim=clim_inputs, climate_damage=climate_damage,
                     seed=seed, world=cfg, dyads=dy, chronicle=n)
        ch, w = o["chronicle"], o["world"]
        gdp_end = o["world_gdp"][:, -1]
        conf_mean = o["world_conflict"].mean(1)
        gpw = w["great_power_war"].max(1)
        rank = lambda a: np.argsort(np.argsort(a)) / (len(a) - 1)
        typical = int(np.argmin(np.abs(rank(gdp_end) - .5) + np.abs(rank(conf_mean) - .5)))
        choices = {"typical": typical} if label == "baseline" else {"ai_typical": typical}
        if label == "baseline":
            choices["calm"] = int(np.argmin(conf_mean + 0.5 * gpw))
            cand = np.where(gpw > 0)[0]
            if len(cand):
                choices["great_power_war"] = int(cand[np.argmin(np.abs(rank(gdp_end)[cand] - .5))])
        for kind, wi in choices.items():
            log = []
            for yr in ch:
                e = {"year": yr["year"], "world_gdp": round(yr["world_gdp"][wi]), "conflict_share": round(yr["conflict_share"][wi], 3)}
                for k in ("onset", "coup", "coup_failed", "dem_drop", "dem_rise", "slump"):
                    if yr[k][wi]:
                        e[k] = [p.names[p.codes.index(c)] for c in yr[k][wi]]
                if yr.get("new_wars") and yr["new_wars"][wi]:
                    e["new_interstate"] = [f"{p.names[p.codes.index(a)]} vs {p.names[p.codes.index(b)]}" + (" (war)" if wr else "")
                                           for a, b, wr in yr["new_wars"][wi]]
                if yr.get("pandemic") and yr["pandemic"][wi] > 0:
                    e["pandemic"] = True
                if yr.get("oil") is not None:
                    e["oil_index"] = round(float(100 * np.exp(yr["oil"][wi] - base["oil_price"][0][p.yi(origin)])))
                if yr.get("ai") is not None and label == "ai2030":
                    e["ai_adoption"] = round(yr["ai"], 2)
                log.append(e)
            picked.append({"kind": kind, "scenario": label, "world": int(wi), "events": log,
                           "great_power_war": bool(gpw[wi] > 0), "world_gdp_2065": float(gdp_end[wi]),
                           "conflict_mean": float(conf_mean[wi])})
    return picked


def crisis_explorer(p, base, net, growth, ddem, origin, end, n_scen, clim_inputs, climate_damage, dy, seed=11):
    """Paired runs (same seed) with and without a forced war, to trace how a crisis propagates."""
    from .world import WorldConfig
    _log("crisis explorer")
    run_ = lambda cfg: simulate(p, base, net, growth, ddem, origin, end, n_scen=n_scen, clim=clim_inputs,
                                climate_damage=climate_damage, seed=seed, world=cfg, dyads=dy)
    b0 = run_(WorldConfig())
    years = b0["years"].tolist()
    med = lambda a: np.median(a, 0)

    def summary(o):
        w = o["world"]
        return {"gdp": med(o["world_gdp"]), "conflict": o["world_conflict"].mean(0), "dem": med(o["world_dem"]),
                "oil": (100 * np.exp(med(w["oil"]) - base["oil_price"][0][p.yi(origin)])) if "oil" in w else None,
                "wars": w["active"].mean(0), "gpw": np.maximum.accumulate(w["great_power_war"], axis=1).mean(0),
                "gdp_c": o["gdp_q"][:, :, 2], "conf_c": o["conflict"],
                "pairs": w.get("pair_prob")}
    S0 = summary(b0)
    now = dy["now"].reset_index(drop=True)
    out = {"years": years, "baseline": {k: np.round(v, 4).tolist() for k, v in S0.items()
                                        if k in ("gdp", "conflict", "dem", "oil", "wars", "gpw") and v is not None},
           "scenarios": {}}
    for key, sc in CRISES.items():
        _log(f"  {sc['name']}")
        S1 = summary(run_(WorldConfig(forced=tuple(sc["forced"]), **({"oil_war_jump": sc["oil_war_jump"]} if "oil_war_jump" in sc else {}))))
        rel = lambda k: np.round(S1[k] / S0[k] - 1, 4).tolist()
        dif = lambda k: np.round(S1[k] - S0[k], 4).tolist()
        y5, y10 = years.index(min(sc["forced"][0][2] + 3, end)), years.index(min(sc["forced"][0][2] + 8, end))
        gc = S1["gdp_c"] / S0["gdp_c"] - 1
        cc = S1["conf_c"] - S0["conf_c"]
        by_country = {code: {"gdp_3y": round(float(gc[c, y5]), 4), "gdp_8y": round(float(gc[c, y10]), 4),
                             "civil_conflict_3y": round(float(cc[c, y5]), 4)}
                      for c, code in enumerate(p.codes) if np.isfinite(gc[c, y5])}
        pp = S1["pairs"] - S0["pairs"]
        forced_keys = {(min(a, b), max(a, b)) for a, b, _ in sc["forced"]}
        drawn = [(now.a[i], now.b[i], float(pp[i])) for i in np.argsort(-pp)[:40]
                 if (now.a[i], now.b[i]) not in forced_keys][:12]
        out["scenarios"][key] = {
            "name": sc["name"], "forced": sc["forced"], "world_gdp_rel": rel("gdp"), "conflict_diff": dif("conflict"),
            "dem_diff": dif("dem"), "oil_rel": rel("oil") if S0["oil"] is not None else None, "wars_diff": dif("wars"),
            "gpw_diff": dif("gpw"), "by_country": by_country,
            "drawn_in": [{"a": a, "b": b, "dp": round(d, 4)} for a, b, d in drawn]}
    return out


def step_interstate_hindcast(p, base, X, Yl, origins=((1995, 2015), (2004, 2024)), n_scen=2000, world_kw=None):
    """run the full simulator with the interstate layer from past origins and compare with what happened."""
    from sklearn.metrics import roc_auc_score
    from . import interstate as INTER
    from .world import WorldConfig
    _log("interstate hindcasts")
    Xf, Yf, ci, year = flatten(p, X, Yl)
    df = INTER.build(verbose=False)
    rows = []
    for O, E in origins:
        Ytr, tr = train_mask(Yf, year, O)
        net = PsychoNet(members=8, **NET_TUNED).fit(Xf[tr], Ytr[tr], ci[tr])
        growth, ddem = fit_transitions(p, X, base, O)
        dy = INTER.sim_inputs(INTER.run(verbose=False, origin=O, backtests=False, df=df))
        out = simulate(p, base, net, growth, ddem, O, E, n_scen=n_scen, demography="extrapolate",
                       world=WorldConfig(pandemics=False, **(world_kw or {})), dyads=dy)
        w = out["world"]
        real = df[(df.year > O) & (df.year <= E)].groupby("year").ev_ucdp.apply(lambda s_: (s_ == 1).sum())
        real = real.reindex(range(O + 1, E + 1)).to_numpy()
        now = dy["now"].reset_index(drop=True)
        fought = df[(df.year > O) & (df.year <= E) & (df.ev_ucdp == 1)][["a", "b"]].drop_duplicates()
        key = set(zip(fought.a, fought.b))
        yv = np.array([(a, b) in key for a, b in zip(now.a, now.b)], float)
        qa = np.quantile(w["active"], [.05, .95], 0)
        rows.append({"origin": O, "end": E, "active_per_year_sim": float(w["active"].mean()),
                     "active_per_year_real": float(real.mean()),
                     "years_in_90": float(((real >= qa[0]) & (real <= qa[1])).mean()),
                     "pairs_fought_real": int(yv.sum()), "pairs_fought_sim": float(w["pair_prob"].sum()),
                     "pair_auc": float(roc_auc_score(yv, w["pair_prob"]))})
        if "battle_deaths" in w:
            # interstate battle deaths over the hindcast (UCDP battle-related deaths, best estimates)
            brd = pd.read_csv(ROOT / "data" / "raw" / "ucdp" / "BattleDeaths_v25_1_conf.csv")
            brd = brd[brd.type_of_conflict.astype(str).str.contains("2") & (brd.year > O) & (brd.year <= E)]
            cum = w["battle_deaths"].sum(-1)
            real_d = float(brd.bd_best.sum())
            rows[-1].update({"battle_deaths_real": real_d, "battle_deaths_sim_median": float(np.median(cum)),
                             "battle_deaths_sim_90": np.quantile(cum, [.05, .95]).round(0).tolist(),
                             "battle_deaths_real_pctile": float((cum < real_d).mean())})
        if "oil" in w:
            ro = np.array([base["oil_price"][0][p.yi(t)] for t in range(O + 1, E + 1)])
            qo = np.quantile(w["oil"], [.05, .25, .75, .95], 0)
            rows[-1].update({"oil_in_90": float(((ro >= qo[0]) & (ro <= qo[3])).mean()),
                             "oil_in_50": float(((ro >= qo[1]) & (ro <= qo[2])).mean())})
        _log(f"  {rows[-1]}")
    return rows


def run_all(refresh=False, quick=False):
    OUT.mkdir(exist_ok=True)
    p, base, X, Yl = prepare(refresh)
    summ, cal, bt = step_backtest(p, X, Yl, origins=range(1990, 2016, 5) if quick else range(1980, 2016, 5),
                                  members=6 if quick else 8)
    hind, fans = step_hindcast(p, base, X, Yl, n_scen=2000 if quick else 4000)
    inter_hind = step_interstate_hindcast(p, base, X, Yl, n_scen=1000 if quick else 2000)
    fc = step_forecast(p, base, X, Yl, n_scen=3000 if quick else 10000, scen_n=1500 if quick else 4000)
    abl = OUT / "ablation.csv"
    fc["validation"] = {
        "backtest": summ.to_dict("records"),
        "backtest_split": pd.read_csv(OUT / "backtest_split.csv").to_dict("records") if (OUT / "backtest_split.csv").exists() else [],
        "calibration": cal.to_dict("records"),
        "hindcast": hind.to_dict("records"),
        "hindcast_fans": {str(k): v for k, v in fans.items()},
        "interstate_hindcast": inter_hind,
        "ablation": pd.read_csv(abl).to_dict("records") if abl.exists() else [],
    }
    with open(OUT / "forecast.json", "w") as f:
        json.dump(_clean(fc), f)
    pd.DataFrame([{k: v for k, v in c.items() if not isinstance(v, (list, dict))} for c in fc["countries"]]) \
        .to_csv(OUT / "country_forecasts.csv", index=False)
    _log(f"wrote {OUT/'forecast.json'}")
    return fc


def run_states():
    """US state module -> outputs/us_states.json."""
    from . import us_states
    _log("US states: data, backtest, 2025-2029 risks, stress index, climate")
    r = us_states.run()
    with open(OUT / "us_states.json", "w") as f:
        json.dump(_clean(r), f)
    _log(f"wrote {OUT / 'us_states.json'}")
    return r
