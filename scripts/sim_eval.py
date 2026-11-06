"""Evaluation harness for simulator changes: hindcasts from past origins (models trained only on data before each
origin, demography extrapolated), scored with proper scores on what happened. The network is the forecast's
(pipeline.build_sim_env, with the family draw; since 2026-11-05) and only the world layer is off. Decisions follow
docs/adopt_rule.md (leave-one-decade-out over all origins, a pre-registered composite, measured noise floors).

python -P scripts/sim_eval.py TAG FIRST LAST STEP ['{"simulate kwarg": value}'] ['{"module.CONST": value}']
-> outputs/sim_eval/TAG.json (per origin)          python -P scripts/sim_eval.py --compare TAG_A TAG_B

Metrics per origin (horizon H = 10 years, or to the last observed year):
- conflict_brier: Brier score of each country's simulated probability of conflict in each year (all country-years)
- onset_auc: AUC of the probability of any new conflict, countries at peace at the origin
- world_share_crps: CRPS of the share of country-years in conflict over the horizon (world level)
- onset_brier / onset_ll: each country's simulated probability of any new conflict over the horizon, countries at
  peace at the origin (Brier, log loss); diag_safe20: simulated / actual countries with a new conflict among the 20 at
  peace with the lowest simulated risk (added 2026-11-03: the safest countries got too many wars)
- onsets_crps: CRPS of the number of new conflicts among countries at peace at the origin
- gdp_crps: mean CRPS of log GDP per person at the horizon (relative to the origin), per country
- gdp_in90: share of countries whose realised GDP falls inside the simulated 90% band
- gdp_disp_crps: CRPS of the cross-country spread (sd) of log GDP per person growth over the horizon
- world_gdp_crps: CRPS of world (population-weighted) log GDP per person growth over the horizon
- dem_crps: mean CRPS of the electoral democracy index at the horizon
- coup_brier / coup_auc: each country's simulated probability of any coup attempt over the horizon vs what happened
- coups_crps: CRPS of the world number of country-years with a coup attempt over the horizon
"""
import importlib
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "sim_eval")
LOWER_BETTER = ("conflict_brier", "onset_brier", "onset_ll", "coup_brier", "coups_crps", "world_share_crps", "onsets_crps", "gdp_crps", "world_gdp_crps", "world_gdppc_crps", "dem_crps", "gdp_disp_crps")


def crps(x, y):
    """CRPS of an ensemble x [S] (or [S, N] -> [N]) against y"""
    x = np.sort(np.asarray(x, float), 0)
    n = x.shape[0]
    w = (2 * np.arange(1, n + 1) - n - 1).reshape((-1,) + (1,) * (x.ndim - 1))
    return np.mean(np.abs(x - y), 0) - np.sum(w * x, 0) / n ** 2


def compare(a, b):
    A, B = (json.load(open(os.path.join(OUT, f"{t}.json")))["by_origin"] for t in (a, b))
    common = sorted(set(A) & set(B), key=int)
    print(f"{b} vs {a}: {len(common)} origins ({common[0]}-{common[-1]})")
    res = {}
    for m in [m_ for m_ in A[common[0]] if m_ in B[common[0]]]:
        va = np.array([A[o][m] for o in common], float)
        vb = np.array([B[o][m] for o in common], float)
        ok = np.isfinite(va) & np.isfinite(vb)
        if not ok.any():
            continue
        lower = m in LOWER_BETTER
        better = (vb < va) if lower else (vb > va)
        rel = (vb[ok].mean() / va[ok].mean() - 1) if va[ok].mean() else np.nan
        res[m] = {"a": round(float(va[ok].mean()), 5), "b": round(float(vb[ok].mean()), 5), "rel": round(float(rel), 4),
                  "b_better": int(better[ok].sum()), "n": int(ok.sum()), "lower_is_better": lower}
        print(f"  {m:18s} {res[m]['a']:>10} -> {res[m]['b']:>10}  ({100 * rel:+.1f}%)  b better at {res[m]['b_better']}/{res[m]['n']}"
              f"{'' if m in ('gdp_in90',) else ''}")
    return res


def main():
    from psychohistory import pipeline as PL
    from psychohistory.backtest import flatten, train_mask
    from psychohistory.models import PsychoNet, NET_TUNED
    from psychohistory import simulate as SIM
    from psychohistory.data import LAST_OBS_YEAR
    from sklearn.metrics import roc_auc_score

    tag, first, last, step = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    kw = json.loads(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else {}
    consts = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
    for k, v in consts.items():
        mod, name = k.rsplit(".", 1)
        setattr(importlib.import_module(f"psychohistory.{mod}"), name, v)
    if "mood_sd" in kw:
        kw["mood_sd"] = tuple(kw["mood_sd"])
    S, H = int(kw.pop("n_scen", 3000)), int(kw.pop("horizon", 10))
    net_seed = int(kw.pop("net_seed", 0))
    # the ensemble's central offset (simulate.ENS_OFFSET, adopted 2026-11-03) from scripts/ens_cache.py; "" = without
    # net_seed: refit of the forecast's ensemble with every seed offset (pipeline.build_sim_env); members: its size
    legacy, members = bool(kw.pop("legacy", 0)), kw.pop("members", None)
    if "ens_offset" in kw:                                    # the ensemble's central offset (simulate.ENS_OFFSET)
        SIM.ENS_OFFSET = kw.pop("ens_offset")
    p, base, X, Yl = PL.prepare(False)
    Xf, Yf, ci, year = flatten(p, X, Yl)
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, f"{tag}.json")
    prev = json.load(open(f)) if os.path.exists(f) else {}
    out, cfgs = prev.get("by_origin", {}), prev.get("config", {})
    for O in range(first, last + 1, step):
        if str(O) in out:
            continue
        E = min(O + H, LAST_OBS_YEAR)
        growth, ddem = SIM.fit_transitions(p, X, base, O)
        kw_o = dict(kw)
        if legacy:
            # (before 2026-11-05: a stand-alone 8-member network, no family draw; for reproducing old tags only)
            Ytr, tr = train_mask(Yf, year, O)
            net = PsychoNet(members=8, seed=net_seed, **NET_TUNED).fit(Xf[tr], Ytr[tr], ci[tr])
        else:
            # the forecast's simulator network and its family draw (pipeline.build_sim_env), without the world layer
            net = PL.build_sim_env(p, X, Xf, Yf, ci, year, O, members=members, seed=net_seed, base=base)
            kw_o["struct"] = True
        o = SIM.simulate(p, base, net, growth, ddem, O, E, n_scen=S, demography="extrapolate", record_paths=S,
                         seed=O + 7, **kw_o)
        cf, gdp, dem = o["paths"]["conflict"], o["paths"]["gdp"], o["paths"]["dem"]       # [S, C, T]
        i0, i1 = p.yi(O), p.yi(E)
        act = base["conflict"][:, i0:i1 + 1]
        ok = np.isfinite(act).all(1)
        r = {}
        full = np.concatenate([np.repeat(act[None, :, :1], S, 0), cf], -1)
        pc = cf[:, ok].mean(0)                                                   # [C, T] probability
        r["conflict_brier"] = float(np.mean((pc - act[ok, 1:]) ** 2))
        ons_sim = ((full[:, :, 1:] > 0.5) & (full[:, :, :-1] < 0.5)).any(-1)       # [S, C]
        ons_act = ((act[:, 1:] == 1) & (act[:, :-1] == 0)).any(-1)
        peace = ok & (act[:, 0] == 0)
        r["onset_auc"] = float(roc_auc_score(ons_act[peace], ons_sim[:, peace].mean(0))) if 0 < ons_act[peace].sum() < peace.sum() else np.nan
        po = np.clip(ons_sim[:, peace].mean(0), 0.5 / S, 1 - 0.5 / S)
        ya = ons_act[peace].astype(float)
        r["onset_brier"] = float(np.mean((po - ya) ** 2))
        r["onset_ll"] = float(-np.mean(ya * np.log(po) + (1 - ya) * np.log(1 - po)))
        safe = np.argsort(po)[:20]
        r["diag_safe20_sim"], r["diag_safe20_act"] = float(po[safe].sum()), float(ya[safe].sum())
        r["world_share_crps"] = float(crps(cf[:, ok].mean((1, 2)), act[ok, 1:].mean()))
        # diagnostics (ratios simulated mean / actual): war level, onsets and endings over all country-years
        ends_sim = ((full[:, ok, :-1] > 0.5) & (full[:, ok, 1:] < 0.5)).sum((1, 2)).mean()
        ends_act = ((act[ok, :-1] == 1) & (act[ok, 1:] == 0)).sum()
        ons_all_sim = ((full[:, ok, :-1] < 0.5) & (full[:, ok, 1:] > 0.5)).sum((1, 2)).mean()
        ons_all_act = ((act[ok, :-1] == 0) & (act[ok, 1:] == 1)).sum()
        r["diag_share_ratio"] = float(cf[:, ok].mean() / max(act[ok, 1:].mean(), 1e-9))
        r["diag_onsets_ratio"] = float(ons_all_sim / max(ons_all_act, 1))
        r["diag_ends_ratio"] = float(ends_sim / max(ends_act, 1))
        r["onsets_crps"] = float(crps(ons_sim[:, peace].sum(1).astype(float), float(ons_act[peace].sum())))
        # year clustering: where each year's actual number of new conflicts falls among the simulated worlds
        # (randomised PIT); a simulator without enough common shocks puts too many years in the outer 5% tails
        # (expected share 0.10)
        on_y_sim = ((full[:, ok, :-1] < 0.5) & (full[:, ok, 1:] > 0.5)).sum(1).astype(float)      # [S, T]
        on_y_act = ((act[ok, :-1] == 0) & (act[ok, 1:] == 1)).sum(0).astype(float)                  # [T]
        u_ = np.random.default_rng(O).random(len(on_y_act))
        pit = (on_y_sim < on_y_act[None]).mean(0) + u_ * (on_y_sim == on_y_act[None]).mean(0)
        r["diag_onset_year_tails"] = float(((pit < 0.05) | (pit > 0.95)).mean())
        r["diag_onset_year_var_ratio"] = float(np.var(on_y_act - on_y_sim.mean(0)) / max(on_y_sim.var(0).mean(), 1e-9))
        lg_sim = np.log(gdp[:, :, -1]) - base["lg"][:, i0][None]
        lg_act = base["lg"][:, i1] - base["lg"][:, i0]
        okg = np.isfinite(lg_act) & np.isfinite(lg_sim).all(0)
        r["gdp_crps"] = float(crps(lg_sim[:, okg], lg_act[okg]).mean())
        q = np.quantile(lg_sim[:, okg], [.05, .95], 0)
        r["gdp_in90"] = float(((lg_act[okg] >= q[0]) & (lg_act[okg] <= q[1])).mean())
        # spread of country growth over the horizon (do simulated countries drift apart too much?)
        disp_sim = lg_sim[:, okg].std(1)
        r["gdp_disp_crps"] = float(crps(disp_sim, lg_act[okg].std()))
        r["diag_gdp_disp_ratio"] = float(np.median(disp_sim) / max(lg_act[okg].std(), 1e-6))
        pop = np.exp(base["lpop"][:, i0]) if "lpop" in base else np.ones(len(lg_act))
        w = np.where(okg & np.isfinite(pop), pop, 0)
        w = w / w.sum()
        r["world_gdp_crps"] = float(crps(np.log((np.exp(lg_sim[:, okg]) * w[okg]).sum(1)),
                                         np.log((np.exp(lg_act[okg]) * w[okg]).sum())))
        # annual volatility of world GDP per person (origin population weights, same countries): simulated spread of
        # yearly world growth relative to the realised spread, and the share of years in which it falls
        lgp = np.log(gdp[:, okg])                                                  # [S, C', T]
        lga = base["lg"][okg, i0 + 1:i1 + 1]
        okt = np.isfinite(lga).all(0)
        if okt.sum() >= 4:
            wg = w[okg][None, :, None]
            ws = np.log((np.exp(lgp[..., okt]) * wg).sum(1))
            wa = np.log((np.exp(lga[:, okt]) * wg[0]).sum(0))
            gs, ga = np.diff(ws, axis=1), np.diff(wa)
            r["diag_world_vol_ratio"] = float(gs.std(1).mean() / max(ga.std(), 1e-6))
            r["diag_world_falls_sim"] = float((gs < 0).mean())
            r["diag_world_falls_act"] = float((ga < 0).mean())
        # world GDP per person (population-weighted levels, as simulate's world_gdp), its log change over the horizon
        # (world_gdp_crps above scores the population-weighted mean of growth ratios; kept for comparability)
        d0 = np.log((np.exp(base["lg"][okg, i0]) * w[okg]).sum())
        r["world_gdppc_crps"] = float(crps(np.log((gdp[:, okg, -1] * w[okg]).sum(1)) - d0,
                                           np.log((np.exp(base["lg"][okg, i1]) * w[okg]).sum()) - d0))
        d_act = base["dem"][:, i1]
        okd = np.isfinite(d_act) & np.isfinite(dem[:, :, -1]).all(0)
        r["dem_crps"] = float(crps(dem[:, okd, -1], d_act[okd]).mean())
        ca = base["coup_attempts"][:, i0 + 1:i1 + 1]
        okc = np.isfinite(ca).all(1)
        if okc.sum() > 10:
            c_act = (ca[okc] > 0).any(1)
            pcp = o["ever_coup"][okc, -1]
            r["coup_brier"] = float(np.mean((pcp - c_act) ** 2))
            r["coup_auc"] = float(roc_auc_score(c_act, pcp)) if 0 < c_act.sum() < okc.sum() else np.nan
            n_sim = o["paths"]["coup"][:, okc].sum((1, 2)).astype(float)          # same countries and years as the actual
            r["coups_crps"] = float(crps(n_sim, float((ca[okc] > 0).sum())))
            r["diag_coups_ratio"] = float(n_sim.mean() / max((ca[okc] > 0).sum(), 1))
        out[str(O)] = r
        print(O, E, json.dumps({k: round(v, 5) for k, v in r.items()}), flush=True)
        cfgs[str(O)] = o["config"]
        json.dump({"tag": tag, "kwargs": kw, "consts": consts, "by_origin": out, "config": cfgs}, open(f, "w"), indent=1,
                  default=str)


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        main()
