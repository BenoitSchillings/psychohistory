"""Out-of-sample scorecard for 2025: models trained on data up to 2024, scored on what happened in 2025.

Outcomes (data/raw/outcomes/README.md): UCDP/PRIO 26.1 conflicts and GED 26.1 events, V-Dem v16 (democracy
and Powell & Thyne coups as carried by V-Dem), and the ViEWS forecast run fatalities002_2024_12_t01 (data
through December 2024) for a head-to-head comparison.
Writes outputs/score_2025.json.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from psychohistory.pipeline import prepare
from psychohistory.backtest import flatten, train_mask
from psychohistory.models import SeldonEnsemble, NaiveModel, load_combine
from psychohistory.features import TASKS
from psychohistory import interstate as I

O = os.path.join(ROOT, "data", "raw", "outcomes")
ORIGIN = 2024


def score(y, p, base_rate):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    out = {"n": int(len(y)), "events": int(y.sum())}
    if 0 < y.sum() < len(y):
        out["auc"] = float(roc_auc_score(y, p))
    out["brier"] = float(np.mean((p - y) ** 2))
    out["brier_base"] = float(np.mean((base_rate - y) ** 2))
    out["bss"] = 1 - out["brier"] / out["brier_base"]
    return out


def top_hits(codes, y, p, k=10):
    order = np.argsort(-np.nan_to_num(p, nan=-1))[:k]
    return [{"code": codes[i], "p": round(float(p[i]), 3), "happened": bool(y[i] == 1)} for i in order], \
        [{"code": codes[i], "p": round(float(p[i]), 3)} for i in np.where(y == 1)[0]]


def main():
    p, base, X, Yl = prepare()
    codes = list(p.codes)
    Xf, Yf, ci, year = flatten(p, X, Yl)
    Ytr, tr = train_mask(Yf, year, ORIGIN, first_year=1860)
    ens = SeldonEnsemble(load_combine(), members=6).fit(Xf[tr], Ytr[tr], ci[tr], year[tr])
    tr55 = tr & (year >= 1955)
    naive = NaiveModel().fit(Xf[tr55], Ytr[tr55])
    X_now = X[:, p.yi(ORIGIN)]
    risk, nr = ens.predict(X_now), naive.predict(X_now)
    base_rate = np.nanmean(Ytr[tr55], 0)
    k = {t: TASKS.index(t) for t in TASKS}
    res = {"origin": ORIGIN, "note": "models trained only on data up to 2024; outcomes are 2025"}
    conf24 = base["conflict"][:, p.yi(ORIGIN)]

    # --- civil conflict (UCDP 26.1, government-located intrastate conflict >= 25 deaths)
    u = pd.read_csv(os.path.join(O, "ucdp_2025_cy.csv"))
    u = u[u.year == 2025].set_index("iso3")
    y_intra = np.array([u.intrastate_conflict.get(c, np.nan) for c in codes], float)
    peace = conf24 == 0
    res["civil_onset"] = {"model": score(y_intra[peace], risk[peace, k["onset_1y"]], base_rate[k["onset_1y"]]),
                          "naive": score(y_intra[peace], nr[peace, k["onset_1y"]], base_rate[k["onset_1y"]])}
    th, ev = top_hits([c for c, m in zip(codes, peace) if m], y_intra[peace], risk[peace, k["onset_1y"]])
    res["civil_onset"]["top10"], res["civil_onset"]["onsets"] = th, ev
    # incidence: P(at peace) * onset + P(at war) * (1 - termination)
    p_inc = np.where(conf24 == 1, 1 - risk[:, k["termination_1y"]], risk[:, k["onset_1y"]])
    ok = np.isfinite(conf24)
    res["civil_incidence"] = {"model": score(y_intra[ok], p_inc[ok], float(np.nanmean(conf24)))}
    res["civil_incidence"]["persistence"] = score(y_intra[ok], conf24[ok], float(np.nanmean(conf24)))

    # --- ViEWS head-to-head on ViEWS's own target (state-based deaths located in the country, GED 26.1)
    vw = pd.read_csv(os.path.join(O, "views_2025_cm.csv"))
    ged = pd.read_csv(os.path.join(O, "ucdp_ged_sb_cm.csv"))
    ged = ged[ged.year == 2025]
    gy = ged.groupby("iso3").sb_deaths_best.sum()
    vp = vw.groupby("iso3").prob_ge25.apply(lambda s: 1 - np.prod(1 - s))         # P(any month >= 25)
    vexp = vw.groupby("iso3").pred_fatalities.sum()
    gm = ged.groupby("iso3").sb_ge25.max()
    common = [c for c in codes if c in vp.index and c in gm.index and np.isfinite(conf24[codes.index(c)])]
    idx = [codes.index(c) for c in common]
    y_v = gm.reindex(common).to_numpy(float)                                 # any month with >= 25 deaths in 2025
    our = np.where(conf24[idx] == 1, 1 - risk[idx, k["termination_1y"]], risk[idx, k["onset_1y"]])
    # our model forecasts intrastate conflict; a country can also host interstate fighting (Ukraine) - add our
    # interstate involvement for countries fighting in 2024 as a simple union
    inter24 = np.nan_to_num(base["interstate_recent"][idx, p.yi(ORIGIN)] > 0.5) if "interstate_recent" in base else np.zeros(len(idx))
    our_u = 1 - (1 - our) * (1 - 0.8 * inter24)
    br = float(y_v.mean())
    res["views_head_to_head"] = {
        "target": "any month of 2025 with >= 25 state-based battle deaths located in the country (UCDP GED 26.1)",
        "views_run": "fatalities002_2024_12_t01", "n": len(common),
        "views": score(y_v, vp.reindex(common).to_numpy(), br),
        "seldon": score(y_v, our_u, br),
        "average": score(y_v, (our_u + vp.reindex(common).to_numpy()) / 2, br),
        "persistence": score(y_v, conf24[idx], br)}
    # blending the monthly model (issued from December 2024 with the data then available) into the annual model
    from psychohistory import monthly as MO
    gw2iso, mcodes = MO.gw_map()
    Dm = MO.deaths(mcodes, gw2iso, 2024, 12, vintage_2024=True)
    Fm, hm = MO.features(Dm, mcodes, p)
    tm = MO.midx(2024, 12)
    mann = MO.annual_probability(MO.predict(MO.fit(Fm, hm, tm), Fm, tm))
    mp = dict(zip(mcodes, mann))
    pm = np.array([mp.get(c, np.nan) for c in common])
    lo = lambda q: np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
    blend = 1 / (1 + np.exp(-(lo(our_u) + lo(pm)) / 2))
    res["views_head_to_head"]["monthly_annualized"] = score(y_v, pm, br)
    res["views_head_to_head"]["seldon_blend_monthly"] = score(y_v, blend, br)
    # and on the annual model's own target (government-located intrastate conflict, UCDP/PRIO 26.1)
    yi = y_intra[idx]
    pin = p_inc[idx]
    res["blend_on_intrastate"] = {"annual": score(yi, pin, float(np.nanmean(yi))),
                                  "monthly": score(yi, pm, float(np.nanmean(yi))),
                                  "blend": score(yi, 1 / (1 + np.exp(-(lo(pin) + lo(pm)) / 2)), float(np.nanmean(yi)))}
    pz = conf24[idx] == 0
    res["views_head_to_head"]["new_cases_only"] = {
        "n": int(pz.sum()), "events": int(y_v[pz].sum()),
        "views": score(y_v[pz], vp.reindex(common).to_numpy()[pz], br),
        "seldon": score(y_v[pz], our_u[pz], br)}

    # --- coups (V-Dem v16 carries Powell & Thyne: GNB, MDG in 2025); sensitivity adds Benin's reported attempt
    cp = pd.read_csv(os.path.join(O, "vdem16_coups_cy.csv"))
    hits = set(cp[(cp.year == 2025) & (cp.e_pt_coup_attempts > 0)].iso3)
    y_c = np.array([1.0 if c in hits else 0.0 for c in codes])
    okc = np.isfinite(risk[:, k["coup_1y"]])
    res["coups"] = {"events": sorted(hits), "model": score(y_c[okc], risk[okc, k["coup_1y"]], base_rate[k["coup_1y"]]),
                    "naive": score(y_c[okc], nr[okc, k["coup_1y"]], base_rate[k["coup_1y"]]),
                    "ranks": {c: int((risk[okc, k["coup_1y"]] > risk[codes.index(c), k["coup_1y"]]).sum() + 1) for c in hits}}
    y_c2 = y_c.copy(); y_c2[codes.index("BEN")] = 1.0
    res["coups"]["with_benin"] = score(y_c2[okc], risk[okc, k["coup_1y"]], base_rate[k["coup_1y"]])
    res["coups"]["ranks"]["BEN"] = int((risk[okc, k["coup_1y"]] > risk[codes.index("BEN"), k["coup_1y"]]).sum() + 1)

    # --- autocratization: the 5-year event (a 0.10 fall in the electoral democracy index) already happened in 2025?
    vd = pd.read_csv(os.path.join(O, "vdem16_cy.csv")).pivot(index="iso3", columns="year", values="v2x_polyarchy")
    drop = (vd[2025] - vd[2024]).reindex(codes).to_numpy()
    lvl = vd[2024].reindex(codes).to_numpy()
    y_a = np.where(np.isfinite(drop), (drop <= -0.10).astype(float), np.nan)
    oka = np.isfinite(y_a) & (lvl >= 0.10) & np.isfinite(risk[:, k["autocratization_5y"]])
    res["autocratization_early"] = {
        "events": [codes[i] for i in np.where(oka & (y_a == 1))[0]],
        "model_5y_risk_auc": score(y_a[oka], risk[oka, k["autocratization_5y"]], 0.02),
        "naive_auc": score(y_a[oka], nr[oka, k["autocratization_5y"]], 0.02),
        "biggest_falls": sorted([(codes[i], round(float(drop[i]), 3), round(float(risk[i, k["autocratization_5y"]]), 3))
                                 for i in np.where(oka)[0]], key=lambda t: t[1])[:10]}

    # --- interstate pairs (dyad model, 1-year)
    r = I.run(verbose=False, backtests=False, origin=ORIGIN)     # models fitted on data up to 2024 only
    now = r["df"][r["df"].year == ORIGIN].reset_index(drop=True)
    m1, cont = r["model_1y"], r["continuation"]
    p_on = m1.predict(now[I.FEATS_ON].to_numpy(float))
    run_y = now.conflict_run_years.fillna(0).clip(0, 3).astype(int).to_numpy()
    p_c = np.array([0.0, cont[1], cont[2], cont[3]])[run_y]
    ong = now.ongoing.fillna(0).to_numpy() > 0.5
    p_pair = np.where(ong, p_c, p_on)      # wars in progress: the duration table; pairs at peace: the onset model
    dy = pd.read_csv(os.path.join(O, "ucdp_2025_interstate_dy.csv"))
    fought = set(zip(dy[dy.year == 2025].iso3_a, dy[dy.year == 2025].iso3_b))
    y_p = np.array([(a, b) in fought for a, b in zip(now.a, now.b)], float)
    res["interstate"] = {"pairs_fought_2025": sorted(f"{a}-{b}" for a, b in fought),
                         "not_in_pair_set": sorted(f"{a}-{b}" for a, b in fought if not ((now.a == a) & (now.b == b)).any()),
                         "all_pairs": score(y_p, p_pair, float(y_p.mean())),
                         "new_only": score(y_p[~ong], p_pair[~ong], float(y_p[~ong].mean())),
                         "pair_probs": {f"{a}-{b}": round(float(p_pair[i]), 4) for i, (a, b) in enumerate(zip(now.a, now.b))
                                        if (a, b) in fought}}
    json.dump(res, open(os.path.join(ROOT, "outputs", "score_2025.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
