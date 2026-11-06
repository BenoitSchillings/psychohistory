"""How far ahead does the simulator have skill? From each of the adopt rule's 16 origins (1970-2009), the forecast's
own simulator (pipeline.build_sim_env, world layer on, AI off) runs 15 years; at every horizon h = 1..15 its
per-country forecasts are scored against what happened and against naive references built from data before the
origin:
- civil conflict (country in conflict in year t0+h): Brier skill against the Markov reference, P(in conflict at
  t0+h | in conflict or not at t0) estimated on all country-years 1946..t0 (persistence with its historical decay);
- new conflict by t0+h among countries at peace at t0: Brier skill against that reference's onset rate;
- log GDP per person change to t0+h: CRPS skill against each country's own 20-year mean growth plus the pooled
  historical spread of h-year deviations before the origin.
Skill > 0 means better than the reference. The horizon where skill reaches zero bounds what may be called a forecast
(second debate, 6 November 2026). python -P scripts/skill_horizon.py [--n 1000] -> outputs/skill_horizon.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

ORIGINS = [1970, 1973, 1976, 1979, 1980, 1983, 1986, 1989, 1990, 1993, 1996, 1999, 2000, 2003, 2006, 2009]
H = 15


def crps_ens(x, y):
    """x [S, N] ensemble, y [N] -> [N]"""
    x = np.sort(x, 0)
    n = x.shape[0]
    w = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    return np.mean(np.abs(x - y[None]), 0) - np.sum(w * x, 0) / n ** 2


def main():
    from psychohistory import pipeline as PL, simulate as SIM, world as WM, interstate as INTER
    from psychohistory.backtest import flatten
    from psychohistory.data import LAST_OBS_YEAR
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 1000
    p, base, X, Yl = PL.prepare(False)
    Xf, Yf, ci, year = flatten(p, X, Yl)
    df = INTER.build(verbose=False)
    cf, lg = base["conflict"], base["lg"]
    acc = {k: {h: [] for h in range(1, H + 1)} for k in ("conf_m", "conf_r", "ons_m", "ons_r", "gdp_m", "gdp_r")}
    rng = np.random.default_rng(0)
    for O in ORIGINS:
        E = min(O + H, LAST_OBS_YEAR)
        i0 = p.yi(O)
        net = PL.build_sim_env(p, X, Xf, Yf, ci, year, O, base=base)
        growth, ddem = SIM.fit_transitions(p, X, base, O)
        dy = INTER.sim_inputs(INTER.run(verbose=False, origin=O, backtests=False, df=df))
        o = SIM.simulate(p, base, net, growth, ddem, O, E, n_scen=n, demography="extrapolate", record_paths=n, seed=O + 11,
                         world=WM.WorldConfig(ai=WM.AI_SCENARIOS["none"], sample_params=True), dyads=dy)
        pc, gdp = o["paths"]["conflict"], o["paths"]["gdp"]                 # [S, C, T]
        s0 = cf[:, i0]
        # Markov reference from 1946..O: P(in conflict h years later | state now), and P(any onset within h | at peace)
        a46 = p.yi(1946)
        for h in range(1, E - O + 1):
            st, fu = cf[:, a46:i0 - h + 1], cf[:, a46 + h:i0 + 1]
            ok = np.isfinite(st) & np.isfinite(fu)
            r_in = (fu[ok & (st == 1)] == 1).mean()
            r_out = (fu[ok & (st == 0)] == 1).mean()
            win = np.stack([cf[:, a46 + k:i0 - h + 1 + k] for k in range(1, h + 1)], 0)       # [h, C, Y]
            st0 = cf[:, a46:i0 - h + 1]
            okw = np.isfinite(win).all(0) & (st0 == 0)
            r_ons = (win.max(0)[okw] == 1).mean()
            act = cf[:, i0 + h]
            okc = np.isfinite(act) & np.isfinite(s0)
            pm = pc[:, okc, h - 1].mean(0)
            pr = np.where(s0[okc] == 1, r_in, r_out)
            acc["conf_m"][h].append(((pm - act[okc]) ** 2))
            acc["conf_r"][h].append(((pr - act[okc]) ** 2))
            seen = cf[:, i0 + 1:i0 + h + 1]
            peace = okc & (s0 == 0) & np.isfinite(seen).all(1)
            ever_a = (seen[peace] == 1).any(1).astype(float)
            ever_m = (pc[:, peace, :h] > 0.5).any(-1).mean(0)
            acc["ons_m"][h].append((ever_m - ever_a) ** 2)
            acc["ons_r"][h].append((r_ons - ever_a) ** 2)
            # GDP: log change to O+h
            ga = lg[:, i0 + h] - lg[:, i0]
            gm = np.log(gdp[:, :, h - 1]) - lg[:, i0][None]
            past = lg[:, i0] - lg[:, max(0, i0 - 20)]
            mu = past / min(20, i0)
            okg = np.isfinite(ga) & np.isfinite(gm).all(0) & np.isfinite(mu)
            # pooled historical h-year deviations from each country's own trailing 20-year mean, before the origin
            devs = []
            for t in range(max(20, i0 - 60), i0 - h + 1, 2):          # the 60 years before the origin (GDP from the 1800s)
                d_ = (lg[:, t + h] - lg[:, t]) - h * (lg[:, t] - lg[:, t - 20]) / 20
                devs.append(d_[np.isfinite(d_)])
            devs = np.concatenate(devs)
            ref = h * mu[okg][None] + rng.choice(devs, size=(500, int(okg.sum())))
            acc["gdp_m"][h].append(crps_ens(gm[:, okg], ga[okg]))
            acc["gdp_r"][h].append(crps_ens(ref, ga[okg]))
        print(O, "done", flush=True)
    out = {"origins": ORIGINS, "n_worlds": n, "by_horizon": {}}
    for h in range(1, H + 1):
        r = {}
        for k, (m, ref) in {"conflict": ("conf_m", "conf_r"), "new_conflict": ("ons_m", "ons_r"), "gdp": ("gdp_m", "gdp_r")}.items():
            if acc[m][h]:
                a, b = np.concatenate(acc[m][h]), np.concatenate(acc[ref][h])
                # bootstrap over origins for an interval on the skill
                per = [(np.mean(x), np.mean(y)) for x, y in zip(acc[m][h], acc[ref][h])]
                bs = []
                for _ in range(1000):
                    idx = rng.integers(0, len(per), len(per))
                    sm, sr = np.mean([per[i][0] for i in idx]), np.mean([per[i][1] for i in idx])
                    bs.append(1 - sm / sr)
                r[k] = {"skill": round(float(1 - a.mean() / b.mean()), 4), "ci90": [round(float(x), 4) for x in np.quantile(bs, [0.05, 0.95])],
                        "origins": len(acc[m][h])}
        out["by_horizon"][h] = r
        print(h, json.dumps(r), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "skill_horizon.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
