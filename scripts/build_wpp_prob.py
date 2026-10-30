"""UN World Population Prospects 2024 probabilistic projections (bayesTFR / bayesPop, via the wpp2024 R package's data,
CC BY 3.0 IGO) -> data/wpp_prob.npz: for each panel country and year 2024-2100, the spread of log total fertility
(sd implied by the 80% interval: for migration-driven populations such as Kuwait or Hong Kong the UN's 95% lower
bound reaches zero late in the century, and a lognormal fitted to it would be infinite) and of log population, and the correlation between countries' fertility errors that
reproduces the UN's 95% interval for world population in 2065.

The simulator draws one persistent fertility error per world and country (WorldConfig.demog_uncertainty), which moves
births, population and the youth share through its fertility channel.
python -P scripts/build_wpp_prob.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import pyreadr

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "demography", "wpp2024_prob")
Z95 = 1.959964
Z80 = 1.281552
SIG_CAP = 1.0


def main():
    from psychohistory.data import get_panel
    p = get_panel()
    pop = list(pyreadr.read_r(os.path.join(RAW, "popproj1dt.rda")).values())[0]
    tfr = list(pyreadr.read_r(os.path.join(RAW, "tfrproj1dt.rda")).values())[0]
    iso = pd.read_csv(os.path.join(ROOT, "data", "raw", "iso3166.csv"), dtype=str, keep_default_na=False)
    num2a3 = {int(n): a for n, a in zip(iso["country-code"], iso["alpha-3"])}
    num2a3.update({412: "XKX", 158: "TWN"})            # (Kosovo, Taiwan: UN codes outside ISO 3166)
    years = np.arange(2024, 2101)
    ci = {c: i for i, c in enumerate(p.codes)}
    C, T = len(p.codes), len(years)
    sig_tfr, sig_pop, med_pop = np.full((C, T), np.nan), np.full((C, T), np.nan), np.full((C, T), np.nan)
    med_tfr = np.full((C, T), np.nan)
    for df, kind in ((tfr, "tfr"), (pop, "pop")):
        df = df.copy()
        df["iso"] = df.country_code.astype(int).map(num2a3)
        for (a3, g) in df.groupby("iso"):
            if a3 not in ci:
                continue
            g = g.set_index(g.year.astype(int)).reindex(years)
            if kind == "tfr":
                sig_tfr[ci[a3]] = (np.log(g.tfr_80u) - np.log(g.tfr_80l)) / (2 * Z80)
                med_tfr[ci[a3]] = g.tfr
            else:
                med = g.popM + g.popF
                sig_pop[ci[a3]] = (np.log(g.pop_80u) - np.log(g.pop_80l.where(g.pop_80l > 0))) / (2 * Z80)
                med_pop[ci[a3]] = med
    # world population: the correlation rho between countries' errors that reproduces the UN world interval in 2065
    w = pop[pop.country_code.astype(int) == 900].set_index("year")
    t65 = list(years).index(2065)
    sig_w = float((np.log(w.loc[2065, "pop_80u"]) - np.log(w.loc[2065, "pop_80l"])) / (2 * Z80))
    a = np.nan_to_num(med_pop[:, t65] * sig_pop[:, t65])
    W = np.nansum(med_pop[:, t65])
    s_ind, s_full = np.sqrt((a ** 2).sum()) / W, a.sum() / W
    rho = float(np.clip((sig_w ** 2 - s_ind ** 2) / (s_full ** 2 - s_ind ** 2), 0, 1))
    # the part of the population interval that fertility does not explain (migration, mortality): what the simulator's
    # births channel adds from 2025 is about the sum of CBR_PER_TFR x TFR x sd(log TFR) (risks.CBR_PER_TFR)
    from psychohistory.risks import CBR_PER_TFR
    sd_fert = np.nancumsum(CBR_PER_TFR * med_tfr * sig_tfr, 1) - CBR_PER_TFR * med_tfr[:, :1] * sig_tfr[:, :1]
    sig_rest = np.sqrt(np.clip(sig_pop ** 2 - sd_fert ** 2, 0, None))
    # (Hong Kong's 80% lower bound also reaches zero in 2099-2100: carry its last finite spread)
    _ok = np.isfinite(sig_tfr[:, 0])
    for a_ in (sig_pop, sig_rest):
        a_[:] = pd.DataFrame(np.where(np.isfinite(a_), a_, np.nan)).ffill(axis=1).to_numpy()
    # (and a cap of 1.0 log points, an 80% band of x0.28-x3.6: Hong Kong's interval reaches x0.05-x20 by 2098)
    sig_pop, sig_rest = np.minimum(sig_pop, SIG_CAP), np.minimum(sig_rest, SIG_CAP)
    assert np.isfinite(sig_rest[_ok]).all() and np.isfinite(sig_tfr[_ok]).all()
    np.savez_compressed(os.path.join(ROOT, "data", "wpp_prob.npz"), codes=np.array(p.codes), years=years,
                        sig_tfr=sig_tfr, sig_pop=sig_pop, sig_rest=sig_rest, rho=rho, sig_world_2065=sig_w)
    print(f"countries matched: {np.isfinite(sig_tfr[:, 0]).sum()} of {C}; rho {rho:.3f}; world 2065 sd {sig_w:.4f} "
          f"(independent {s_ind:.4f}, fully correlated {s_full:.4f})")
    for c in ("NGA", "IND", "CHN", "USA", "KOR", "NER"):
        if c in ci:
            i = ci[c]
            print(c, "tfr sd 2030/2065", np.round(sig_tfr[i, [6, t65]], 3), "pop sd 2065", round(float(sig_pop[i, t65]), 3),
                  "of which not fertility", round(float(sig_rest[i, t65]), 3))


if __name__ == "__main__":
    main()
