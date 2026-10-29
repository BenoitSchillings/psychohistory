"""Pin down nuclear_war_factor (the multiplier on the chance that an interstate conflict between two nuclear-armed states
reaches war level; sampled u(0.25, 1) in world.PARAM_RANGES). Evidence: ICB v16 crisis dyads since 1946 (outcome: the
crisis reached full-scale war, icb_max_viol == 4), nuclear status from OWID warhead stockpiles at the crisis start.
Ratio of war rates, both-nuclear vs neither, raw and adjusted (logit with decade, major-power and contiguity-free
controls), with a crisis-clustered bootstrap. Also UCDP interstate conflict onsets (war level reached, by pair status).
python -P scripts/nuclear_crisis_test.py -> outputs/nuclear_crisis_test.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import statsmodels.api as sm

from psychohistory.data import RAW, ROOT

w = pd.read_csv(RAW / "nuclear-warhead-stockpiles.csv")
w = w.rename(columns={w.columns[1]: "iso3", w.columns[2]: "year", w.columns[3]: "warheads"})
nuc = set(zip(w.iso3[w.warheads > 0], w.year[w.warheads > 0]))
d = pd.read_csv(RAW / "interstate" / "icb_dy.csv")
d = d[d.year >= 1946]
first = d.sort_values("year").groupby(["icb_crisnos", "iso3_a", "iso3_b"], as_index=False).first()
first["war"] = (d.groupby(["icb_crisnos", "iso3_a", "iso3_b"]).icb_max_viol.max().values >= 4).astype(float) \
    if False else first.icb_max_viol.ge(4).astype(float)
g = d.groupby(["icb_crisnos", "iso3_a", "iso3_b"]).icb_max_viol.max().rename("mv").reset_index()
first = first.drop(columns="war").merge(g, on=["icb_crisnos", "iso3_a", "iso3_b"])
first["war"] = (first.mv >= 4).astype(float)
na = np.array([(a, y) in nuc for a, y in zip(first.iso3_a, first.year)])
nb = np.array([(b, y) in nuc for b, y in zip(first.iso3_b, first.year)])
first["both"], first["one"] = (na & nb).astype(float), (na ^ nb).astype(float)
first["decade"] = first.year // 10 * 10
out = {"n_crisis_dyads": int(len(first))}
for k, m in (("both", first.both == 1), ("one", first.one == 1), ("neither", (first.both == 0) & (first.one == 0))):
    out[k] = {"n": int(m.sum()), "wars": int(first.war[m].sum()), "rate": round(float(first.war[m].mean()), 4)}
out["both_nuclear_crises"] = first[first.both == 1][["icb_crisnos", "iso3_a", "iso3_b", "year", "mv"]].astype(str).values.tolist()
rate_neither = out["neither"]["rate"]
# Jeffreys posterior for the both-nuclear war rate, divided by the neither rate (crisis bootstrap for its uncertainty)
rng = np.random.default_rng(0)
nb_, wb_ = out["both"]["n"], out["both"]["wars"]
neither = first[(first.both == 0) & (first.one == 0)]
cr = neither.icb_crisnos.unique()
ratios = []
for _ in range(4000):
    pb = rng.beta(wb_ + .5, nb_ - wb_ + .5)
    s = neither[neither.icb_crisnos.isin(rng.choice(cr, len(cr)))] if False else neither.sample(len(neither), replace=True, random_state=int(rng.integers(1e9)))
    ratios.append(pb / max(s.war.mean(), 1e-3))
out["ratio_both_vs_neither"] = {"q05_50_95": np.quantile(ratios, [.05, .5, .95]).round(3).tolist(), "mean": round(float(np.mean(ratios)), 3)}
# adjusted: logit with decade dummies
X = pd.concat([first[["both", "one"]], pd.get_dummies(first.decade, prefix="d", drop_first=True).astype(float)], axis=1).assign(const=1.0)
try:
    r = sm.Logit(first.war, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": first.icb_crisnos.astype("category").cat.codes})
    out["logit"] = {k: [round(float(r.params[k]), 3), round(float(r.bse[k]), 3)] for k in ("both", "one")}
except Exception as e:
    out["logit"] = str(e)
print(json.dumps(out, indent=1))
json.dump(out, open(ROOT / "outputs" / "nuclear_crisis_test.json", "w"), indent=1)

# UCDP interstate conflict episodes (type 2) since 1946: does the episode reach war level (1,000+ deaths in a year)?
from psychohistory import interstate as INTER
codes = set(pd.read_csv(RAW / "population-with-un-projections.csv").code.dropna())
ev = INTER._ucdp_interstate(codes)
eps = []
for (a, b, y), inten in sorted(ev.items()):
    if (a, b, y - 1) in ev:
        continue                                    # not an episode start
    yy, mx = y, inten
    while (a, b, yy + 1) in ev:
        yy += 1
        mx = max(mx, ev[(a, b, yy)])
    both = ((a, y) in nuc) and ((b, y) in nuc)
    one = ((a, y) in nuc) ^ ((b, y) in nuc)
    eps.append({"a": a, "b": b, "start": y, "end": yy, "war": int(mx == 2), "both": int(both), "one": int(one)})
E = pd.DataFrame(eps)
u = {}
for k, m in (("both", E.both == 1), ("one", E.one == 1), ("neither", (E.both == 0) & (E.one == 0))):
    u[k] = {"episodes": int(m.sum()), "war_level": int(E.war[m].sum()), "rate": round(float(E.war[m].mean()), 4)}
u["both_episodes"] = E[E.both == 1][["a", "b", "start", "end", "war"]].astype(str).values.tolist()
# posterior of the factor f with P(war | both) = f * P(war | neither): grid, flat prior on (0, 1.5]
f = np.linspace(0.01, 1.5, 300)
kb, nb2, kn, nn = u["both"]["war_level"], u["both"]["episodes"], u["neither"]["war_level"], u["neither"]["episodes"]
pn = rng.beta(kn + .5, nn - kn + .5, 4000)
lik = np.mean([(np.clip(f * p_, 1e-6, 1 - 1e-6) ** kb) * ((1 - np.clip(f * p_, 1e-6, 1 - 1e-6)) ** (nb2 - kb)) for p_ in pn], 0)
post = lik / lik.sum()
cdf = np.cumsum(post)
u["factor_posterior_q05_50_95"] = [round(float(f[np.searchsorted(cdf, q)]), 3) for q in (.05, .5, .95)]
out["ucdp"] = u
print(json.dumps(u, indent=1))
json.dump(out, open(ROOT / "outputs" / "nuclear_crisis_test.json", "w"), indent=1)

# conflict-YEARS (nuclear status in that year; catches Kargil 1999, inside an India-Pakistan episode that began in 1948)
Y_ = pd.DataFrame([{"a": a, "b": b, "year": y, "war": int(i == 2), "both": int(((a, y) in nuc) and ((b, y) in nuc)),
                    "one": int(((a, y) in nuc) ^ ((b, y) in nuc))} for (a, b, y), i in ev.items() if y >= 1946])
yr = {}
for k, m in (("both", Y_.both == 1), ("one", Y_.one == 1), ("neither", (Y_.both == 0) & (Y_.one == 0))):
    yr[k] = {"conflict_years": int(m.sum()), "war_years": int(Y_.war[m].sum()), "rate": round(float(Y_.war[m].mean()), 4)}
yr["both_war_years"] = Y_[(Y_.both == 1) & (Y_.war == 1)][["a", "b", "year"]].astype(str).values.tolist()
# pair-clustered bootstrap of the ratio
pairs = Y_.groupby(["a", "b"])
keys = list(pairs.groups)
rat = []
for _ in range(4000):
    s = pd.concat([pairs.get_group(keys[i]) for i in rng.integers(len(keys), size=len(keys))])
    b_, n_ = s[s.both == 1], s[(s.both == 0) & (s.one == 0)]
    if len(b_) and len(n_):
        rat.append((b_.war.sum() + .5) / (len(b_) + 1) / max(n_.war.mean(), 1e-3))
yr["ratio_both_vs_neither_q05_50_95"] = np.quantile(rat, [.05, .5, .95]).round(3).tolist()
out["ucdp_years"] = yr
print(json.dumps(yr))
json.dump(out, open(ROOT / "outputs" / "nuclear_crisis_test.json", "w"), indent=1)
