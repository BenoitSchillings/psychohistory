"""A/B hindcast: slow structural variables frozen (old) vs following estimated equations (psychohistory/dynamics.py).

Same origins, same seeds and same trained models; reports the usual simulator hindcast metrics and, for each slow
variable, the share of countries whose actual value at the end year fell inside the simulated 90% / 50% band.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import pipeline as PL
from psychohistory.dynamics import SPEC
from psychohistory.simulate import simulate

p, base, X, Yl = PL.prepare(False)
Xf, Yf, ci, year = PL.flatten(p, X, Yl)
res = []
for O in (1985, 1995, 2005):
    E = min(O + 20, PL.LAST_OBS_YEAR)
    Ytr, tr = PL.train_mask(Yf, year, O)
    net = PL.PsychoNet(members=6, **PL.NET_TUNED).fit(Xf[tr], Ytr[tr], ci[tr])
    growth, ddem = PL.fit_transitions(p, X, base, O)
    for flag in (False, True):
        out = simulate(p, base, net, growth, ddem, O, E, n_scen=2000, demography="extrapolate", slow_dynamics=flag, seed=5)
        m = PL._hindcast_metrics(p, base, out, O, E)
        cov = {}
        for k in SPEC:
            if k not in out["slow_end"]:
                continue
            a = base[k][:, p.yi(E)]
            q = out["slow_end"][k]
            ok = np.isfinite(a) & np.isfinite(q).all(1) & (a != base[k][:, p.yi(O)])
            if ok.sum() < 10:
                continue
            cov[k] = {"in90": round(float(((a >= q[:, 0]) & (a <= q[:, 4]))[ok].mean()), 3),
                      "in50": round(float(((a >= q[:, 1]) & (a <= q[:, 3]))[ok].mean()), 3), "n": int(ok.sum())}
        res.append({"origin": O, "end": E, "slow_dynamics": flag, **{k: round(v, 4) for k, v in m.items() if k not in ("origin", "end")},
                    "slow_coverage": cov})
        print(json.dumps(res[-1]), flush=True)
json.dump(res, open(os.path.join(PL.OUT, "slow_hindcast_ab.json"), "w"), indent=1)
