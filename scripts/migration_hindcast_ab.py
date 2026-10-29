"""A/B hindcast for the refugee-flow mechanism (psychohistory/migration.py): same origins, seeds and models with
migration off and on; also compares the simulated world refugee total with the recorded one."""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from psychohistory import pipeline as PL
from psychohistory.simulate import simulate

p, base, X, Yl = PL.prepare(False)
Xf, Yf, ci, year = PL.flatten(p, X, Yl)
res = []
for O in (1995, 2005):
    E = min(O + 20, PL.LAST_OBS_YEAR)
    Ytr, tr = PL.train_mask(Yf, year, O)
    net = PL.PsychoNet(members=6, **PL.NET_TUNED).fit(Xf[tr], Ytr[tr], ci[tr])
    growth, ddem = PL.fit_transitions(p, X, base, O)
    actual = [float(np.nansum(base["refugees_pc"][:, p.yi(y)] * np.exp(base["lpop"][:, p.yi(y)])) / 1e6)
              for y in range(O + 1, E + 1)]
    for flag in (False, True):
        out = simulate(p, base, net, growth, ddem, O, E, n_scen=2000, demography="extrapolate", migration=flag, seed=5)
        m = PL._hindcast_metrics(p, base, out, O, E)
        row = {"origin": O, "end": E, "migration": flag, **{k: round(v, 4) for k, v in m.items() if k not in ("origin", "end")}}
        if flag:
            q = np.quantile(out["displaced"], [.05, .5, .95], 0)
            row["refugees_world_m"] = {"actual_end": round(actual[-1], 1), "sim_end_q": q[:, -1].round(1).tolist(),
                                       "years_actual_in_90": float(np.mean([(q[0, t] <= a <= q[2, t]) for t, a in enumerate(actual)]))}
        res.append(row)
        print(json.dumps(row), flush=True)
json.dump(res, open(os.path.join(PL.OUT, "migration_hindcast_ab.json"), "w"), indent=1)
