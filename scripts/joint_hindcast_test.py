"""Does the simulator get the JOINT behaviour right - which countries go to war or slump together, and how a bad
year follows a bad year - or only each country's marginal odds? Hindcasts from the dev origins (models trained only on
data before each origin, demography extrapolated), scored with multivariate proper scores: the energy score (ES) and
the variogram score (VS, p = 0.5, which reacts mostly to dependence). The reference keeps every marginal but breaks the
dependence: each dimension's worlds are shuffled independently. If the simulated co-movement is real, the simulator
beats its shuffled self, above all on VS.
Vectors scored: (a) share of years in conflict, per country; (b) the world conflict share, year by year;
(c) log GDP per person change over the horizon, per country; (d) the same, by region-year.
Since 2026-11-05 the forecast's configuration (pipeline.build_sim_env, family draw, world layer on with AI off), and
a distance-weighted variogram (vsw) for (a) and (c) next to the unweighted one.
python -P scripts/joint_hindcast_test.py [first last step tag ['{"VAR": {"simulate.CONST": v}, ...}']]
    -> outputs/joint_hindcast_test[_tag][_VAR].json (variants run on the same network at each origin)"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from psychohistory import pipeline as PL
from psychohistory.backtest import flatten, train_mask
from psychohistory.models import PsychoNet, NET_TUNED, DEVICE
from psychohistory.simulate import fit_transitions, simulate
from psychohistory.data import LAST_OBS_YEAR

S, H = 4000, 15


def es(X, y):
    """energy score; X [S, D] ensemble, y [D]"""
    X, y = torch.as_tensor(X, device=DEVICE), torch.as_tensor(y, device=DEVICE)
    a = (X - y).norm(dim=1).mean()
    sub = X[:1500]
    b = torch.cdist(sub, sub).mean() / 2
    return float(a - b)


def vs(X, y, p=0.5, w=None):
    """variogram score; w [D, D] pair weights (None: every pair alike). Scheuerer & Hamill weight pairs by proximity,
    so that the many distant pairs do not drown the co-movement of neighbours"""
    X, y = torch.as_tensor(X, device=DEVICE), torch.as_tensor(y, device=DEVICE)
    D = X.shape[1]
    ey = torch.zeros(D, D, device=DEVICE)
    for c in range(0, X.shape[0], 250):
        x = X[c:c + 250]
        ey += (x[:, :, None] - x[:, None, :]).abs().pow(p).sum(0)
    ey /= X.shape[0]
    oy = (y[:, None] - y[None, :]).abs().pow(p)
    d2 = (oy - ey) ** 2
    if w is not None:
        d2 = d2 * torch.as_tensor(w, dtype=d2.dtype, device=DEVICE)
    return float(d2.triu(1).sum())


def shuffled(X, rng):
    return np.stack([X[rng.permutation(len(X)), j] for j in range(X.shape[1])], 1)


p, base, X, Yl = PL.prepare(False)
Xf, Yf, ci, year = flatten(p, X, Yl)
rng = np.random.default_rng(0)
regions = np.asarray(p.regions)
out = {}
A = sys.argv[1:]
ORIG = range(int(A[0]), int(A[1]) + 1, int(A[2])) if A else range(1975, 2010, 5)
TAG = f"_{A[3]}" if len(A) > 3 else ""
import importlib
VARIANTS = json.loads(A[4]) if len(A) > 4 else {"": {}}
from psychohistory import features as FT, interstate as INTER, simulate as SIM_, world as WM
FT.set_spatial(p.codes)
Wsp = FT.SPATIAL["W"].cpu().numpy()                      # exp(-km / 800) between country centroids
df_int = INTER.build(verbose=False)
# every variant starts from the module defaults (the variant loop used to leave its settings in place for the next)
DEFAULTS = {k_: getattr(importlib.import_module(f"psychohistory.{k_.rsplit('.', 1)[0]}"), k_.rsplit(".", 1)[1])
            for c_ in VARIANTS.values() for k_ in c_}
for O in ORIG:
    E = min(O + H, LAST_OBS_YEAR)
    # the forecast's simulator network and family draw, with the world layer on (interstate wars, cascades), AI off
    net = PL.build_sim_env(p, X, Xf, Yf, ci, year, O, base=base)
    growth, ddem = fit_transitions(p, X, base, O)
    dy = INTER.sim_inputs(INTER.run(verbose=False, origin=O, backtests=False, df=df_int))
    for VAR, consts in VARIANTS.items():
      for k_, v_ in {**DEFAULTS, **consts}.items():
          mod_, nm_ = k_.rsplit(".", 1)
          setattr(importlib.import_module(f"psychohistory.{mod_}"), nm_, v_)
      rng = np.random.default_rng(O)
      o = simulate(p, base, net, growth, ddem, O, E, n_scen=S, demography="extrapolate", record_paths=S, seed=O,
                   world=WM.WorldConfig(ai=WM.AI_SCENARIOS["none"], sample_params=True), dyads=dy)
      cf, gdp = o["paths"]["conflict"], o["paths"]["gdp"]          # [S, C, T]
      i0, i1 = p.yi(O), p.yi(E)
      act_c = base["conflict"][:, i0 + 1:i1 + 1]
      act_g = base["lg"][:, i1] - base["lg"][:, i0]
      vec = {}
      okc = np.isfinite(act_c).all(1)
      vec["conflict_share_by_country"] = (cf[:, okc].mean(-1), act_c[okc].mean(-1), Wsp[np.ix_(okc, okc)])
      vec["world_conflict_by_year"] = (cf[:, okc].mean(1), act_c[okc].mean(0), None)
      sim_g = np.log(gdp[:, :, -1]) - base["lg"][:, i0][None]
      okg = np.isfinite(act_g) & np.isfinite(sim_g).all(0)
      vec["gdp_change_by_country"] = (sim_g[:, okg], act_g[okg], Wsp[np.ix_(okg, okg)])
      lgp = np.log(gdp) - base["lg"][:, i0][None, :, None]
      act_lp = base["lg"][:, i0 + 1:i1 + 1] - base["lg"][:, i0][:, None]
      okr = okg & np.isfinite(act_lp).all(1) & np.isfinite(lgp).all((0, 2))
      ur = np.unique(regions[okr])
      reg_sim = np.concatenate([lgp[:, okr & (regions == r)].mean(1) for r in ur], 1)
      reg_act = np.concatenate([act_lp[okr & (regions == r)].mean(0) for r in ur])
      vec["gdp_path_by_region_year"] = (reg_sim, reg_act, None)
      res = {}
      for k, (Xs, y, wk) in vec.items():
          Xs, y = Xs.astype(np.float32), y.astype(np.float32)
          Xsh = shuffled(Xs, rng)
          res[k] = {"dims": int(Xs.shape[1]), "es_sim": es(Xs, y), "es_shuffled": es(Xsh, y), "vs_sim": vs(Xs, y), "vs_shuffled": vs(Xsh, y)}
          if wk is not None:                       # neighbours' co-movement (distance-weighted pairs)
              res[k]["vsw_sim"], res[k]["vsw_shuffled"] = vs(Xs, y, w=wk), vs(Xsh, y, w=wk)
      out.setdefault(VAR, {})[O] = res
      print(VAR, O, E, json.dumps({k: {m: round(v, 4) for m, v in r.items()} for k, r in res.items()}), flush=True)

for VAR, outv in out.items():
    summ = {}
    for k in next(iter(outv.values())):
        for m in ("es", "vs", "vsw"):
            if f"{m}_sim" not in next(iter(outv.values()))[k]:
                continue
            d = [outv[O][k][f"{m}_sim"] / outv[O][k][f"{m}_shuffled"] - 1 for O in outv]
            summ[f"{k}:{m}"] = {"mean_rel_change_vs_shuffled": round(float(np.mean(d)), 4), "origins_sim_better": int(sum(x < 0 for x in d)), "n": len(d)}
    print(VAR, json.dumps(summ, indent=1))
    json.dump({"by_origin": {str(k): v for k, v in outv.items()}, "summary": summ},
              open(PL.OUT / f"joint_hindcast_test{TAG}{'_' + VAR if VAR else ''}.json", "w"), indent=1)
