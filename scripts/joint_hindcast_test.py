"""Does the simulator get the JOINT behaviour right - which countries go to war or slump together, and how a bad
year follows a bad year - or only each country's marginal odds? Hindcasts from the dev origins (models trained only on
data before each origin, demography extrapolated), scored with multivariate proper scores: the energy score (ES) and
the variogram score (VS, p = 0.5, which reacts mostly to dependence). The reference keeps every marginal but breaks the
dependence: each dimension's worlds are shuffled independently. If the simulated co-movement is real, the simulator
beats its shuffled self, above all on VS.
Vectors scored: (a) share of years in conflict, per country; (b) the world conflict share, year by year;
(c) log GDP per person change over the horizon, per country; (d) the same, by region-year.
python -P scripts/joint_hindcast_test.py [first last step tag] -> outputs/joint_hindcast_test[_tag].json"""
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


def vs(X, y, p=0.5):
    X, y = torch.as_tensor(X, device=DEVICE), torch.as_tensor(y, device=DEVICE)
    D = X.shape[1]
    ey = torch.zeros(D, D, device=DEVICE)
    for c in range(0, X.shape[0], 250):
        x = X[c:c + 250]
        ey += (x[:, :, None] - x[:, None, :]).abs().pow(p).sum(0)
    ey /= X.shape[0]
    oy = (y[:, None] - y[None, :]).abs().pow(p)
    return float(((oy - ey) ** 2).triu(1).sum())


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
for O in ORIG:
    E = min(O + H, LAST_OBS_YEAR)
    Ytr, tr = train_mask(Yf, year, O)
    net = PsychoNet(members=8, **NET_TUNED).fit(Xf[tr], Ytr[tr], ci[tr])
    growth, ddem = fit_transitions(p, X, base, O)
    o = simulate(p, base, net, growth, ddem, O, E, n_scen=S, demography="extrapolate", record_paths=S, seed=O)
    cf, gdp = o["paths"]["conflict"], o["paths"]["gdp"]          # [S, C, T]
    i0, i1 = p.yi(O), p.yi(E)
    act_c = base["conflict"][:, i0 + 1:i1 + 1]
    act_g = base["lg"][:, i1] - base["lg"][:, i0]
    vec = {}
    okc = np.isfinite(act_c).all(1)
    vec["conflict_share_by_country"] = (cf[:, okc].mean(-1), act_c[okc].mean(-1))
    vec["world_conflict_by_year"] = (cf[:, okc].mean(1), act_c[okc].mean(0))
    sim_g = np.log(gdp[:, :, -1]) - base["lg"][:, i0][None]
    okg = np.isfinite(act_g) & np.isfinite(sim_g).all(0)
    vec["gdp_change_by_country"] = (sim_g[:, okg], act_g[okg])
    lgp = np.log(gdp) - base["lg"][:, i0][None, :, None]
    act_lp = base["lg"][:, i0 + 1:i1 + 1] - base["lg"][:, i0][:, None]
    okr = okg & np.isfinite(act_lp).all(1) & np.isfinite(lgp).all((0, 2))
    ur = np.unique(regions[okr])
    reg_sim = np.concatenate([lgp[:, okr & (regions == r)].mean(1) for r in ur], 1)
    reg_act = np.concatenate([act_lp[okr & (regions == r)].mean(0) for r in ur])
    vec["gdp_path_by_region_year"] = (reg_sim, reg_act)
    res = {}
    for k, (Xs, y) in vec.items():
        Xs, y = Xs.astype(np.float32), y.astype(np.float32)
        Xsh = shuffled(Xs, rng)
        res[k] = {"dims": int(Xs.shape[1]), "es_sim": es(Xs, y), "es_shuffled": es(Xsh, y), "vs_sim": vs(Xs, y), "vs_shuffled": vs(Xsh, y)}
    out[O] = res
    print(O, E, json.dumps({k: {m: round(v, 4) for m, v in r.items()} for k, r in res.items()}), flush=True)

summ = {}
for k in next(iter(out.values())):
    for m in ("es", "vs"):
        d = [out[O][k][f"{m}_sim"] / out[O][k][f"{m}_shuffled"] - 1 for O in out]
        summ[f"{k}:{m}"] = {"mean_rel_change_vs_shuffled": round(float(np.mean(d)), 4), "origins_sim_better": int(sum(x < 0 for x in d)), "n": len(d)}
print(json.dumps(summ, indent=1))
json.dump({"by_origin": {str(k): v for k, v in out.items()}, "summary": summ}, open(PL.OUT / f"joint_hindcast_test{TAG}.json", "w"), indent=1)
