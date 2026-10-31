"""A neural quantile model for country-month fatalities, alone and averaged with the production LightGBM quantile model.
Same design matrix and residual target (log deaths at t + h minus the 3-month mean at t) as cm_fatal; an MLP (2 x 256,
missing-value indicators, inputs standardised) with pinball loss on the same 17 quantiles, 5 seeds averaged.
Blend: rank-by-rank average of the sorted draws (quantile averaging). Dev windows 2010-2017 only (the production
model's cached dev draws, outputs/predcomp_tune_cache/resid3_*.npz); the 2018-2023 and live windows are the test.
python -P scripts/cm_nn_test.py dev                -> outputs/cm_nn_test.json
python -P scripts/cm_nn_test.py run NAME W [json]  (write a challenge submission: NN blended with weight W)"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from torch import nn

from psychohistory import cm_fatal as CF
from predcomp_eval import crps_ensemble

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class NNQuantile(CF.QuantileModel):
    def __init__(self, quants=CF.QUANTS, seeds=5, epochs=6, hidden=256, lr=2e-3):
        self.q = np.asarray(quants)
        self.seeds, self.epochs, self.hidden, self.lr = seeds, epochs, hidden, lr

    def _z(self, X):
        Z = np.nan_to_num((X - self.mu) / self.sd).clip(-8, 8)
        return np.concatenate([Z, np.isnan(X).astype(np.float32)], 1).astype(np.float32)

    def fit(self, X, y):
        self.mu, self.sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
        Z = torch.as_tensor(self._z(X), device=DEV)
        Y = torch.as_tensor(y.astype(np.float32), device=DEV)
        qt = torch.as_tensor(self.q, dtype=torch.float32, device=DEV)
        self.nets = []
        for s in range(self.seeds):
            torch.manual_seed(s)
            net = nn.Sequential(nn.Linear(Z.shape[1], self.hidden), nn.GELU(), nn.Dropout(0.1),
                                nn.Linear(self.hidden, self.hidden), nn.GELU(), nn.Linear(self.hidden, len(self.q))).to(DEV)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
            steps = self.epochs * (len(Z) // 4096 + 1)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=steps)
            for _ in range(self.epochs):
                perm = torch.randperm(len(Z), device=DEV)
                for b in range(0, len(Z), 4096):
                    i = perm[b:b + 4096]
                    e = Y[i, None] - net(Z[i])
                    loss = torch.maximum(qt * e, (qt - 1) * e).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
            net.eval()
            self.nets.append(net)
        return self

    @torch.no_grad()
    def predict_q(self, X, base=None):
        Z = torch.as_tensor(self._z(X), device=DEV)
        Q = torch.stack([n(Z) for n in self.nets]).mean(0).cpu().numpy()
        if base is not None:
            Q = Q + base[:, None]
        return np.maximum.accumulate(np.clip(np.sort(Q, 1), 0, None), 1)


NN_KW = {}


def nn_draws(origin, months, final_through=None):
    """mirror of cm_fatal.train_and_forecast with the neural model"""
    kw = dict(NN_KW)
    onehot, direct = kw.pop("onehot", False), kw.pop("direct", False)
    D = CF.deaths(origin, final_through)
    F = CF.features(D)
    nc = D.shape[0]
    hs = sorted({m - origin for m in months})
    base = lambda ts: np.zeros(nc * len(ts)) if direct else np.concatenate([np.nan_to_num(F[CF.RESID][:, t]) for t in ts])
    oh = lambda X: np.concatenate([X, np.eye(nc, dtype=np.float32)[np.arange(len(X)) % nc]], 1) if onehot else X
    Xp, yp = [], []
    for h in hs:
        ts = list(range(CF.mid(1995, 1), origin - h + 1))
        X, _, _ = CF.design(F, ts, [h])
        Xp.append(oh(X))
        yp.append(np.concatenate([np.log1p(D[:, t + h]) for t in ts]) - base(ts))
    m = NNQuantile(**kw).fit(np.concatenate(Xp), np.concatenate(yp))
    out = {}
    for tm in months:
        Xt, _, _ = CF.design(F, [origin], [tm - origin])
        out[tm] = m.sample(oh(Xt), base=base([origin]))
    return out


def blend(a, b, w):
    return np.expm1((1 - w) * np.log1p(np.sort(a, 1)) + w * np.log1p(np.sort(b, 1)))


def dev(tag=""):
    truth = CF.deaths(CF.mid(2023, 12))
    cache = os.path.join(ROOT, "outputs", "predcomp_tune_cache")
    ws = (0.0, 0.25, 0.5, 0.75, 1.0)
    tot = {w: 0.0 for w in ws}
    n = 0
    by_year = {}
    for y in range(2010, 2018):
        z = np.load(os.path.join(cache, f"resid3_{y}.npz"))
        g = {int(k): z[k].astype(float) for k in z.files}
        o = CF.mid(y - 1, 10)
        nd = nn_draws(o, sorted(g))
        yr = {w: 0.0 for w in ws}
        for m in g:
            for w in ws:
                c = crps_ensemble(truth[:, m], np.clip(blend(g[m], nd[m], w), 0, None))
                tot[w] += c.sum()
                yr[w] += c.mean() / len(g)
            n += len(truth)
        by_year[y] = {str(w): round(v, 3) for w, v in yr.items()}
        print(y, by_year[y], flush=True)
    res = {"dev_crps": {str(w): round(tot[w] / n, 4) for w in ws}, "by_year": by_year}
    print(res["dev_crps"])
    res["nn_kw"] = NN_KW
    json.dump(res, open(os.path.join(ROOT, "outputs", f"cm_nn_test{tag}.json"), "w"), indent=1)


def run(name, w):
    """challenge submission: production LightGBM draws (from seldon_v5 live / seldon_v4 history) blended with NN draws"""
    import glob
    import pandas as pd
    from predcomp_run import write
    cs = CF.countries()
    windows = [(f"Y{y}", CF.mid(y - 1, 10), [CF.mid(y, m) for m in range(1, 13)], None, "seldon_v4") for y in range(2018, 2024)]
    live = [CF.mid(2024, m) for m in range(7, 13)] + [CF.mid(2025, m) for m in range(1, 7)]
    windows.append(("LIVE", CF.mid(2024, 4), live, CF.mid(2023, 12), "seldon_v5"))
    for wname, origin, months, ft, src in windows:
        nd = nn_draws(origin, months, ft)
        g = {}
        for wy in (["Y2024", "Y2025"] if wname == "LIVE" else [wname]):
            f = glob.glob(os.path.join(ROOT, "outputs", "predcomp", src, "cm", f"window={wy}", "*.parquet"))[0]
            P = pd.read_parquet(f).reset_index().pivot_table(index=["month_id", "country_id"], columns="draw", values="outcome")
            for m in P.index.get_level_values(0).unique():
                g[int(m)] = P.loc[m].reindex(cs.views_country_id).to_numpy(float)
        dr = {m: blend(g[m], nd[m], w) for m in g}
        if wname == "LIVE":
            write(name, "Y2024", {m: v for m, v in dr.items() if m <= CF.mid(2024, 12)})
            write(name, "Y2025", {m: v for m, v in dr.items() if m > CF.mid(2024, 12)})
        else:
            write(name, wname, dr)
        print("wrote", name, wname, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "dev":
        NN_KW.update(json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})
        dev(f"_{sys.argv[2]}" if len(sys.argv) > 2 else "")
    else:
        NN_KW.update(json.loads(sys.argv[4]) if len(sys.argv) > 4 else {})
        run(sys.argv[2], float(sys.argv[3]))
