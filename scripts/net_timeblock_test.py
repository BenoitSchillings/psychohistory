"""Does PsychoNet's early stopping leak time? Production members stop each task at the epoch with the best loss on
out-of-bag COUNTRIES, but those countries share years (global waves) with the training rows. Variant "tb" picks the
stopping epoch per task in a first pass trained on in-bag rows up to Y-13 and scored on out-of-bag rows after Y-8
(Y = last training year; 5-year gap so labels cannot overlap), then retrains on all in-bag rows and keeps the
snapshot at that epoch. Dev origins only; paired country bootstrap of the score difference.
python -P scripts/net_timeblock_test.py -> outputs/net_timeblock_test.json"""
import copy
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from torch import nn

from psychohistory import models as M
from psychohistory.models import PsychoNet, Preprocessor, _Net, DEVICE, NET_TUNED
from psychohistory.pipeline import prepare
from psychohistory.evaluate import evaluate, metrics, HEADLINE, DEV_ORIGINS
from psychohistory.features import TASKS
from psychohistory.data import ROOT


class NetTB(PsychoNet):
    uses_years = True

    def fit(self, X, Y, groups, years):
        self.pre = Preprocessor().fit(X)
        Z = torch.as_tensor(self.pre.transform(X), device=DEVICE)
        Yt = torch.as_tensor(Y, dtype=torch.float32, device=DEVICE)
        Mk = torch.isfinite(Yt)
        Yt = torch.nan_to_num(Yt)
        n_task = Y.shape[1]
        w_task = (Mk.sum(0).clamp(min=1).float().reciprocal() * Mk.sum().float() / n_task)
        rng = np.random.default_rng(self.seed)
        ug = np.unique(groups)
        ylast = years.max()

        def train(idx, val, snap_at=None):
            net = _Net(Z.shape[1], n_task, h=self.hidden, p=self.dropout).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=self.epochs * (len(idx) // 512 + 1))
            best, best_ep, snaps = [float("inf")] * n_task, [self.epochs - 1] * n_task, [None] * n_task
            for ep in range(self.epochs):
                net.train()
                perm = idx[torch.randperm(len(idx), device=DEVICE)]
                for b in range(0, len(perm), 512):
                    i = perm[b:b + 512]
                    loss = nn.functional.binary_cross_entropy_with_logits(net(Z[i]), Yt[i], reduction="none")
                    loss = (loss * Mk[i] * w_task).sum() / Mk[i].sum().clamp(min=1)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
                net.eval()
                if snap_at is not None:
                    for k in range(n_task):
                        if snap_at[k] == ep:
                            snaps[k] = copy.deepcopy(net)
                    continue
                with torch.no_grad():
                    l = nn.functional.binary_cross_entropy_with_logits(net(Z[val]), Yt[val], reduction="none")
                    l = ((l * Mk[val]).sum(0) / Mk[val].sum(0).clamp(min=1)).tolist()
                for k in range(n_task):
                    if Mk[val][:, k].sum() > 0 and l[k] < best[k]:
                        best[k], best_ep[k] = l[k], ep
            return best_ep, snaps

        self.nets, self.epochs_chosen = [], []
        for m in range(self.members):
            torch.manual_seed(self.seed * 1000 + m)
            boot = rng.choice(ug, len(ug), replace=True)
            inb = np.concatenate([np.where(groups == g)[0] for g in boot])
            oob = ~np.isin(groups, boot)
            early = torch.as_tensor(inb[years[inb] <= ylast - 13], device=DEVICE)
            val = torch.as_tensor(np.where(oob & (years > ylast - 8))[0], device=DEVICE)
            ep, _ = train(early, val)
            _, snaps = train(torch.as_tensor(inb, device=DEVICE), None, snap_at=ep)
            self.nets.append(snaps)
            self.epochs_chosen.append(ep)
        return self


class Wrap:
    def __init__(self, m): self.m = m; self.uses_years = getattr(m, "uses_years", False)
    def fit(self, X, Y, g, years=None):
        (self.m.fit(X, Y, g, years) if self.uses_years else self.m.fit(X, Y, g)); return self
    def predict(self, X): return self.m.predict(X)


p, base, X, Yl = prepare()
res, cats = {}, {}
for name, fac in (("base", lambda: Wrap(PsychoNet(members=8, **NET_TUNED))), ("tb", lambda: Wrap(NetTB(members=8, **NET_TUNED)))):
    s, out, preds = evaluate(p, X, Yl, fac, DEV_ORIGINS, verbose=True)
    res[name] = {"score": round(s, 4), "tasks": {t: {k: round(float(v), 4) for k, v in out[t].items() if isinstance(v, (int, float))} for t in HEADLINE}}
    cats[name] = [np.concatenate([a[i] for a in preds]) for i in (1, 2, 3, 4)]
    print(name, res[name]["score"], flush=True)

y, _, b, g = cats["base"]
ug = np.unique(g)
rows = {c: np.where(g == c)[0] for c in ug}
sc = lambda pr, idx: np.mean([sum(metrics(y[idx, k], pr[idx, k], b[idx, k]).get(m_, np.nan) for m_ in ("auc", "bss"))
                              for k in [TASKS.index(t) for t in HEADLINE]])
rng = np.random.default_rng(0)
d = []
for _ in range(300):
    idx = np.concatenate([rows[c] for c in rng.choice(ug, len(ug))])
    d.append(sc(cats["tb"][1], idx) - sc(cats["base"][1], idx))
res["diff_tb_minus_base"] = {"point": round(res["tb"]["score"] - res["base"]["score"], 4),
                             "ci90": np.quantile(d, [.05, .95]).round(4).tolist()}
print(res["diff_tb_minus_base"])
json.dump(res, open(ROOT / "outputs" / "net_timeblock_test.json", "w"), indent=1)
