"""Predictive models of structural crises.

* Logit      - regularised logistic regression (interpretable baseline)
* GBM        - LightGBM gradient boosting (non-linear, native missing values)
* PsychoNet  - multi-task neural deep ensemble on the GPU; each member is trained
               on a country-level bootstrap so the spread across members is an
               estimate of epistemic uncertainty.
* Naive      - Fearon & Laitin style benchmark (wealth, size, conflict history)
"""
from __future__ import annotations

import copy
import warnings

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from .features import FEATURES, TASKS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NAIVE_FEATURES = ["log_gdp", "log_pop", "conflict_now", "conflict_past_10y", "log_years_since_conflict"]


class Preprocessor:
    """standardise, clip, zero-impute and append missingness indicators."""

    def fit(self, X: np.ndarray):
        self.mu = np.nanmean(X, 0)
        self.sd = np.nanstd(X, 0) + 1e-6
        self.mu = np.nan_to_num(self.mu)
        self.sd = np.where(np.isfinite(self.sd), self.sd, 1.0)
        self._t = None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = np.clip((X - self.mu) / self.sd, -6, 6)
        m = np.isnan(z)
        return np.concatenate([np.where(m, 0, z), m], -1).astype(np.float32)

    def transform_t(self, X: torch.Tensor) -> torch.Tensor:
        if self._t is None or self._t[0].device != X.device:
            self._t = (torch.as_tensor(self.mu, dtype=X.dtype, device=X.device),
                       torch.as_tensor(self.sd, dtype=X.dtype, device=X.device))
        z = ((X - self._t[0]) / self._t[1]).clamp(-6, 6)
        m = torch.isnan(z)
        return torch.cat([torch.where(m, torch.zeros_like(z), z), m.to(z.dtype)], -1)


def task_blocked(k):
    """indices (in FEATURES) of the features task k may not use (features.TASK_ONLY)"""
    from .features import TASK_ONLY
    return [FEATURES.index(f) for f, ts in TASK_ONLY.items() if f in FEATURES and TASKS[k] not in ts]


def _blank(X, cols):
    """a copy of X with the given feature columns missing (numpy)"""
    if not cols:
        return X
    X = np.array(X, copy=True)
    X[:, cols] = np.nan
    return X


class _Masked(nn.Module):
    """a network that sees some inputs blanked (zeroed, as a missing value after Preprocessor, flag off)"""

    def __init__(self, inner, mask):
        super().__init__()
        self.inner = inner
        self.register_buffer("mask", mask)

    def forward(self, x):
        return self.inner(x * self.mask)


# ---------------------------------------------------------------------------

class LogitModel:
    name = "Logit"

    def __init__(self, features=FEATURES, C=0.05):
        self.cols = [FEATURES.index(f) for f in features]
        self.C = C

    def fit(self, X, Y):
        if self.cols == list(range(len(FEATURES))) and X.shape[1] > len(FEATURES):
            self.cols = list(range(X.shape[1]))     # candidate columns appended by the test harnesses reach the logit too
        X = X[:, self.cols]
        self.pre = Preprocessor().fit(X)
        Z = self.pre.transform(X)
        self.models = {}
        for k in range(Y.shape[1]):
            ok = np.isfinite(Y[:, k])
            if ok.sum() > 50 and 0 < Y[ok, k].mean() < 1:
                self.models[k] = LogisticRegression(C=self.C, max_iter=3000).fit(self._zk(Z[ok], k), Y[ok, k])
        return self

    def _zk(self, Z, k):
        """Z with task k's blocked features zeroed (value and missing flag)"""
        pos = [self.cols.index(c) for c in task_blocked(k) if c in self.cols]
        if not pos:
            return Z
        Z = Z.copy()
        Z[:, pos] = 0
        Z[:, [len(self.cols) + q for q in pos]] = 0
        return Z

    def predict(self, X):
        Z = self.pre.transform(X[:, self.cols])
        out = np.full((len(X), len(TASKS)), np.nan)
        for k, m in self.models.items():
            out[:, k] = m.predict_proba(self._zk(Z, k))[:, 1]
        return out

    def coefficients(self, k):
        m = self.models[k]
        names = [FEATURES[c] for c in self.cols]
        return dict(zip(names, m.coef_[0][: len(names)]))


class NaiveModel(LogitModel):
    name = "Naive (Fearon-Laitin)"

    def __init__(self):
        super().__init__(NAIVE_FEATURES, C=1.0)


GBM_DEFAULT = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=40, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0)


class GBMModel:
    name = "GBM"

    def __init__(self, params=None, per_task=None):
        """params: LightGBM settings for every task; per_task: {task name: settings} overrides."""
        self.params = {**GBM_DEFAULT, **(params or {})}
        self.per_task = per_task or {}

    def fit(self, X, Y):
        self.models = {}
        for k in range(Y.shape[1]):
            ok = np.isfinite(Y[:, k])
            if ok.sum() > 50 and 0 < Y[ok, k].mean() < 1:
                prm = {**self.params, **self.per_task.get(TASKS[k], {})}
                m = lgb.LGBMClassifier(**prm, verbose=-1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.models[k] = m.fit(_blank(X[ok], task_blocked(k)), Y[ok, k])
        return self

    def predict(self, X):
        out = np.full((len(X), len(TASKS)), np.nan)
        for k, m in self.models.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[:, k] = m.predict_proba(_blank(X, task_blocked(k)))[:, 1]
        return out

    def importance(self, k):
        m = self.models[k]
        g = m.booster_.feature_importance("gain")
        return dict(zip(FEATURES, g / g.sum()))


# ---------------------------------------------------------------------------

class _Net(nn.Module):
    def __init__(self, d_in, d_out, h=192, p=0.25):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d_in, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(p),
            nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(p),
            nn.Linear(h, d_out))

    def forward(self, x):
        return self.f(x)


class PsychoNet:
    """Multi-task deep ensemble; members trained on country-bootstrap resamples."""
    name = "PsychoNet"

    def __init__(self, members=10, epochs=20, lr=3e-3, wd=3e-3, seed=0, hidden=192, dropout=0.25):
        self.members, self.epochs, self.lr, self.wd, self.seed = members, epochs, lr, wd, seed
        self.hidden, self.dropout = hidden, dropout

    def fit(self, X, Y, groups, sample_weight=None):
        """`sample_weight` (optional, per row): e.g. a recency weight; None = every row counts the same"""
        self.pre = Preprocessor().fit(X)
        Z = torch.as_tensor(self.pre.transform(X), device=DEVICE)
        self._sw = None if sample_weight is None else torch.as_tensor(np.asarray(sample_weight), dtype=torch.float32, device=DEVICE)
        # tasks that may not see some features (features.TASK_ONLY) get the snapshots of a pass trained with those
        # inputs blanked, wrapped so that they are blanked at prediction too; one pass per distinct blocked set
        F = X.shape[1]
        sets = {}
        for k in range(Y.shape[1]):
            sets.setdefault(tuple(task_blocked(k)), []).append(k)
        passes = {}
        for b in sets:
            mask = torch.ones(Z.shape[1], device=DEVICE)
            if b:
                mask[list(b)] = 0
                mask[[F + c for c in b]] = 0
            passes[b] = (mask, self._train(Z * mask if b else Z, Y, groups))
        self.nets = []
        for m in range(self.members):
            snaps = [None] * Y.shape[1]
            for b, ks in sets.items():
                mask, nets = passes[b]
                for k in ks:
                    snaps[k] = _Masked(nets[m][k], mask) if b else nets[m][k]
            self.nets.append(snaps)
        return self

    def _train(self, Z, Y, groups):
        Yt = torch.as_tensor(Y, dtype=torch.float32, device=DEVICE)
        M = torch.isfinite(Yt)
        Yt = torch.nan_to_num(Yt)
        n_task = Y.shape[1]
        # balance tasks: weight each task by 1/(observed rows)
        w_task = (M.sum(0).clamp(min=1).float().reciprocal() * M.sum().float() / n_task)
        rng = np.random.default_rng(self.seed)
        ug = np.unique(groups)
        nets = []  # per member: list of per-task snapshots
        for m in range(self.members):
            torch.manual_seed(self.seed * 1000 + m)
            boot = rng.choice(ug, len(ug), replace=True)
            idx = torch.as_tensor(np.concatenate([np.where(groups == g)[0] for g in boot]), device=DEVICE)
            oob = torch.as_tensor(np.where(~np.isin(groups, boot))[0], device=DEVICE)  # out-of-bag countries
            net = _Net(Z.shape[1], n_task, h=self.hidden, p=self.dropout).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=self.epochs * (len(idx) // 512 + 1))
            best = [float("inf")] * n_task
            snaps = [None] * n_task
            for ep in range(self.epochs):
                net.train()
                perm = idx[torch.randperm(len(idx), device=DEVICE)]
                for b in range(0, len(perm), 512):
                    i = perm[b:b + 512]
                    loss = nn.functional.binary_cross_entropy_with_logits(net(Z[i]), Yt[i], reduction="none")
                    if self._sw is None:
                        loss = (loss * M[i] * w_task).sum() / M[i].sum().clamp(min=1)
                    else:
                        sw = self._sw[i][:, None]
                        loss = (loss * M[i] * w_task * sw).sum() / (M[i] * sw).sum().clamp(min=1e-6)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
                # per-task early stopping on out-of-bag countries
                net.eval()
                with torch.no_grad():
                    l = nn.functional.binary_cross_entropy_with_logits(net(Z[oob]), Yt[oob], reduction="none")
                    l = ((l * M[oob]).sum(0) / M[oob].sum(0).clamp(min=1)).tolist()
                for k in range(n_task):
                    if l[k] < best[k]:
                        best[k] = l[k]
                        snaps[k] = copy.deepcopy(net)
            nets.append(snaps)
        return nets

    @torch.no_grad()
    def predict_members_t(self, X: torch.Tensor, tasks=None) -> torch.Tensor:
        """X raw features [..., F] (torch) -> probabilities [members, ..., T]."""
        Z = self.pre.transform_t(X)
        tasks = range(len(TASKS)) if tasks is None else tasks
        return torch.stack([torch.stack([torch.sigmoid(snaps[k](Z)[..., k]) for k in tasks], -1)
                            for snaps in self.nets])

    def predict_members(self, X):
        out = []
        for b in range(0, len(X), 65536):
            out.append(self.predict_members_t(torch.as_tensor(X[b:b + 65536], device=DEVICE)).cpu().numpy())
        return np.concatenate(out, 1)

    def predict(self, X):
        return self.predict_members(X).mean(0)


class ForestModel:
    """Extremely randomised trees per task on standardised, zero-imputed features + missing flags."""
    name = "Forest"

    def __init__(self, n_estimators=500, min_samples_leaf=20, max_features=0.3, seed=0):
        self.kw = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,
                       n_jobs=-1, random_state=seed)

    def fit(self, X, Y):
        from sklearn.ensemble import ExtraTreesClassifier
        self.pre = Preprocessor().fit(X)
        Z = self.pre.transform(X)
        self.models = {}
        for k in range(Y.shape[1]):
            ok = np.isfinite(Y[:, k])
            if ok.sum() > 50 and 0 < Y[ok, k].mean() < 1:
                self.models[k] = ExtraTreesClassifier(**self.kw).fit(self._zk(Z[ok], k), Y[ok, k])
        return self

    def _zk(self, Z, k):
        """Z with task k's blocked features zeroed (value and missing flag), as LogitModel"""
        b = task_blocked(k)
        if not b:
            return Z
        Z = Z.copy()
        F = len(FEATURES)
        Z[:, b] = 0
        Z[:, [F + c for c in b]] = 0
        return Z

    def predict(self, X):
        Z = self.pre.transform(X)
        out = np.full((len(X), len(TASKS)), np.nan)
        for k, m in self.models.items():
            out[:, k] = m.predict_proba(self._zk(Z, k))[:, 1]
        return out


class Ensemble:
    """'Seldon' ensemble of Logit, GBM and PsychoNet. `weights`: {task: {model: w}} for a weighted
    average of probabilities (equal weights by default); a model with weight 0 everywhere is skipped."""
    name = "Seldon ensemble"

    def __init__(self, members=10, gbm_params=None, gbm_per_task=None, net_params=None, weights=None,
                 use_logit=True):
        self.parts = ([LogitModel()] if use_logit else []) + [GBMModel(gbm_params, gbm_per_task),
                                                              PsychoNet(members=members, **(net_params or {}))]
        self.weights = weights

    def fit(self, X, Y, groups):
        for m in self.parts:
            m.fit(X, Y, groups) if isinstance(m, PsychoNet) else m.fit(X, Y)
        return self

    def predict_parts(self, X):
        return {m.name: m.predict(X) for m in self.parts}

    def predict(self, X):
        parts = self.predict_parts(X)
        if not self.weights:
            return np.nanmean(np.stack(list(parts.values())), 0)
        out = np.nanmean(np.stack(list(parts.values())), 0)
        for t, w in self.weights.items():
            k = TASKS.index(t)
            tot = sum(w.get(n, 0) for n in parts)
            out[:, k] = sum(w.get(n, 0) * parts[n][:, k] for n in parts) / tot
        return out


# ---------------------------------------------------------------------------
# Production ensemble (after the October 2026 improvement round)

GBM_TUNED = dict(num_leaves=7, min_child_samples=80, learning_rate=0.01, n_estimators=600, colsample_bytree=0.4,
                 reg_lambda=30, subsample=0.6)
GBM_TUNED_TASK = {"coup_5y": dict(num_leaves=4, min_child_samples=80, learning_rate=0.01, n_estimators=300,
                                  colsample_bytree=0.2, reg_lambda=10, subsample=0.6)}
NET_TUNED = dict(hidden=384, dropout=0.25, epochs=20, lr=0.01, wd=0.001)


class QuantileNet:
    """PsychoNet on rank-gauss transformed inputs (robust to heavy tails)."""
    name = "PsychoNet (quantile)"

    def __init__(self, members=8, seed=0):
        self.net = PsychoNet(members=members, seed=seed, **NET_TUNED)

    def fit(self, X, Y, groups):
        from sklearn.preprocessing import QuantileTransformer
        # NaNs are ignored when estimating the quantiles and stay NaN after transform
        self.qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=100000).fit(X)
        self.net.fit(self._q(X), Y, groups)
        return self

    def _q(self, X):
        return self.qt.transform(X)

    def predict(self, X):
        return self.net.predict(self._q(X))


def load_combine():
    """per-task combination rule, chosen on the development backtest (see scripts/search_weights2.py)."""
    import json
    from pathlib import Path
    f = Path(__file__).with_name("ensemble_config.json")
    return json.loads(f.read_text()) if f.exists() else {}


class SeldonEnsemble:
    """Families of models, each trained on its own history window, combined per task.

    families: G (tuned GBM, 3 seeds, from 1955), G60 (tuned GBM, 2 seeds, from 1860), N (tuned PsychoNet,
    2 seeds), Nq (quantile PsychoNet, 2 seeds), F (extra trees), L (logit).
    combine: {task: {"w": {family: weight}, "mode": "prob" | "logit", "platt": [a, b] (optional)}}; tasks not listed use the
    default (equal 4-family nonlinear ensemble: G + G60 + N + F). Rows earlier than a family's window are dropped for that family; the
    `years` argument gives each training row's year."""
    name = "Seldon ensemble"
    uses_years = True
    DEFAULT = {"w": {"G": 0.25, "G60": 0.25, "N": 0.25, "F": 0.25}, "mode": "prob"}

    def __init__(self, combine=None, members=8, families=None):
        self.combine = combine or {}
        self.members = members
        mk = {
            "G": lambda s: GBMModel({**GBM_TUNED, "random_state": s}, GBM_TUNED_TASK),
            "G60": lambda s: GBMModel({**GBM_TUNED, "random_state": s}, GBM_TUNED_TASK),
            "N": lambda s: PsychoNet(members=members, seed=s * 5, **NET_TUNED),
            "Nq": lambda s: QuantileNet(members=members, seed=s * 5),
            "F": lambda s: ForestModel(seed=s),
            "L": lambda s: LogitModel(),
        }
        seeds = {"G": 3, "G60": 2, "N": 2, "Nq": 2, "F": 1, "L": 1}
        need = families or sorted({f for c in list(self.combine.values()) + [self.DEFAULT] for f, w in c["w"].items() if w})
        self.fam = {f: [mk[f](s) for s in range(seeds[f])] for f in need}
        self.first_year = {f: (1860 if f == "G60" else 1955) for f in need}

    def fit(self, X, Y, groups, years):
        for f, ms in self.fam.items():
            r = years >= self.first_year[f]
            for m in ms:
                m.fit(X[r], Y[r], groups[r]) if isinstance(m, (PsychoNet, QuantileNet)) else m.fit(X[r], Y[r])
        return self

    def predict_families(self, X):
        return {f: np.mean([m.predict(X) for m in ms], 0) for f, ms in self.fam.items()}

    def predict(self, X):
        P = self.predict_families(X)
        out = np.full(next(iter(P.values())).shape, np.nan)
        lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        for k, t in enumerate(TASKS):
            c = self.combine.get(t, self.DEFAULT)
            ws = {f: w for f, w in c["w"].items() if w and f in P}
            tot = sum(ws.values())
            if c["mode"] == "prob":
                out[:, k] = sum(w * P[f][:, k] for f, w in ws.items()) / tot
            else:
                out[:, k] = 1 / (1 + np.exp(-sum(w * lg(P[f][:, k]) for f, w in ws.items()) / tot))
            if "platt" in c:   # recalibration fitted on the 1975-2005 development forecasts
                a_, b_ = c["platt"]
                out[:, k] = 1 / (1 + np.exp(-(a_ + b_ * lg(out[:, k]))))
        return out

    @property
    def psychonet(self):
        return self.fam["N"][0] if "N" in self.fam else None
