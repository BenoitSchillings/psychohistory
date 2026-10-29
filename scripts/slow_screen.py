"""Would more drivers improve the simulator's slow-variable equations (dynamics.SPEC: polarisation, mobilisation,
political violence, debt, inequality...)? Each now responds to growth, civil conflict, crises and democratic change
only. For each variable: ridge of the yearly change (in the variable's transform) on its level, the current drivers
and one candidate model feature, scored out of sample (yearly change net of each year's world mean) at rolling
origins, development 1975-1995 and test 2000-2014 (5 years after each origin), against the same model without the
candidate. Reports the candidates that improve both.
python -P scripts/slow_screen.py -> outputs/slow_screen.json"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn.linear_model import Ridge

from psychohistory import dynamics as D
from psychohistory.data import ROOT
from psychohistory.features import FEATURES
from psychohistory.pipeline import prepare

DEV, TEST = (1975, 1980, 1985, 1990, 1995), (2000, 2005, 2010, 2014)


def main():
    p, base, X, Yl = prepare()
    X = X.cpu().numpy() if hasattr(X, "cpu") else np.asarray(X)
    C, T, F = X.shape
    yrs = np.asarray(p.years)
    cy = np.broadcast_to(yrs[None], (C, T))
    lg, dem, conf = base["lg"], base["dem"], np.nan_to_num(base["conflict"])
    g = np.concatenate([np.full((C, 1), np.nan), np.diff(lg, axis=1)], 1)
    ddem = np.concatenate([np.full((C, 1), np.nan), np.diff(dem, axis=1)], 1)
    drivers = np.stack([g, conf, ddem], -1)
    res = {}
    for var, (kind, _) in D.SPEC.items():
        x = D._fwd(base[var].astype(float), kind)
        dx = np.full((C, T), np.nan)
        dx[:, :-1] = x[:, 1:] - x[:, :-1]
        lo, hi = np.nanquantile(dx, [0.01, 0.99])
        dx = np.clip(dx, lo, hi)

        def score(extra, origins):
            se, n = 0.0, 0
            for O in origins:
                tr = (cy >= 1960) & (cy + 1 <= O) & np.isfinite(dx) & np.isfinite(x)
                te = (cy >= O) & (cy < O + 5) & np.isfinite(dx) & np.isfinite(x)
                parts = [x[..., None], drivers] + ([X[..., [FEATURES.index(extra)]]] if extra else [])
                A = np.concatenate(parts, -1)
                if tr.sum() < 500 or te.sum() < 50:      # (series that start late: origins before their data)
                    continue
                Atr, Ate = A[tr], A[te]
                mu, sd = np.nanmean(Atr, 0), np.nanstd(Atr, 0) + 1e-9
                z = lambda M: np.nan_to_num(np.clip((M - mu) / sd, -6, 6))
                ytr, yte = dx[tr].copy(), dx[te].copy()
                for Y_, msk in ((ytr, tr), (yte, te)):
                    yy = cy[msk]
                    for u in np.unique(yy):
                        s = yy == u
                        Y_[s] -= Y_[s].mean()
                m = Ridge(alpha=10.0).fit(z(Atr), ytr)
                r = yte - m.predict(z(Ate))
                se += float(np.sum(r ** 2)); n += len(r)
            return se / max(n, 1)
        b = {"dev": score(None, DEV), "test": score(None, TEST)}
        hits = {}
        for c in FEATURES:
            if c == var:
                continue
            d_, t_ = score(c, DEV), score(c, TEST)
            rel = (100 * (d_ / b["dev"] - 1), 100 * (t_ / b["test"] - 1))
            if rel[0] < 0 and rel[1] < 0:
                hits[c] = [round(rel[0], 2), round(rel[1], 2)]
        top = dict(sorted(hits.items(), key=lambda kv: kv[1][0] + kv[1][1])[:6])
        res[var] = {"n_improving": len(hits), "top": top}
        print(f"{var:22s} {len(hits):3d} improve both; best: {top}", flush=True)
    json.dump(res, open(os.path.join(ROOT, "outputs", "slow_screen.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
