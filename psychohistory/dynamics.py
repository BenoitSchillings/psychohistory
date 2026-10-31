"""Slow structural variables that used to be frozen at their origin value in simulation.

Each variable follows an equation estimated on country-years 1960..origin (pooled OLS):

    v[t+1] - v[t] = a + rho * (v[t] - anchor[t]) + b . drivers[t+1] + e,   e ~ N(0, sd)

anchor: the country's trailing 20-year mean (in simulation it moves too, as an exponential average with the same
20-year span, starting from the origin's 20-year mean; plus any AI-driven shift).
drivers: growth of GDP per person, civil conflict, the change in democracy and financial crises in the same year,
all of which the simulator produces. The intercept is the historical average drift (e.g. rising debt since 1970);
the simulation sets it to zero, so the variables move only through reversion, drivers and shocks, not by
extrapolating a past trend for forty years. Drivers are centred on their historical means for the same reason.

Validation: out-of-sample on 2006..origin with equations fitted to 1960-2005 (R^2 of the yearly change against
"no change", and the coverage of the 90% band of the simulated 10-year change).
"""
from __future__ import annotations

import numpy as np

# variable: (transform, drivers); variables marked in NO_REVERT trend rather than revert (poverty falls with growth)
SPEC = {
    "public_debt": ("log", ["g", "conf", "crisis", "ddem"]),
    "polarization": ("level", ["g", "conf", "ddem"]),
    "mass_mobilization": ("level", ["g", "conf", "ddem"]),
    "mob_democracy": ("level", ["g", "conf", "ddem"]),
    "mob_autocracy": ("level", ["g", "conf", "ddem"]),
    "political_violence": ("level", ["g", "conf", "ddem"]),
    "gini": ("level", ["g", "ddem"]),
    "top1": ("level", ["g", "ddem"]),
    "extreme_poverty": ("log", ["g", "conf"]),
    "military": ("level", ["g", "conf"]),
    # added 2026-10-26 (independent review): non-state and one-sided violence (per capita, 3-year) used to fade to zero in
    # simulation with nothing to renew them; state reach and fiscal capacity were frozen at their origin values
    "nonstate_3y_pc": ("log", ["g", "conf", "ddem"]),
    "onesided_3y_pc": ("log", ["g", "conf", "ddem"]),
    "territorial_control": ("logit100", ["g", "conf", "ddem"]),      # % of territory: logit scale, no pile-up at 100
    "fiscal_capacity": ("level", ["g", "conf", "ddem"]),
}
# V-Dem institutions given their own equation (driven by the democracy change, with their own shocks) instead of moving
# in fixed proportion to democracy (simulate.COMOVE). Free expression adopted 2026-10-30: in lockstep, its 5-year
# erosion (a democracy feature: speech goes before elections) could never lead in simulation. Judicial constraints
# too added nothing (sim_eval l2fe vs l2fej); anti-pluralism and polarization as drivers did not forecast its changes
INST_SLOW = ("free_expression",)


def _spec():
    return {**SPEC, **{v: ("level", ["ddem", "conf"]) for v in INST_SLOW}}


DRIVERS = ["g", "conf", "crisis", "ddem"]
NO_REVERT = {"extreme_poverty", "gini"}
# household-survey series, observed every few years and carried forward up to 10 years in the panel; the others are
# annual, so an unchanged value is a real observation of no change
SURVEY = {"extreme_poverty", "gini", "top1"}
# bounded variables where many countries sit at a bound (no non-state violence; full territorial control): a pooled
# shock would push them off it only in one direction and drift every such country. Their shocks are scaled by the
# country's own yearly variability over the 20 years before the origin (relative to the pooled residual), and in
# full whenever the country is in civil conflict
HETERO = {"nonstate_3y_pc", "onesided_3y_pc", "territorial_control", "fiscal_capacity"}
# violence reverts to the country's own level before the origin (its 20-year mean), not to a moving anchor: with a
# moving anchor, years of conflict ratcheted the level up without limit (one-sided violence tripled by 2065 in a test)
FIXED_ANCHOR = {"nonstate_3y_pc", "onesided_3y_pc"}
WINSOR = 0.01                     # yearly changes are winsorised at the 1st/99th percentiles before fitting (data breaks)
ANCHOR_YEARS = 20


def _fwd(v, kind):
    if kind == "logit100":
        q = np.clip(v, 0, 100) / 100 * 0.99 + 0.005
        return np.log(q / (1 - q))
    return np.log(np.clip(v, 0, None) + 1.0) if kind == "log" else v


def _inv(v, kind):
    if kind == "logit100":
        return (1 / (1 + np.exp(-v)) - 0.005) / 0.99 * 100
    return np.exp(v) - 1.0 if kind == "log" else v


def _anchor(x):
    """trailing mean over ANCHOR_YEARS (NaN-aware, at least 5 observations)"""
    out = np.full(x.shape, np.nan)
    for t in range(x.shape[1]):
        seg = x[:, max(0, t - ANCHOR_YEARS + 1):t + 1]
        n = np.isfinite(seg).sum(1)
        with np.errstate(all="ignore"):
            out[:, t] = np.where(n >= 5, np.nanmean(seg, 1), np.nan)
    return out


def _drivers(base):
    lg, dem = base["lg"], base["dem"]
    g = np.full(lg.shape, np.nan)
    g[:, 1:] = lg[:, 1:] - lg[:, :-1]
    dd = np.full(dem.shape, np.nan)
    dd[:, 1:] = dem[:, 1:] - dem[:, :-1]
    return {"g": np.clip(g, -0.3, 0.3), "conf": np.nan_to_num(base["conflict"]), "crisis": np.nan_to_num(base["fin_crisis"]),
            "ddem": dd}


def _rows(base, years, var, lo, hi, gaps=False):
    """(X, y) for yearly changes. Only changes between two actual observations count: y == 0 is a value carried
    forward, and a change right after carried values spans several years (household surveys every few years). With
    gaps=True (variables surveyed too rarely for enough consecutive pairs) such a change is used as its average
    yearly change over the gap, with the reversion term taken at the previous observation."""
    kind, drv = _spec()[var]
    x = _fwd(base[var].astype(float), kind)
    anc = _anchor(x)
    D = _drivers(base)
    t = np.where((years >= lo) & (years < hi))[0]
    t = t[t + 1 < x.shape[1]]
    # years since the previous distinct value (1 = consecutive observations)
    gap = np.ones(x.shape)
    for j in range(1, x.shape[1]):
        gap[:, j] = np.where(x[:, j] == x[:, j - 1], gap[:, j - 1] + 1, 1)
    if var in SURVEY:
        m = np.where(x[:, t + 1] != x[:, t], gap[:, t] if gaps else np.where(gap[:, t] == 1, 1, np.nan), np.nan)
    else:
        m = np.ones((x.shape[0], len(t)))
    m = np.where(m > 11, np.nan, m)
    t0 = np.clip(t[None, :] - (np.nan_to_num(m, nan=1) - 1).astype(int), 0, None)
    rows_i = np.arange(x.shape[0])[:, None]
    y = (x[:, t + 1] - x[rows_i, t0]) / m
    rev = x[rows_i, t0] - anc[rows_i, t0]

    def span_mean(a):
        """driver averaged over the years the change spans, (t0, t+1] (one year for consecutive observations)"""
        if not gaps:
            return a[:, t + 1]
        ok_ = np.isfinite(a)
        cs = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(np.nan_to_num(a), 1)], 1)
        cn = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(ok_, 1)], 1)
        tot = cs[:, t + 2] - cs[rows_i, t0 + 1]
        n_ = cn[:, t + 2] - cn[rows_i, t0 + 1]
        return np.where(n_ > 0, tot / np.maximum(n_, 1), np.nan)
    X = [rev if var not in NO_REVERT else np.zeros_like(y)] + [span_mean(D[d]) for d in drv]
    X = np.stack([a.ravel() for a in X], 1)
    y = y.ravel()
    ok = np.isfinite(y) & np.isfinite(X).all(1) & ((y != 0) | (var not in SURVEY))
    return X[ok], y[ok]


def fit(base, years, origin, start=1960):
    """{var: {kind, drivers, rho, beta, mean_x, intercept, sd, lo, hi, n}}"""
    out = {}
    for var, (kind, drv) in _spec().items():
        if var not in base:
            continue
        X, y = _rows(base, years, var, start, origin)
        gaps = var in SURVEY and len(y) < 200
        if gaps:
            X, y = _rows(base, years, var, start, origin, gaps=True)
        if len(y) < 100:
            continue
        mx = X.mean(0)
        # (not institutions: their largest yearly changes are regime transitions, where they jump with democracy;
        #  clipped, free expression's slope on the democracy change fell from 0.84 to 0.50 and its reversion vanished)
        if var not in INST_SLOW:
            y = np.clip(y, *np.quantile(y, [WINSOR, 1 - WINSOR]))
        A = np.c_[np.ones(len(y)), X - mx]
        if var in NO_REVERT:
            A[:, 1] = 0
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ b
        v = _fwd(base[var].astype(float), kind)[:, np.asarray(years) <= origin]     # nothing after the origin
        # yearly residuals include measurement noise that does not accumulate: the shock size is scaled so that the
        # spread of simulated 10-year changes matches the spread of observed 10-year changes (same years)
        ty = np.where((years >= start) & (years + 10 <= origin))[0]
        d10 = (v[:, ty + 10] - v[:, ty]).ravel()
        if var in INST_SLOW:
            # (only the part of the 10-year change the democracy change does not explain: that part is the driver's)
            dm = base["dem"][:, np.asarray(years) <= origin]
            d10 = d10 - float(b[2 + drv.index("ddem")]) * (dm[:, ty + 10] - dm[:, ty]).ravel()
        # (an unchanged value is a carried-forward observation, except for the bounded HETERO variables, where no
        # change - no violence, full control - is the real outcome for most countries)
        d10 = d10[np.isfinite(d10) & ((d10 != 0) | (var in HETERO))]
        r = 1 + float(b[1])
        sd10_noise = res.std() * np.sqrt(sum(r ** (2 * j) for j in range(10)))
        # (in either direction: yearly residuals carry measurement noise that does not accumulate, while averaged
        # multi-year survey changes and many unchanged annual values understate the yearly shock)
        if len(d10) >= 20:
            d10 = np.clip(d10, *np.quantile(d10, [WINSOR, 1 - WINSOR]))
            scale = float(np.clip(d10.std() / sd10_noise, 0, 2.5)) if sd10_noise > 0 else 1.0
        else:
            scale = 1.0                   # (too few 10-year changes before an early origin)
        out[var] = {"kind": kind, "drivers": drv, "intercept": float(b[0]), "rho": float(b[1]), "beta": b[2:].tolist(),
                    "mean_x": mx[1:].tolist(), "sd": float(res.std() * scale), "sd_yearly_resid": float(res.std()),
                    "sd10_observed": float(d10.std()), "n": int(len(y)), "gap_averaged": gaps,
                    "lo": float(np.nanmin(v)), "hi": float(np.nanmax(v))}
    return out


def validate(base, years, split=2005, origin=2025):
    """fit on 1960-split, score the yearly change on split..origin: R^2 against no change"""
    eq = fit(base, years, split + 1)
    out = {}
    for var, e in eq.items():
        X, y = _rows(base, years, var, split + 1, origin)
        if not len(y):
            continue
        pred = e["rho"] * X[:, 0] + (X[:, 1:] - np.array(e["mean_x"])) @ np.array(e["beta"])
        y = np.clip(y, *np.quantile(y, [WINSOR, 1 - WINSOR]))
        out[var] = {"r2_vs_no_change": float(1 - np.mean((y - pred) ** 2) / np.mean(y ** 2)),
                    "r2_with_drift": float(1 - np.mean((y - pred - e["intercept"]) ** 2) / np.mean(y ** 2)),
                    "rho": round(e["rho"], 4), "n_test": int(len(y)),
                    "beta": dict(zip(e["drivers"], np.round(e["beta"], 4).tolist()))}
    return out


class SlowState:
    """simulation state for the dynamic slow variables (torch, [S, C])"""

    def __init__(self, eq, base, t0, cur, device, gen, noise=1.0):
        import torch
        self.eq, self.gen, self.noise = eq, gen, noise
        self.anchor = {}
        for var, e in eq.items():
            x = _fwd(base[var][:, max(0, t0 - ANCHOR_YEARS + 1):t0 + 1].astype(float), e["kind"])
            with np.errstate(all="ignore"):
                a = np.nanmean(x, 1)
            x0 = _fwd(cur[var].detach().cpu().numpy()[0].astype(float), e["kind"])
            a = np.where(np.isfinite(a), a, x0)
            self.anchor[var] = torch.as_tensor(np.nan_to_num(a), dtype=torch.float32, device=device)
        self.shift = {}                   # externally imposed level shifts (AI scenarios), [S, C]
        self.scale = {}
        for var, e in eq.items():
            # (institutions too: free expression sits near its top for 28 countries, where pooled shocks could only
            #  push it down, -0.15 in 40 years by the bound alone; own volatility is a quarter of the pooled size)
            if var in HETERO or var in INST_SLOW:
                x = _fwd(base[var][:, max(0, t0 - ANCHOR_YEARS):t0 + 1].astype(float), e["kind"])
                with np.errstate(all="ignore"):
                    sd_c = np.nanstd(np.diff(x, axis=1), 1)
                sc = np.clip(np.nan_to_num(sd_c / max(e["sd_yearly_resid"], 1e-9), nan=1.0), 0, 1.5)
                self.scale[var] = torch.as_tensor(sc, dtype=torch.float32, device=device)

    def step(self, cur, g, conf, crisis, ddem):
        """advance every variable one year in place on `cur`"""
        import torch
        D = {"g": g.clamp(-0.3, 0.3), "conf": conf, "crisis": crisis, "ddem": ddem}
        for var, e in self.eq.items():
            if var not in cur:
                continue
            x = cur[var]
            finite = torch.isfinite(x)
            xf = _fwd_t(torch.nan_to_num(x), e["kind"])
            sh = self.shift.get(var, 0.0)
            # the moving anchor follows the level without the imposed AI shift, which is then added once
            if var not in FIXED_ANCHOR:
                self.anchor[var] = self.anchor[var] + (xf - sh - self.anchor[var]) / ANCHOR_YEARS
            anc = self.anchor[var] + sh
            dx = e["rho"] * (xf - anc)
            for d, b, m in zip(e["drivers"], e["beta"], e["mean_x"]):
                dx = dx + b * (D[d] - m)
            sc = torch.maximum(self.scale[var][None], conf) if var in self.scale else 1.0
            dx = dx + self.noise * e["sd"] * sc * torch.randn(x.shape, device=x.device, generator=self.gen)
            xn = _inv_t((xf + dx).clamp(e["lo"], e["hi"]), e["kind"])
            cur[var] = torch.where(finite, xn, x)       # countries without data stay missing


def _fwd_t(v, kind):
    import torch
    if kind == "logit100":
        q = v.clamp(0, 100) / 100 * 0.99 + 0.005
        return torch.log(q / (1 - q))
    return torch.log(v.clamp(min=0) + 1.0) if kind == "log" else v


def _inv_t(v, kind):
    import torch
    if kind == "logit100":
        return (torch.sigmoid(v) - 0.005) / 0.99 * 100
    return torch.exp(v) - 1.0 if kind == "log" else v


PARTY_VARS = ["vp_gov_antiplural", "vp_gov_populism", "vp_all_antiplural_votew", "vp_max_antiplural_major"]


def fit_party(p, base, origin):
    """party-system change at elections (V-Party, 1970-origin): between consecutive V-Party elections,
    x[next] - x = rev * (x - the country's 20-year mean) + e. Before 2026-10-26 the party measures stayed at their last
    value for the whole simulation, whatever the simulated elections did. Growth over the interval had no significant
    effect (tested, |t| < 1.5), so the change is reversion plus a shock. Out of sample (fit < 2005, test >= 2005) the
    predictable part is ~0 (R^2 vs no change -0.02..+0.02): the gain is the spread, not a forecast of direction.
    2026-10-26: rev -0.25 / -0.42 / -0.07 / -0.09, residual sd 0.18 / 0.17 / 0.11 / 0.14 per election."""
    ey = p.x.get("feat__vp_vparty_election_year")
    out = {}
    if ey is None:
        return out
    for k in PARTY_VARS:
        a = base.get(k)
        if a is None:
            continue
        rows = []
        for c in range(a.shape[0]):
            els = sorted({int(v) for v in ey[c][np.isfinite(ey[c])]})
            for e0, e1 in zip(els[:-1], els[1:]):
                if e1 > origin:
                    continue
                t0, t1 = p.yi(e0), p.yi(e1)
                if np.isfinite(a[c, t0]) and np.isfinite(a[c, t1]):
                    with np.errstate(all="ignore"):
                        anc = np.nanmean(a[c, max(0, t0 - ANCHOR_YEARS):t0 + 1])
                    rows.append([a[c, t0], a[c, t1], a[c, t0] - anc])
        if len(rows) < 100:
            continue
        R = np.array(rows)
        d = R[:, 1] - R[:, 0]
        A = np.c_[np.ones(len(R)), R[:, 2]]
        bb = np.linalg.lstsq(A, d, rcond=None)[0]
        out[k] = {"rev": float(bb[1]), "sd": float(np.std(d - A @ bb)), "n": len(R)}
    return out
