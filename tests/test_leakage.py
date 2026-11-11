"""Leakage tests by mutation (11 November 2026). Every observed value after an origin year is replaced by another
country's value for that year (keeping which cells are missing); nothing the models are given or fitted on at that
origin may change. Reading the code found no leak on 10 November 2026; these tests keep it so.

Not covered: series revised after the fact (today's data for 1995 is not what was known in 1995), which cells are
missing, the constants estimated once on the whole record (docs/methodology.md, Limits), and the interstate pair model."""
import copy
import warnings

import numpy as np
import pytest

ORIGIN = 1995


def _same(a, b, path="", out=None):
    """recursive comparison of fitted objects: arrays, tensors, dicts, lists, objects with __dict__"""
    import torch
    out = [] if out is None else out
    if torch.is_tensor(a):
        a, b = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    if isinstance(a, np.ndarray):
        if a.shape != b.shape:
            out.append(path + " shape")
        elif a.dtype.kind in "fc":
            if not np.allclose(a, b, rtol=1e-5, atol=1e-7, equal_nan=True):
                out.append(path)
        elif not np.array_equal(a, b):
            out.append(path)
    elif isinstance(a, dict):
        for k in a:
            _same(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append(path + " length")
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float):
        if not (a == b or (np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-7 + 1e-5 * abs(a)):
            out.append(path)
    elif hasattr(a, "__dict__") and not callable(a):
        _same(vars(a), vars(b), path, out)
    elif isinstance(a, (int, str, bool, type(None))):
        if a != b:
            out.append(path)
    return out


@pytest.fixture(scope="module")
def worlds():
    """the real panel and one whose future (after ORIGIN) is scrambled"""
    from psychohistory.pipeline import prepare
    from psychohistory.features import base_arrays, labels, panel_features
    from psychohistory.models import DEVICE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, base, X, Yl = prepare()
        t0 = p.yi(ORIGIN)
        perm = np.random.default_rng(0).permutation(len(p.codes))
        p2 = copy.deepcopy(p)
        n = 0
        for v in p2.x.values():
            if v.ndim == 2 and v.shape[1] == len(p.years):
                w = v[perm]
                m = np.isfinite(v) & np.isfinite(w)
                m[:, :t0 + 1] = False
                n += int((v[m] != w[m]).sum())
                v[m] = w[m]
        assert n > 100_000                                   # the mutation did change the future
        base2 = base_arrays(p2)
        X2 = panel_features(p2, base2, DEVICE)
    return (p, base, X, Yl), (p2, base2, X2, labels(p2, base2)), t0


def test_features_up_to_the_origin_do_not_use_the_future(worlds):
    (p, base, X, _), (_, base2, X2, _), t0 = worlds
    bad = [k for k, a in base.items() if getattr(a, "ndim", 0) == 2 and a.shape[1] == len(p.years)
           and _same(np.asarray(a, float)[:, :t0 + 1], np.asarray(base2[k], float)[:, :t0 + 1])]
    assert not bad, bad
    from psychohistory.features import FEATURES
    d = ~(np.isclose(X[:, :t0 + 1], X2[:, :t0 + 1], rtol=1e-5, atol=1e-7, equal_nan=True))
    assert not d.any(), [FEATURES[j] for j in np.where(d.any((0, 1)))[0]]


def test_training_rows_and_labels_do_not_use_the_future(worlds):
    """what the risk models are fitted on at the origin: rows, features and labels (5-year labels need 5 observed years)"""
    from psychohistory.backtest import flatten, train_mask
    from psychohistory.backtest import HORIZON, TASKS
    (p, _, X, Yl), (p2, _, X2, Yl2), _ = worlds
    Xf, Yf, ci, year = flatten(p, X, Yl)
    Xf2, Yf2, _, _ = flatten(p2, X2, Yl2)
    Y, tr = train_mask(Yf, year, ORIGIN)
    Y2, tr2 = train_mask(Yf2, year, ORIGIN)
    assert np.array_equal(tr, tr2) and year[tr].max() < ORIGIN
    assert np.allclose(Xf[tr], Xf2[tr], rtol=1e-5, atol=1e-7, equal_nan=True)
    assert np.array_equal(Y[tr], Y2[tr], equal_nan=True)
    for k, t in enumerate(TASKS):                          # no label whose window reaches past the origin
        assert (year[np.isfinite(Y[:, k])] + HORIZON[t] <= ORIGIN).all(), t


def test_simulator_equations_do_not_use_the_future(worlds):
    """the simulator's fitted pieces at the origin: growth and democracy equations, slow variables, co-movement,
    crisis-given-growth"""
    from psychohistory import simulate as SIM
    from psychohistory.dynamics import fit as slow_fit
    (p, base, X, _), (p2, base2, X2, _), _ = worlds
    years = np.asarray(p.years)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairs = {"growth and democracy": (SIM.fit_transitions(p, X, base, ORIGIN), SIM.fit_transitions(p2, X2, base2, ORIGIN)),
                 "slow variables": (slow_fit(base, years, ORIGIN), slow_fit(base2, years, ORIGIN)),
                 "co-movement": (SIM.comove_slopes(base, ORIGIN, years), SIM.comove_slopes(base2, ORIGIN, years)),
                 "crisis given growth": (SIM.crisis_given_growth(base, ORIGIN, years), SIM.crisis_given_growth(base2, ORIGIN, years))}
    bad = {k: _same(a, b)[:8] for k, (a, b) in pairs.items() if _same(a, b)}
    assert not bad, bad
    # (the comparison has teeth: the same pieces fitted five years later do differ)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _same(pairs["growth and democracy"][0], SIM.fit_transitions(p, X, base, ORIGIN + 5))
        assert _same(pairs["slow variables"][0], slow_fit(base, years, ORIGIN + 5))
