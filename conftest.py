"""Puts the repository root on the import path, so `python -P -m pytest -q tests` works without PYTHONPATH."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs the simulator (tens of seconds)")


@pytest.fixture(scope="session")
def sim_env():
    """A small fitted simulator shared by the simulation tests: panel, base arrays, a 1-member network, the fitted
    transitions and the interstate inputs."""
    from psychohistory.pipeline import prepare
    from psychohistory.backtest import flatten, train_mask
    from psychohistory.models import PsychoNet
    from psychohistory.data import LAST_OBS_YEAR
    from psychohistory import simulate as SIM, interstate as I
    p, base, X, Yl = prepare()
    Xf, Yf, ci, year = flatten(p, X, Yl)
    Ytr, tr = train_mask(Yf, year, LAST_OBS_YEAR)
    net = PsychoNet(members=1, epochs=2).fit(Xf[tr], Ytr[tr], ci[tr])
    gr, dd = SIM.fit_transitions(p, X, base, LAST_OBS_YEAR)
    dy = I.sim_inputs(I.run(verbose=False, backtests=False))
    return p, base, net, gr, dd, dy
