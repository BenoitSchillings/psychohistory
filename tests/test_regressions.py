"""Regression tests for bugs found in use. Each test names the bug it guards against."""
import numpy as np
import pytest

from psychohistory.data import LAST_OBS_YEAR


def _run(sim_env, years=4, n=48, **kw):
    from psychohistory import simulate as SIM
    from psychohistory.world import WorldConfig
    p, base, net, gr, dd, dy = sim_env
    world = kw.pop("world", None) or WorldConfig(sample_params=True)
    return SIM.simulate(p, base, net, gr, dd, LAST_OBS_YEAR, LAST_OBS_YEAR + years, n_scen=n, dyads=dy, world=world, **kw)


@pytest.mark.slow
def test_same_seed_same_worlds(sim_env):
    """paired comparisons (the adopt rule) need a run to repeat exactly on the same seed"""
    a, b = _run(sim_env, seed=3), _run(sim_env, seed=3)
    assert np.array_equal(a["world_gdp"], b["world_gdp"]) and np.array_equal(a["world_conflict"], b["world_conflict"])
    assert not np.array_equal(a["world_gdp"], _run(sim_env, seed=4)["world_gdp"])


@pytest.mark.slow
def test_income_term_reaches_health_with_world_layer(sim_env, monkeypatch):
    """10 November 2026: the world layer rebuilt health from its trend and dropped the income (Preston) term, so the
    term scored exactly zero in its test"""
    from psychohistory import simulate as SIM
    monkeypatch.setattr(SIM, "PRESTON", (1.13, -0.11))
    on = _run(sim_env, years=6)["health_end"]
    monkeypatch.setattr(SIM, "PRESTON", None)
    off = _run(sim_env, years=6)["health_end"]
    for k in ("life_exp", "log_cm"):
        assert np.array_equal(np.isfinite(on[k]), np.isfinite(off[k])) and np.isfinite(on[k]).mean() > 0.9
        assert np.nanmax(np.abs(on[k] - off[k])) > 1e-4, k


@pytest.mark.slow
def test_runs_without_interstate_layer(sim_env):
    """10 November 2026: with the catastrophe tails on, a run without the interstate layer (the knockout analysis)
    crashed on an empty record of large nuclear exchanges"""
    from psychohistory.world import WorldConfig
    o = _run(sim_env, world=WorldConfig(sample_params=True, interstate=False))
    assert np.isfinite(o["world_gdp"]).all() and np.isfinite(o["world_output"]).all()


@pytest.mark.slow
@pytest.mark.parametrize("tails,harvest", [(False, None), (True, None), (True, (0.0144, 0.1534))])
def test_catastrophe_switches(sim_env, monkeypatch, tails, harvest):
    """every combination of the catastrophe tails and the harvest-to-output link runs and stays finite"""
    from psychohistory import simulate as SIM
    monkeypatch.setattr(SIM, "CATASTROPHE_TAILS", tails)
    monkeypatch.setattr(SIM, "HARVEST_GDP", harvest)
    o = _run(sim_env)
    assert np.isfinite(o["world_gdp"]).all() and (np.asarray(o["world"]["excess_deaths"]) >= 0).all()


def test_release_scripts_stay_inside_the_repository():
    """the release chain once lived in a scratch folder outside the repository"""
    import glob, os
    root = os.path.join(os.path.dirname(__file__), "..")
    for f in glob.glob(os.path.join(root, "scripts", "*.sh")):
        src = open(f).read()
        assert "psycho_scratch" not in src and "/home/" not in src, f
