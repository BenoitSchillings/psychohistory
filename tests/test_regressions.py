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


@pytest.mark.slow
def test_switches_keep_runs_paired(sim_env):
    """10 November 2026 audit: draws taken from the main generator only when a switch was on (the AI catastrophe, country
    frailty, the interstate layer, pandemics) shifted every later draw, so knockouts and the AI / no-AI comparison were
    not paired with the baseline"""
    from psychohistory.world import WorldConfig, AI_SCENARIOS
    base = _run(sim_env, seed=5)["world"]
    variants = {"frailty": dict(frailty_sd=0.0),
                "interstate": dict(world=WorldConfig(sample_params=True, interstate=False)),
                "ai_catastrophe": dict(world=WorldConfig(sample_params=True, ai_catastrophe=False)),
                "ai_none": dict(world=WorldConfig(sample_params=True, ai=AI_SCENARIOS["none"])),
                "pandemics": dict(world=WorldConfig(sample_params=True, pandemics=False))}
    for name, kw in variants.items():
        w = _run(sim_env, seed=5, **kw)["world"]
        for k in ("volcano", "bubble") + (() if name == "pandemics" else ("pandemic",)):
            assert np.array_equal(np.asarray(base[k]), np.asarray(w[k])), (name, k)


def test_seed_offsets_are_unique():
    """10 November 2026 audit: seed + 7777 seeded both the oil draws and the nuclear-risk generator (and seed + 919 two
    draws before that): every generator needs its own offset"""
    import os, re
    root = os.path.join(os.path.dirname(__file__), "..", "psychohistory")
    found = []
    for f in ("simulate.py", "world.py", "tech.py"):
        for line in open(os.path.join(root, f)):
            code = line.split("#")[0]
            if "manual_seed" in code:
                found += [(int(n), f) for n in re.findall(r"\+ (\d{1,6})\b", code)]
    offsets = [n for n, _ in found]
    exclusive = {4242}          # the structural draw: two branches of one if / elif, never both
    dup = sorted({n for n in offsets if offsets.count(n) > 1} - exclusive)
    assert len(offsets) >= 15 and not dup, dup


@pytest.mark.slow
def test_gaza_deaths_do_not_fall_on_israel(sim_env, monkeypatch):
    """10 November 2026 audit: Israel lost about 0.8% of its population a year, because UCDP places the Gaza war's
    deaths in Israel and an ongoing conflict kept its origin-year death rate for ever"""
    from psychohistory import simulate as SIM
    p = sim_env[0]
    i, j = p.codes.index("ISR"), p.codes.index("PSE")
    new = _run(sim_env, years=5, seed=2)["paths"]["pop"]
    monkeypatch.setattr(SIM, "CIVIL_DEATHS", None)
    old = _run(sim_env, years=5, seed=2)["paths"]["pop"]
    assert (new[:, i, -1] / old[:, i, -1]).mean() > 1.01          # Israel no longer loses them
    assert (new[:, j, -1] / old[:, j, -1]).mean() < 1.0           # they fall on Palestine


@pytest.mark.slow
def test_civil_war_deaths_match_the_record_in_size(sim_env):
    """10 November 2026, first version 57 build: every country at war had the average country's death rate per person,
    whatever its size, and the world's civil-war deaths came out about 4.5 times the record (61,000 a year in
    1990-2024, at most 231,000 in one year)"""
    d = np.asarray(_run(sim_env, years=10, n=200, seed=6)["world"]["civil_battle_deaths"])
    per_year = d.mean(axis=tuple(i for i, n in enumerate(d.shape) if n == 200))      # mean over worlds, by year
    assert 20_000 < per_year[1:].mean() < 150_000, per_year


def test_random_channels_are_isolated():
    """11 November 2026: named channels (psychohistory/rng.py). Extra draws in one channel, or a channel never used,
    change nothing in another; a year's draws do not depend on how many were taken the year before"""
    import torch
    from psychohistory.rng import Channels
    a, b = Channels(7, "cpu"), Channels(7, "cpu")
    for k in (1, 2):
        a.at(k); b.at(k)
        torch.rand(1000, generator=b("pandemic"))                     # b draws far more here, and from one more channel
        torch.rand(k * 17, generator=b("coup"))
        ya = torch.rand(5, generator=a("growth"))
        assert torch.equal(ya, torch.rand(5, generator=b("growth")))
    a.at(2)
    assert torch.equal(ya, torch.rand(5, generator=a("growth")))      # the same key gives the same numbers again
    assert not torch.equal(ya, torch.rand(5, generator=Channels(8, "cpu")("growth")))


def test_grain_price_equation_is_finite_at_early_origins():
    """11 November 2026: origins before 1975 had no year in the estimation window; the NaN average reached the coup
    risk and silenced every coup after the first simulated year of the 1970 and 1973 hindcasts."""
    import math
    from psychohistory.food import fit_world_price
    for origin in (1970, 1973, 1976, 1990, 2025):
        w = fit_world_price(origin)
        assert all(math.isfinite(v) for k, v in w.items() if k not in ("resid", "resid_sd")), (origin, w)
        assert len(w["resid"]) >= 8, origin
        assert 0 <= w["oil"] <= 0.5 and -3 <= w["supply"] <= 0 and 0 <= w["revert"] <= 0.7, (origin, w)
