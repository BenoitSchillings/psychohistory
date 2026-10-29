"""Smoke tests: the panel, features, labels, interstate inputs, a short simulation with the world layer,
forced wars, parameter sampling and the monthly model's helpers. Run with: python -m pytest -q tests"""
import numpy as np
import pytest

from psychohistory.data import LAST_OBS_YEAR, get_panel
from psychohistory.features import FEATURES, TASKS, base_arrays, labels


@pytest.fixture(scope="module")
def panel():
    return get_panel()


def test_panel_shapes(panel):
    C, Y = len(panel.codes), len(panel.years)
    assert C > 150 and panel.years[0] == 1850
    for k in ("gdp_pc", "democracy", "conflict", "coup_attempts", "pop"):
        assert panel.x[k].shape == (C, Y)
    t = panel.yi(LAST_OBS_YEAR)
    assert np.isfinite(panel.x["democracy"][:, t]).mean() > 0.9
    assert np.isfinite(panel.x["conflict"][:, t]).mean() > 0.9


def test_labels_binary(panel):
    b = base_arrays(panel)
    Y = labels(panel, b)
    vals = np.unique(Y[np.isfinite(Y)])
    assert set(vals.tolist()) <= {0.0, 1.0}
    assert Y.shape[-1] == len(TASKS)
    # no 5-year label can use years after the last observed year
    assert np.isnan(Y[:, panel.yi(LAST_OBS_YEAR) - 4:, 0]).all()


def test_feature_list_unique():
    assert len(FEATURES) == len(set(FEATURES))


def test_param_ranges_and_defaults():
    from psychohistory.world import PARAM_RANGES, WorldConfig, AIScenario
    cfg, ai = WorldConfig(), AIScenario()
    for k in PARAM_RANGES:
        assert hasattr(cfg, k) or hasattr(ai, k) or k == "climate_damage", k
    assert 0 < cfg.dem_peace_factor < 1 and cfg.war_site_loss > 0
    from psychohistory.world import WATER_RANGES, EXTRA_RANGES
    for k, (kind, a, b) in {**WATER_RANGES, **EXTRA_RANGES}.items():
        assert hasattr(cfg, k), k
        v = getattr(cfg, k)
        assert (a <= v <= b) if kind == "u" else abs(v - a) <= 2 * b, (k, v)
    assert not set(WATER_RANGES) & set(EXTRA_RANGES) and not set(PARAM_RANGES) & set(EXTRA_RANGES)


def test_monthly_annual_probability():
    from psychohistory.monthly import annual_probability
    pred = {h: np.full(3, 0.1) for h in range(1, 13)}
    assert np.allclose(annual_probability(pred), 1 - 0.9 ** 12)


@pytest.mark.slow
def test_short_simulation_with_world_layer(panel):
    from psychohistory.pipeline import prepare
    from psychohistory.backtest import flatten, train_mask
    from psychohistory.models import PsychoNet
    from psychohistory import simulate as SIM, interstate as I
    from psychohistory.world import WorldConfig
    p, base, X, Yl = prepare()
    Xf, Yf, ci, year = flatten(p, X, Yl)
    Ytr, tr = train_mask(Yf, year, LAST_OBS_YEAR)
    net = PsychoNet(members=1, epochs=2).fit(Xf[tr], Ytr[tr], ci[tr])
    gr, dd = SIM.fit_transitions(p, X, base, LAST_OBS_YEAR)
    dy = I.sim_inputs(I.run(verbose=False, backtests=False))
    o = SIM.simulate(p, base, net, gr, dd, LAST_OBS_YEAR, LAST_OBS_YEAR + 5, n_scen=64, dyads=dy,
                     world=WorldConfig(sample_params=True, forced=(("CHN", "TWN", LAST_OBS_YEAR + 2),)), chronicle=2)
    w = o["world"]
    assert np.isfinite(o["world_gdp"]).all()
    assert (o["world_conflict"] >= 0).all() and (o["world_conflict"] <= 1).all()
    assert "params" in w and len(w["params"]["cascade_q"]) == 64
    # fixed-parameter world (scenario runs): every assumption enters as a plain number
    o2 = SIM.simulate(p, base, net, gr, dd, LAST_OBS_YEAR, LAST_OBS_YEAR + 3, n_scen=16, dyads=dy,
                      world=WorldConfig(sample_params=False))
    assert np.isfinite(o2["world_gdp"]).all()
    # a hindcast origin (before the 2025 shocks and before some series start)
    o3 = SIM.simulate(p, base, net, gr, dd, 1995, 1998, n_scen=16, dyads=dy, world=WorldConfig(sample_params=True))
    assert np.isfinite(o3["world_gdp"]).all()
    assert len(o["chronicle"]) == 5
    i = [k for k, (a, b) in enumerate(zip(dy["now"].a, dy["now"].b)) if {a, b} == {"CHN", "TWN"}][0]
    assert w["pair_prob"][i] == 1.0          # the forced war happened in every world
    # conflicts already recorded for the year after the origin happen in every world
    for a_, b_, y_, w_ in dy["observed"]:
        j = [k for k, (a, b) in enumerate(zip(dy["now"].a, dy["now"].b)) if {a, b} == {a_, b_}]
        if j:
            assert w["pair_prob"][j[0]] == 1.0


def test_uncertain_ai_arrival(panel):
    """per-world arrival years: some worlds get no AI shock, the arrival share rises with the CDF"""
    import torch
    from psychohistory.world import AI_SCENARIOS, ARRIVAL_CDF, World, WorldConfig
    from psychohistory.features import base_arrays
    S = 4000
    W = World(WorldConfig(ai=AI_SCENARIOS["uncertain"], sample_params=True, interstate=False), panel, base_arrays(panel),
              LAST_OBS_YEAR, S, "cpu", torch.Generator().manual_seed(0))
    arr = W.arrival_w.numpy().ravel()
    assert abs(np.isinf(arr).mean() - (1 - ARRIVAL_CDF[-1][1])) < 0.03
    assert abs((arr <= 2032.5).mean() - 0.375) < 0.03
    A, _, fr = W.adoption(LAST_OBS_YEAR + 40)
    assert float(A[torch.isinf(W.arrival_w).ravel()].max()) == 0.0
