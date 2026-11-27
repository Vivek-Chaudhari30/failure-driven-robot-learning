import numpy as np

from fdrl.conditions import Condition, seed_for
from fdrl.expert.scripted import ScriptedExpert
from fdrl.sim.base import make_backend
from fdrl.sim.obs import OBS_DIM, PRIV_KEYS, build_obs
from fdrl.sim.rollout import rollout


def _run(cfg, conds):
    be = make_backend(cfg, 16)
    ex = ScriptedExpert(cfg.expert, cfg.task)
    items = [(c, seed_for("calib", 0, 0, i)) for i, c in enumerate(conds)]
    return rollout(be, items, lambda o, p: ex.act(p), cfg.task)


def test_priv_keys_and_obs(cfg):
    be = make_backend(cfg, 4)
    priv = be.reset([Condition(0, 0, 0, 0.5)] * 4, [1, 2, 3, 4])
    assert set(PRIV_KEYS) <= set(priv)
    assert build_obs(priv, np.zeros((4, 7))).shape == (4, OBS_DIM)


def test_expert_succeeds_in_feasible_conditions(cfg, rng):
    conds = [Condition(rng.uniform(-0.1, 0.1), rng.uniform(-0.25, 0.25), rng.uniform(-0.78, 0.78),
                       float(np.exp(rng.uniform(np.log(0.25), 0.0)))) for _ in range(32)]
    res = _run(cfg, conds)
    assert np.mean([r.success for r in res]) >= 0.95


def test_expert_fails_in_infeasible_band(cfg):
    res = _run(cfg, [Condition(0.0, 0.0, 0.0, 0.1)] * 8)
    assert not any(r.success for r in res)


def test_rollout_is_batch_independent(cfg):
    """An episode's outcome depends on (θ, seed, controller) only, not on its batch neighbours."""
    ex = ScriptedExpert(cfg.expert, cfg.task)
    c = Condition(0.05, -0.1, 0.4, 0.3)
    a = rollout(make_backend(cfg, 1), [(c, 7)], lambda o, p: ex.act(p), cfg.task)[0]
    others = [(Condition(-0.05, 0.2, -0.3, 0.9), 8)] * 3
    b = rollout(make_backend(cfg, 4), [(c, 7)] + others, lambda o, p: ex.act(p), cfg.task)[0]
    assert a.length == b.length and a.success == b.success
    np.testing.assert_allclose(a.actions, b.actions)


def test_mu_unaware_expert_slips_more(cfg):
    conds = [Condition(0.0, 0.0, 0.0, 0.17)] * 16
    aware = np.mean([r.success for r in _run(cfg, conds)])
    cfg.expert.mu_aware = False
    unaware = np.mean([r.success for r in _run(cfg, conds)])
    assert aware > unaware
