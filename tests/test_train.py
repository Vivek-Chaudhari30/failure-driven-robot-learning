import numpy as np
import torch

from fdrl.policy.bc import MLPPolicy
from fdrl.train import fit_error_per_cell, train_bc


def _data(rng, n=2000):
    obs = rng.normal(size=(n, 41)).astype(np.float32)
    w = rng.normal(size=(41, 6)).astype(np.float32) * 0.1
    act = np.zeros((n, 7), np.float32)
    act[:, :6] = np.tanh(obs @ w)
    act[:, 6] = np.where(obs[:, 0] > 0, 1.0, -1.0)
    return obs, act


def test_loss_decreases(cfg, rng):
    obs, act = _data(rng)
    cfg.train.steps = 400
    cfg.train.log_every = 100
    r = train_bc(obs, act, cfg.train, cfg.policy, seed=0)
    assert r.losses[-1][1] < r.losses[0][1] * 0.5


def test_cpu_determinism(cfg, rng):
    obs, act = _data(rng)
    a = train_bc(obs, act, cfg.train, cfg.policy, seed=3).policy
    b = train_bc(obs, act, cfg.train, cfg.policy, seed=3).policy
    assert a.policy_id() == b.policy_id()


def test_save_load(tmp_path, cfg, rng):
    obs, act = _data(rng)
    pol = train_bc(obs, act, cfg.train, cfg.policy, seed=0).policy
    pid = pol.save(tmp_path / "p.pt")
    back = MLPPolicy.load(tmp_path / "p.pt")
    assert back.policy_id() == pid
    np.testing.assert_allclose(back.predict(obs[:10]), pol.predict(obs[:10]))
    errs = fit_error_per_cell(pol, obs, act, np.arange(len(obs)) % 3)
    assert set(errs) == {0, 1, 2}
    assert torch.isfinite(torch.tensor(list(errs.values()))).all()
