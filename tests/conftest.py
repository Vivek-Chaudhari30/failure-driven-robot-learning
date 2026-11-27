import numpy as np
import pytest

from fdrl.conditions import CellGrid
from fdrl.config import Config


@pytest.fixture
def cfg():
    c = Config()
    c.sim.num_envs = 32
    c.policy.hidden = 64
    c.policy.ensemble_size = 2
    c.train.steps = 300
    c.eval.n_per_cell_val = 2
    c.eval.n_per_cell_test = 2
    c.experiment.n0 = 20
    c.experiment.demo_equiv_steps = 85
    return c


@pytest.fixture
def grid(cfg):
    return CellGrid(cfg.conditions)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
