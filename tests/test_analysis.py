import math

import numpy as np

from fdrl.analysis.stats import (
    bootstrap_ci,
    holm,
    iqm,
    n_star,
    paired_bootstrap,
    required_seeds,
    rho,
)


def test_bootstrap_ci_contains_mean():
    ci = bootstrap_ci(np.arange(10.0), n_boot=2000)
    assert ci.lo < 4.5 < ci.hi


def test_paired_detects_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(0.5, 0.1, 10)
    ci = paired_bootstrap(base + 0.1 + rng.normal(0, 0.01, 10), base, n_boot=2000)
    assert ci.lo > 0 and ci.p_value < 0.01


def test_holm_monotone():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert math.isclose(adj["a"], 0.03) and adj["c"] >= adj["a"] and adj["b"] >= adj["c"]


def test_iqm():
    assert iqm([0, 1, 2, 3, 100, 4, 5, 6]) < 10


def test_n_star_and_rho():
    sizes = [25, 50, 100, 200, 400]
    succ = [0.2, 0.4, 0.6, 0.8, 0.9]
    assert math.isclose(n_star(sizes, succ, 0.8), 200)
    assert 100 < n_star(sizes, succ, 0.7) < 200
    assert n_star(sizes, succ, 0.95) == math.inf
    U = np.tile(succ, (5, 1))
    MN = np.tile([100, 150, 200], (5, 1))
    MS = np.tile([0.6, 0.8, 0.9], (5, 1))
    r = rho(sizes, U, MN, MS, 0.8, n_boot=200)
    assert math.isclose(r.mean, 200 / 150)


def test_required_seeds():
    assert required_seeds(0.05, effect=0.05) == 8
