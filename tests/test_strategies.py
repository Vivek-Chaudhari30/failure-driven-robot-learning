import numpy as np
import pandas as pd
import pytest

from fdrl.config import StrategyCfg
from fdrl.evaluate import CellStats, EvalReport
from fdrl.intel.diagnose import CellDiagnosis, Diagnosis
from fdrl.intel.stats import ExpertCellStats, beta_posterior
from fdrl.intel.strategies import STRATEGIES, AcquisitionContext, make_plan


def _ctx(grid, cfg, succ, labels=None, scfg=None):
    rep = EvalReport("val", "p", 20, 0, {c: CellStats(20, succ(c)) for c in grid.train_cells}, sim_steps=1000)
    diag = None
    if labels:
        diag = Diagnosis({c: CellDiagnosis(labels(c), 1.0, (scfg or StrategyCfg()).fixability_priors[labels(c)], {})
                          for c in grid.train_cells})
    expert = {c: ExpertCellStats(10, 10, 0.95, 80.0, 84.0) for c in range(grid.n_cells)}
    return AcquisitionContext(grid, cfg.conditions, scfg or StrategyCfg(), "R2", 8500.0, 0, 1, pd.DataFrame(), (),
                              val_report=rep, diagnosis=diag, expert=expert)


def test_beta_posterior():
    p = beta_posterior(18, 20)
    assert 0.8 < p.mean < 0.9 and p.upper > p.mean and p.std > 0


def test_s3_weights_follow_failures_with_floor(grid, cfg):
    bad = grid.train_cells[0]
    plan = make_plan("S3", _ctx(grid, cfg, lambda c: 0 if c == bad else 20))
    w = plan.cell_weights
    assert abs(sum(w.values()) - 1) < 1e-9
    assert w[bad] == max(w.values())
    assert min(w.values()) >= 0.10 / len(grid.train_cells) - 1e-12


def test_s5_skips_infeasible_and_healthy(grid, cfg):
    c_inf, c_cov = grid.train_cells[0], grid.train_cells[1]

    def labels(c):
        return "expert_infeasible" if c == c_inf else ("coverage" if c == c_cov else "healthy")

    plan = make_plan("S5", _ctx(grid, cfg, lambda c: 5 if c in (c_inf, c_cov) else 19, labels))
    w = plan.cell_weights
    floor_w = 0.10 / len(grid.train_cells)
    assert abs(w[c_inf] - floor_w) < 1e-3  # only the uniform floor
    assert w[c_cov] == max(w.values())


def test_s5_shuffled_permutes(grid, cfg):
    def labels(c):
        return "coverage"

    ctx = _ctx(grid, cfg, lambda c: c % 20, labels)
    a, b = make_plan("S5", ctx).cell_weights, make_plan("S5-shuffled", ctx).cell_weights
    assert sorted(np.round(list(a.values()), 9)) == sorted(np.round(list(b.values()), 9))
    assert a != b


def test_s4_sequences_avoid_holdout(grid, cfg):
    for name in ("S4", "S4b"):
        plan = make_plan(name, _ctx(grid, cfg, lambda c: 10))
        assert plan.mode == "sequence" and plan.sequence
        assert not {grid.cell_of(c) for c in plan.sequence} & grid.holdout


def test_all_strategies_construct(grid, cfg):
    def labels(c):
        return "coverage"

    for name in STRATEGIES:
        if name == "S7":
            continue
        plan = make_plan(name, _ctx(grid, cfg, lambda c: 10, labels))
        assert plan.strategy == name


def test_unknown_strategy(grid, cfg):
    with pytest.raises(ValueError):
        make_plan("S9", _ctx(grid, cfg, lambda c: 10))
