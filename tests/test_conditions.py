import pytest

from fdrl.conditions import (
    SEED_STREAMS,
    CellGrid,
    CellMixture,
    LeakageError,
    NominalDistribution,
    eval_plan,
    sample_condition,
    seed_for,
    stream_of,
    uniform_train,
)
from fdrl.config import ConditionSpaceCfg


def test_seed_streams_disjoint():
    ranges = sorted(SEED_STREAMS.values())
    for (_, a_hi), (b_lo, _) in zip(ranges, ranges[1:], strict=False):
        assert a_hi <= b_lo


@pytest.mark.parametrize("stream", list(SEED_STREAMS))
def test_seed_for_stays_in_stream(stream):
    for ps in (0, 3):
        for i in (0, 5, 999):
            assert stream_of(seed_for(stream, ps, 0, i)) == stream


def test_seed_for_rejects_overflow():
    with pytest.raises(LeakageError):
        seed_for("acq", 0, 0, 10_000)
    with pytest.raises(LeakageError):
        seed_for("acq", 0, 5, 0)  # block 5 overflows the per-seed stride
    with pytest.raises(LeakageError):
        seed_for("acq", 20, 0, 0)  # leaves the acquisition range


def test_cell_roundtrip(grid, rng):
    for cell in range(grid.n_cells):
        for _ in range(20):
            c = grid.sample_in_cell(cell, rng)
            assert grid.cell_of(c) == cell
    assert grid.n_cells == 36


def test_holdout_never_sampled_for_acq_or_val(grid):
    dist = CellMixture({c: 1.0 for c in range(grid.n_cells)})  # includes H on purpose
    hit = 0
    for i in range(400):
        s = seed_for("acq", 0, 1, i)
        try:
            c = sample_condition(grid, dist, s, "acq")
            assert grid.cell_of(c) not in grid.holdout
        except LeakageError:
            hit += 1
    assert hit > 0  # the guard actually fired


def test_sample_condition_checks_stream(grid):
    with pytest.raises(LeakageError):
        sample_condition(grid, uniform_train(grid), seed_for("test", 0, 0, 0), "acq")


def test_nominal_avoids_holdout(grid, cfg):
    d = NominalDistribution(cfg.conditions)
    for i in range(300):
        c = sample_condition(grid, d, seed_for("acq", 0, 0, i), "acq")
        assert c.mu >= cfg.conditions.nom_mu_min - 1e-9


def test_eval_plan_pairing_and_holdout(grid):
    val = eval_plan(grid, "val", 0, 3)
    assert val == eval_plan(grid, "val", 0, 3)  # identical for every strategy
    assert not {p.cell for p in val} & grid.holdout
    test = eval_plan(grid, "test", 0, 2)
    assert grid.holdout <= {p.cell for p in test}
    assert all(stream_of(p.seed) == "test" for p in test)


def test_holdout_must_keep_factor_bins():
    bad = ConditionSpaceCfg(holdout=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2],
                                     [0, 2, 0], [0, 2, 1], [0, 2, 2]])
    with pytest.raises(ValueError):
        CellGrid(bad)


def test_normalize_in_unit_box(grid, rng):
    conds = [grid.sample_in_cell(int(c), rng) for c in rng.integers(0, 36, 100)]
    z = grid.normalize(conds)
    assert z.min() >= -1e-9 and z.max() <= 1 + 1e-9
