"""One small end-to-end acquisition round on the toy backend, for every strategy family."""
import json

import pytest

from fdrl.loop import Pipeline


@pytest.mark.parametrize("strategies", [["S0", "S2", "S3"], ["S3b", "S5", "S6"]])
def test_round_runs(tmp_path, cfg, strategies):
    cfg.experiment.strategies = strategies
    cfg.experiment.budgets = [10.0]
    pipe = Pipeline(cfg, exp_root=tmp_path / "exp")
    results = []
    for s in strategies:
        results += pipe.run_strategy("R2", 0, s, 10.0, rounds=1)
    for r in results:
        led = r.ledger
        assert r.round == 1
        if r.strategy == "S0":
            assert led["acquisition_cost"] == 0
        else:
            # budget respected up to one episode of overshoot
            assert led["acquisition_cost"] <= 10.0 * cfg.experiment.demo_equiv_steps + cfg.task.horizon
            assert led["acquisition_cost"] > 0
        m = json.loads((tmp_path / "exp" / "R2" / "B10" / r.strategy / "seed0" / "round1" / "metrics.json").read_text())
        assert 0.0 <= m["val_success"] <= 1.0
    # round 0 is shared and cached: D0 identical for all strategies
    assert (tmp_path / "exp" / "shared" / "R2" / "seed0" / "d0_manifest.json").exists()


def test_resume_skips_finished_rounds(tmp_path, cfg):
    pipe = Pipeline(cfg, exp_root=tmp_path / "exp")
    a = pipe.run_strategy("R2", 0, "S2", 5.0, rounds=1)
    b = pipe.run_strategy("R2", 0, "S2", 5.0, rounds=1)
    assert a[0].val_success == b[0].val_success and a[0].ledger == b[0].ledger


def test_scaling_is_nested(tmp_path, cfg):
    pipe = Pipeline(cfg, exp_root=tmp_path / "exp")
    out = pipe.run_scaling("R2", 0, [5, 10])
    assert [o["n"] for o in out] == [5, 10]
