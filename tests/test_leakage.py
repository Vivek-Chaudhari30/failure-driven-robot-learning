"""ARCHITECTURE section 5: leakage guards are structural, and tested."""
import re
from pathlib import Path

import pandas as pd
import pytest

from fdrl.conditions import LeakageError
from fdrl.config import StrategyCfg
from fdrl.evaluate import CellStats, EvalReport, evaluate
from fdrl.intel.diagnose import DiagnosisSignals, diagnose
from fdrl.intel.stats import success_posteriors
from fdrl.intel.strategies import AcquisitionContext, make_plan
from fdrl.sim.base import make_backend

SRC = Path(__file__).resolve().parents[1] / "src" / "fdrl"


def _report(grid, split):
    return EvalReport(split, "p", 2, 0, {c: CellStats(2, 1) for c in grid.train_cells}, sim_steps=0)


def test_intel_rejects_test_reports(grid, cfg):
    rep = _report(grid, "test")
    with pytest.raises(LeakageError):
        success_posteriors(rep, grid.train_cells)
    with pytest.raises(LeakageError):
        diagnose(rep, DiagnosisSignals(expert={}, density={}), grid.train_cells, cfg.diagnosis, {})
    ctx = AcquisitionContext(grid, cfg.conditions, StrategyCfg(), "R2", 1000.0, 0, 1, pd.DataFrame(), (),
                             val_report=rep)
    with pytest.raises(LeakageError):
        make_plan("S3", ctx)


def test_oracle_needs_explicit_flag(grid, cfg):
    ctx = AcquisitionContext(grid, cfg.conditions, StrategyCfg(), "R2", 1000.0, 0, 1, pd.DataFrame(), (),
                             oracle_report=_report(grid, "test"))
    with pytest.raises(LeakageError):
        make_plan("S7", ctx)
    ctx.allow_oracle = True
    assert make_plan("S7", ctx).notes["oracle"] is True


def test_test_eval_requires_frozen_sha_and_test_dir(tmp_path, grid, cfg):
    be = make_backend(cfg, 4)
    ctrl = lambda o, p: o[:, :7] * 0  # noqa: E731
    with pytest.raises(LeakageError):
        evaluate(ctrl, be, grid, cfg, split="test", pipeline_seed=0, policy_id="x", out_dir=tmp_path / "test")
    with pytest.raises(LeakageError):
        evaluate(ctrl, be, grid, cfg, split="test", pipeline_seed=0, policy_id="x", out_dir=tmp_path / "val",
                 frozen_analysis="abc")
    rep, _ = evaluate(ctrl, be, grid, cfg, split="test", pipeline_seed=0, policy_id="x", n_per_cell=1,
                      out_dir=tmp_path / "test", frozen_analysis="abc")
    assert rep.frozen_analysis == "abc" and set(rep.per_cell) == set(range(grid.n_cells))


def test_intel_never_references_test_paths():
    pat = re.compile(r"""["'][^"']*\btest/""")
    for f in (SRC / "intel").glob("*.py"):
        assert not pat.search(f.read_text()), f"{f} references a test/ path"


def test_only_sim_imports_isaaclab():
    pat = re.compile(r"^\s*(import|from)\s+(isaaclab|isaacsim|omni)\b", re.M)
    for f in SRC.rglob("*.py"):
        if f.parent.name == "sim":
            continue
        assert not pat.search(f.read_text()), f"{f} imports Isaac Lab outside fdrl.sim"
