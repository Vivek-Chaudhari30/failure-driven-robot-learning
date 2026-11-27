#!/usr/bin/env python
"""Calibration sweep (EXPERIMENT_DESIGN section 3): scripted-expert success and episode length per
cell, on CALIBRATION seeds only. Prints the per-cell table and the mean accepted-demo length under
uniform sampling (use it for experiment.demo_equiv_steps).

  python scripts/calibrate.py --per-cell 50 [--config configs/env/isaac.yaml]
"""
import json
import sys
from pathlib import Path

import numpy as np

from fdrl.cli import base_parser, config_from_args
from fdrl.conditions import CellGrid, seed_for
from fdrl.expert.scripted import ScriptedExpert
from fdrl.sim.base import make_backend
from fdrl.sim.rollout import rollout


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--per-cell", type=int, default=50)
    p.add_argument("--out", default="runs/calibration/expert_sweep.json")
    args = p.parse_args()
    cfg = config_from_args(args)
    grid = CellGrid(cfg.conditions)
    be = make_backend(cfg)
    ex = ScriptedExpert(cfg.expert, cfg.task, cfg.sim.toy.gripper_open_width)
    items, i = [], 0
    for cell in range(grid.n_cells):
        for _ in range(args.per_cell):
            s = seed_for("calib", 0, 0, i)
            i += 1
            items.append((grid.sample_in_cell(cell, np.random.default_rng(s)), s))
    res = rollout(be, items, lambda o, pr: ex.act(pr), cfg.task, record_obs=False)
    table = {}
    for cell in range(grid.n_cells):
        rs = [r for r in res if grid.cell_of(r.condition) == cell]
        succ = [r.success for r in rs]
        lens = [r.length for r in rs if r.success]
        table[cell] = {"bins": grid.bins(cell), "holdout": cell in grid.holdout,
                       "expert_success": float(np.mean(succ)), "mean_len": float(np.mean(lens)) if lens else None}
        print(f"cell {cell:2d} bins={grid.bins(cell)} H={cell in grid.holdout!s:5} "
              f"expert_success={np.mean(succ):.2f} mean_len={table[cell]['mean_len']}")
    train_lens = [r.length for r in res if r.success and grid.cell_of(r.condition) not in grid.holdout]
    summary = {"demo_equiv_steps": float(np.mean(train_lens)), "max_len": int(np.max(train_lens)),
               "horizon": cfg.task.horizon, "cells": table, "backend": be.info()}
    print(f"mean accepted length (training cells) = {summary['demo_equiv_steps']:.1f}; max = {summary['max_len']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1, default=str))
    be.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
