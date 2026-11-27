#!/usr/bin/env python
"""Calibration sweep (EXPERIMENT_DESIGN section 3): scripted-expert success, episode length and
failure types per cell, on CALIBRATION seeds only. Optionally a friction sweep at random poses.

  python scripts/calibrate.py --per-cell 10 [--mu-sweep] [--config configs/env/isaac.yaml]

Prints one line per cell, a summary per friction bin, overall failure counts, and the mean
accepted-demo length under uniform sampling (use it for experiment.demo_equiv_steps).
"""
import dataclasses
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from fdrl.cli import base_parser, config_from_args
from fdrl.conditions import CellGrid, Condition, seed_for
from fdrl.expert.scripted import ScriptedExpert
from fdrl.failures import label_episode
from fdrl.sim.base import make_backend
from fdrl.sim.rollout import rollout


def _labels(res, cfg) -> list[str | None]:
    return [label_episode(r.privileged, r.actions, r.success, cfg.task, cfg.failures).label for r in res]


def _causes(res, cfg) -> list[str | None]:
    """Primary label with F7 (unstable motion) switched off: the outcome underneath the jerk flag."""
    fc = dataclasses.replace(cfg.failures, unstable_steps=10**9)
    return [label_episode(r.privileged, r.actions, r.success, cfg.task, fc).label for r in res]


def _fmt_counts(c: Counter) -> str:
    return " ".join(f"{k}:{v}" for k, v in sorted(c.items())) or "-"


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--per-cell", type=int, default=50)
    p.add_argument("--mu-sweep", action="store_true", help="also sweep friction at random training poses")
    p.add_argument("--sweep-n", type=int, default=32, help="episodes per friction value in the sweep")
    p.add_argument("--out", default="runs/calibration/expert_sweep.json")
    args = p.parse_args()
    cfg = config_from_args(args)
    grid = CellGrid(cfg.conditions)
    be = make_backend(cfg)
    ex = ScriptedExpert(cfg.expert, cfg.task, cfg.sim.toy.gripper_open_width)
    ctrl = lambda o, pr: ex.act(pr)  # noqa: E731

    items, i = [], 0
    for cell in range(grid.n_cells):
        for _ in range(args.per_cell):
            s = seed_for("calib", 0, 0, i)
            i += 1
            items.append((grid.sample_in_cell(cell, np.random.default_rng(s)), s))
    res = rollout(be, items, ctrl, cfg.task, record_obs=False)
    labels = _labels(res, cfg)
    causes = _causes(res, cfg)

    table, by_mu = {}, {}
    for cell in range(grid.n_cells):
        idx = [k for k, r in enumerate(res) if grid.cell_of(r.condition) == cell]
        succ = [res[k].success for k in idx]
        lens = [res[k].length for k in idx if res[k].success]
        fails = Counter(labels[k] for k in idx if not res[k].success)
        cause = Counter(causes[k] for k in idx if not res[k].success)
        q, yb, mb = grid.bins(cell)
        table[cell] = {"bins": (q, yb, mb), "holdout": cell in grid.holdout, "expert_success": float(np.mean(succ)),
                       "mean_len": float(np.mean(lens)) if lens else None, "failures": dict(fails),
                       "causes": dict(cause)}
        by_mu.setdefault(mb, []).extend(succ)
        ml = f"{np.mean(lens):5.0f}" if lens else "  -  "
        print(f"cell {cell:2d} bins={(q, yb, mb)} H={'Y' if cell in grid.holdout else 'n'} "
              f"success={np.mean(succ):.2f} len={ml} fails: {_fmt_counts(fails)} | cause: {_fmt_counts(cause)}")

    edges = np.exp(grid.logmu_edges)
    for mb in sorted(by_mu):
        print(f"mu bin {mb} [{edges[mb]:.3f}, {edges[mb + 1]:.3f}): expert success {np.mean(by_mu[mb]):.2f}")
    all_fails = Counter(l for r, l in zip(res, labels, strict=True) if not r.success)
    print(f"failure totals: {_fmt_counts(all_fails)}")
    print(f"cause totals (F7 ignored): {_fmt_counts(Counter(c for r, c in zip(res, causes, strict=True) if not r.success))}")
    train_lens = [r.length for r in res if r.success and grid.cell_of(r.condition) not in grid.holdout]
    summary = {"demo_equiv_steps": float(np.mean(train_lens)) if train_lens else None,
               "max_len": int(np.max(train_lens)) if train_lens else None,
               "horizon": cfg.task.horizon, "cells": table, "backend": be.info(), "failure_totals": dict(all_fails)}
    if train_lens:
        lens_arr = np.array(train_lens)
        print(f"mean accepted length (training cells) = {lens_arr.mean():.1f}; "
              f"p90 = {np.percentile(lens_arr, 90):.0f}; max = {lens_arr.max()}; horizon = {cfg.task.horizon}")

    if args.mu_sweep:
        sweep = {}
        rng = np.random.default_rng(123)
        mus = [0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40, 0.55, 0.75, 1.00]
        j = 0
        for mu in mus:
            its = []
            for _ in range(args.sweep_n):
                x = rng.uniform(*cfg.conditions.x_range)
                y = rng.uniform(*cfg.conditions.y_range)
                yaw = math.radians(rng.uniform(*cfg.conditions.yaw_range_deg))
                its.append((Condition(float(x), float(y), float(yaw), mu), seed_for("calib", 5, 0, j)))
                j += 1
            rr = rollout(be, its, ctrl, cfg.task, record_obs=False)
            ll = _causes(rr, cfg)
            fails = Counter(l for r, l in zip(rr, ll, strict=True) if not r.success)
            ok_len = [r.length for r in rr if r.success]
            sweep[mu] = {"success": float(np.mean([r.success for r in rr])), "failures": dict(fails)}
            print(f"mu={mu:.2f}: success {np.mean([r.success for r in rr]):.2f}  "
                  f"len {np.mean(ok_len) if ok_len else float('nan'):.0f}  cause (F7 ignored): {_fmt_counts(fails)}")
        summary["mu_sweep"] = sweep

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1, default=str))
    be.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
