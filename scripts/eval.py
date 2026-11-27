#!/usr/bin/env python
"""Evaluate frozen checkpoints of an experiment.

Validation (allowed any time):
  python scripts/eval.py --config ... --split val --policy runs/x/.../policy.pt --pipeline-seed 0
Final test (ONE batch per experiment, after the analysis code is frozen):
  python scripts/eval.py --config <same configs> --split test --frozen-analysis $(git rev-parse HEAD)
It finds every final-round policy.pt of the experiment and writes reports under runs/<exp>/test/ only.
"""
import json
import sys
from pathlib import Path

from fdrl.cli import base_parser, config_from_args
from fdrl.conditions import CellGrid
from fdrl.evaluate import evaluate
from fdrl.policy.bc import MLPPolicy
from fdrl.sim.base import make_backend


def final_policies(root: Path):
    """Last round per (regime, budget, strategy, seed), plus scaling checkpoints."""
    out = []
    for seed_dir in root.glob("R*/B*/*/seed*"):
        rounds = sorted(seed_dir.glob("round*/policy.pt"), key=lambda p: int(p.parent.name[5:]))
        if rounds:
            out.append(rounds[-1])
    out += sorted(root.glob("scaling/*/N*/seed*/policy.pt"))
    return out


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--policy", help="single checkpoint (default: all final checkpoints of the experiment)")
    p.add_argument("--pipeline-seed", type=int, help="required with --policy")
    p.add_argument("--frozen-analysis", help="git SHA of the frozen analysis code (required for test)")
    args = p.parse_args()
    cfg = config_from_args(args)
    root = Path(cfg.experiment.run_root) / cfg.experiment.name
    grid = CellGrid(cfg.conditions)
    be = make_backend(cfg)
    pols = [Path(args.policy)] if args.policy else final_policies(root)
    for pp in pols:
        ps = args.pipeline_seed if args.policy else int(pp.parent.parent.name[4:] if pp.parent.name.startswith("round") else pp.parent.name[4:])
        pol = MLPPolicy.load(pp, device=cfg.train.device)
        rel = pp.parent.relative_to(root) if pp.is_relative_to(root) else Path(pp.parent.name)
        out = root / args.split / rel if args.split == "test" else pp.parent
        rep, _ = evaluate(pol.controller(), be, grid, cfg, split=args.split, pipeline_seed=ps,
                          policy_id=pol.policy_id(), out_dir=out, frozen_analysis=args.frozen_analysis)
        print(json.dumps({"policy": str(pp), "split": args.split, "overall": rep.overall(),
                          "holdout": rep.overall(sorted(grid.holdout)) if args.split == "test" else None}))
    be.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
