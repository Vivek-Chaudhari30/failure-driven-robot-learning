#!/usr/bin/env python
"""Aggregate an experiment: paired comparisons vs S2, bootstrap CIs, Holm correction, plots.

  python scripts/analyze.py --config configs/experiment/smoke.yaml [--split val|test]

Validation-split numbers are for development only; reported claims use --split test.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from fdrl.analysis import plots
from fdrl.analysis.stats import holm, iqm, paired_bootstrap
from fdrl.cli import base_parser, config_from_args
from fdrl.conditions import CellGrid
from fdrl.evaluate import EvalReport


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--reference", default="S2")
    args = p.parse_args()
    cfg = config_from_args(args)
    root = Path(cfg.experiment.run_root) / cfg.experiment.name
    grid = CellGrid(cfg.conditions)
    out_dir = root / "analysis" / args.split
    summary: dict = {"split": args.split, "comparisons": {}}

    # value[(regime, budget, strategy)][seed] = success of the final round
    vals: dict = defaultdict(dict)
    for seed_dir in root.glob("R*/B*/*/seed*"):
        rounds = sorted(seed_dir.glob("round*"), key=lambda p: int(p.name[5:]))
        if not rounds:
            continue
        last = rounds[-1]
        rep_path = last / "val_report.json" if args.split == "val" else root / "test" / last.relative_to(root) / "test_report.json"
        if not rep_path.exists():
            continue
        rep = EvalReport.load(rep_path)
        regime, budget, strat = seed_dir.parts[-4], seed_dir.parts[-3], seed_dir.parts[-2]
        vals[(regime, budget, strat)][int(seed_dir.name[4:])] = rep.overall(grid.train_cells)

    pvals = {}
    for (regime, budget, strat), by_seed in sorted(vals.items()):
        ref = vals.get((regime, budget, args.reference), {})
        seeds = sorted(set(by_seed) & set(ref))
        entry = {"mean": float(np.mean(list(by_seed.values()))), "iqm": iqm(list(by_seed.values())), "n_seeds": len(by_seed)}
        if strat != args.reference and len(seeds) >= 2:
            ci = paired_bootstrap([by_seed[s] for s in seeds], [ref[s] for s in seeds])
            entry["diff_vs_ref"] = {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "p": ci.p_value}
            pvals[f"{regime}/{budget}/{strat}"] = ci.p_value
        summary["comparisons"][f"{regime}/{budget}/{strat}"] = entry
    if pvals:
        for k, v in holm(pvals).items():
            summary["comparisons"][k]["diff_vs_ref"]["p_holm"] = v

    groups = defaultdict(dict)
    for (regime, budget, strat), by_seed in vals.items():
        groups[(regime, budget)][strat] = list(by_seed.values())
    for (regime, budget), d in groups.items():
        plots.strategy_comparison(dict(sorted(d.items())), out_dir / f"strategies_{regime}_{budget}.png",
                                  title=f"{regime} {budget} ({args.split})")

    # scaling curves
    curves: dict = defaultdict(lambda: defaultdict(list))
    for m in root.glob("scaling/*/N*/seed*/metrics.json"):
        d = json.loads(m.read_text())
        curves[d["regime"]][d["n"]].append(d["val_success"])
    if curves:
        plots.scaling_curve({k: dict(v) for k, v in curves.items()}, out_dir / "scaling.png")

    # coverage heatmaps of acquired data
    for plan in root.glob("R*/B*/*/seed0/round1/plan.json"):
        d = json.loads(plan.read_text())
        counts = {int(k): v for k, v in d.get("accepted_per_cell", {}).items()}
        name = "_".join(plan.parts[-6:-3])
        plots.coverage_heatmap(grid, counts, out_dir / f"coverage_{name}.png", title=name)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary["comparisons"].items():
        dv = v.get("diff_vs_ref")
        tail = f"  diff={dv['mean']:+.3f} [{dv['lo']:+.3f},{dv['hi']:+.3f}] p_holm={dv.get('p_holm', float('nan')):.3f}" if dv else ""
        print(f"{k:28s} mean={v['mean']:.3f} n={v['n_seeds']}{tail}")
    print(f"plots and summary in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
