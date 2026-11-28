#!/usr/bin/env python
"""Run an experiment (scaling / equal_budget / iterative) from a config.

Examples:
  python scripts/run_experiment.py --config configs/experiment/smoke.yaml
  python scripts/run_experiment.py --config configs/env/isaac.yaml --config configs/experiment/exp2_equal_budget.yaml \
      experiment.pipeline_seeds=[3] experiment.strategies=[S2,S5]

Runs are resumable: finished rounds (metrics.json present) are skipped. To parallelise, launch one
process per pipeline seed (shared round-0 artifacts are per seed, so seeds never collide).
"""
import json
import sys
import time

from fdrl.cli import base_parser, config_from_args
from fdrl.loop import Pipeline


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    t0 = time.time()
    pipe = Pipeline(cfg)
    print(f"[fdrl] experiment={cfg.experiment.name} kind={cfg.experiment.kind} backend={cfg.sim.backend} "
          f"root={pipe.root}", flush=True)
    try:
        results = pipe.run_experiment()
        for r in results:
            d = r if isinstance(r, dict) else r.__dict__
            keys = ("strategy", "regime", "pipeline_seed", "round", "budget_demo_eq", "n", "n_episodes", "val_success")
            print(json.dumps({k: d[k] for k in keys if k in d}), flush=True)
        print(f"[fdrl] done in {time.time() - t0:.1f}s", flush=True)
    finally:
        # Isaac Sim keeps the process alive after main() returns unless the app is closed.
        close = getattr(pipe.backend, "close", None)
        if close is not None:
            close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
