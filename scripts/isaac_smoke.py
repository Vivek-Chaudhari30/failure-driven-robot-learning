#!/usr/bin/env python
"""First thing to run on a GPU machine (M1/M2 exit check for the Isaac Lab adapter).

1. Launches the Isaac Lab lift env through fdrl.sim.isaaclab_lift.
2. Applies a sweep of conditions θ and reads back cube pose and friction -> checks they match.
3. Runs the scripted expert for a few episodes per friction level and prints success.

  python scripts/isaac_smoke.py --config configs/env/isaac.yaml sim.num_envs=16
"""
import math
import sys

import numpy as np

from fdrl.cli import base_parser, config_from_args
from fdrl.conditions import Condition, seed_for
from fdrl.expert.scripted import ScriptedExpert
from fdrl.sim.base import make_backend
from fdrl.sim.obs import wrap_sym90, yaw_from_quat_xyzw
from fdrl.sim.rollout import rollout


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    be = make_backend(cfg)
    n = be.num_envs
    rng = np.random.default_rng(0)
    conds = [Condition(rng.uniform(-0.1, 0.1), rng.uniform(-0.25, 0.25), math.radians(rng.uniform(-45, 45)),
                       float(np.exp(rng.uniform(np.log(0.1), 0.0)))) for _ in range(n)]
    priv = be.reset(conds, [seed_for("calib", 0, 0, i) for i in range(n)])
    nx, ny = cfg.task.nominal_cube_xy
    ok = True
    for i, c in enumerate(conds):
        dx = priv["cube_pos"][i, 0] - (nx + c.x)
        dy = priv["cube_pos"][i, 1] - (ny + c.y)
        dyaw = wrap_sym90(yaw_from_quat_xyzw(priv["cube_quat"][i]) - c.yaw)
        good = abs(dx) < 0.005 and abs(dy) < 0.005 and abs(dyaw) < math.radians(2)
        ok &= good
        print(f"env {i}: applied θ={c} | read-back dxy=({dx:+.4f},{dy:+.4f}) dyaw={math.degrees(dyaw):+.2f}° "
              f"mu={priv['mu'][i]:.3f} z0={priv['cube_z0'][i]:.4f} {'OK' if good else 'MISMATCH'}")
    print("pose read-back:", "PASS" if ok else "FAIL")
    ex = ScriptedExpert(cfg.expert, cfg.task)
    for mu in (0.12, 0.2, 0.4, 1.0):
        items = [(Condition(0.0, 0.0, 0.3, mu), seed_for("calib", 1, 0, j)) for j in range(n)]
        res = rollout(be, items, lambda o, p: ex.act(p), cfg.task, record_obs=False)
        print(f"mu={mu}: expert success {np.mean([r.success for r in res]):.2f}, mean length {np.mean([r.length for r in res]):.0f}")
    be.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
