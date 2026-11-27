#!/usr/bin/env python
"""Print what the scripted expert sees and does, step by step (for tuning it on a new simulator).

  python scripts/debug_expert.py --config configs/env/isaac.yaml sim.num_envs=4
  python scripts/debug_expert.py sim.backend=toy sim.num_envs=4        # reference behaviour

Prints env 0 every --every steps (phase, end-effector vs cube, yaw error, gripper width, finger
contact forces, action), then a per-env summary. Paste the output to diagnose a failing expert.
"""
import math
import sys

import numpy as np

from fdrl.cli import base_parser, config_from_args
from fdrl.conditions import Condition, seed_for
from fdrl.expert.scripted import ScriptedExpert
from fdrl.sim.base import make_backend
from fdrl.sim.obs import wrap_sym90, yaw_from_quat_xyzw


def fmt(v) -> str:
    return ",".join(f"{x:.3f}" for x in v)


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--every", type=int, default=5)
    p.add_argument("--steps", type=int, default=0, help="default: task horizon")
    args = p.parse_args()
    cfg = config_from_args(args)
    be = make_backend(cfg)
    n = be.num_envs
    base = [Condition(0.0, 0.0, 0.0, 1.0), Condition(0.05, -0.10, 0.3, 0.5),
            Condition(-0.05, 0.15, -0.5, 0.3), Condition(0.08, 0.20, 0.7, 0.8)]
    conds = [base[i % len(base)] for i in range(n)]
    priv = be.reset(conds, [seed_for("calib", 2, 0, i) for i in range(n)])
    ex = ScriptedExpert(cfg.expert, cfg.task, cfg.sim.toy.gripper_open_width)
    T = args.steps or cfg.task.horizon
    z0 = priv["cube_z0"].copy()
    max_f = np.zeros(n)
    max_h = np.zeros(n)
    phases_seen = [set() for _ in range(n)]
    print(f"cube_z0={z0[0]:.4f}  start ee={np.round(priv['ee_pos'][0], 4)}  cube={np.round(priv['cube_pos'][0], 4)}")
    print(" t  phase     ee_xyz                 cube_xyz               dxy    dz     yawerr  width  fL     fR     act(dx,dy,dz,dyaw,grip)")
    for t in range(T):
        act, phase = ex.act(priv, return_phase=True)
        for i in range(n):
            phases_seen[i].add(phase[i])
        ee, cube = priv["ee_pos"], priv["cube_pos"]
        f = priv["contact_lr"]
        max_f = np.maximum(max_f, f.min(axis=1))
        max_h = np.maximum(max_h, cube[:, 2] - z0)
        if t % args.every == 0:
            dxy = np.linalg.norm(cube[0, :2] - ee[0, :2])
            dz = ee[0, 2] - cube[0, 2]
            ye = math.degrees(wrap_sym90(yaw_from_quat_xyzw(priv["cube_quat"][0]) - yaw_from_quat_xyzw(priv["ee_quat"][0])))
            a = act[0]
            print(f"{t:3d} {phase[0]:9s} {fmt(ee[0]):22s} "
                  f"{fmt(cube[0]):22s} {dxy:.4f} {dz:+.4f} {ye:+6.1f}  "
                  f"{priv['gripper_width'][0]:.4f} {f[0, 0]:6.2f} {f[0, 1]:6.2f} "
                  f"({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f},{a[5]:+.2f},{a[6]:+.0f})")
        priv = be.step(act)
    print("\nper-env summary:")
    yaw_end = np.degrees(wrap_sym90(yaw_from_quat_xyzw(priv["cube_quat"]) - yaw_from_quat_xyzw(priv["ee_quat"])))
    dxy_end = np.linalg.norm(priv["cube_pos"][:, :2] - priv["ee_pos"][:, :2], axis=1)
    for i in range(n):
        print(f"env {i}: θ={conds[i]}  max_min_finger_force={max_f[i]:.2f}N  max_cube_height={max_h[i]:.4f}m  "
              f"final_dxy={dxy_end[i]:.4f} final_yaw_err={yaw_end[i]:+.1f}deg  phases={sorted(phases_seen[i])}")
    be.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
