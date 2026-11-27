"""Numpy stand-in for the Franka cube-lift task.

Purpose: run and test the *whole* pipeline (expert, recording, training, evaluation, diagnosis,
acquisition, analysis) on a laptop without Isaac Sim. It is a kinematic end-effector with a
simple friction-limited grasp model, NOT a physics simulator, and results from it are not
research results. It plants the same root causes as EXPERIMENT_DESIGN section 3:

* position / yaw  -> the policy must reach and align precisely (coverage-limited);
* friction mu     -> holding the cube needs 2*mu*F >= m*(g + a): lifting fast slips at low mu,
                     so an expert that knows mu lifts gently while a policy that cannot see mu
                     must guess (observability-limited);
* mu below m*g/(2F) cannot be held at all (expert-infeasible band).

All arrays are batched over environments. Quaternions are XYZW.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from fdrl.conditions import Condition
from fdrl.config import TaskCfg, ToySimCfg
from fdrl.sim.obs import quat_xyzw_from_yaw, wrap_sym90


class ToyLiftBackend:
    def __init__(self, task: TaskCfg, cfg: ToySimCfg, num_envs: int):
        self.task = task
        self.cfg = cfg
        self.num_envs = num_envs
        self.dt = task.policy_dt
        self.half = task.cube_half
        self.lo = np.array(cfg.workspace_lo)
        self.hi = np.array(cfg.workspace_hi)
        self._t = 0
        self._noise = np.zeros((num_envs, task.horizon + 2, 6))
        n = num_envs
        self.ee = np.zeros((n, 3))
        self.ee_yaw = np.zeros(n)
        self.ee_vel = np.zeros((n, 3))
        self.width = np.full(n, cfg.gripper_open_width)
        self.cube = np.zeros((n, 3))
        self.cube_yaw = np.zeros(n)
        self.cube_vel = np.zeros((n, 3))
        self.cube_tilt = np.zeros(n)
        self.grasped = np.zeros(n, dtype=bool)
        self.rel0 = np.zeros(n)
        self.f_eff = np.zeros(n)
        self.mu = np.ones(n)
        self.joint_pos = np.zeros((n, 9))
        self.joint_vel = np.zeros((n, 9))
        self.at_limit = np.zeros(n)

    # ------------------------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        return {"backend": "toy", "note": "numpy stand-in; not a physics simulator"}

    def close(self) -> None:
        pass

    def reset(self, conditions: Sequence[Condition], seeds: Sequence[int]) -> dict[str, np.ndarray]:
        n = self.num_envs
        if len(conditions) != n or len(seeds) != n:
            raise ValueError(f"reset needs exactly {n} conditions and seeds")
        cfg = self.cfg
        nx, ny = self.task.nominal_cube_xy
        for i, (c, s) in enumerate(zip(conditions, seeds, strict=True)):
            rng = np.random.default_rng(s)
            self.ee[i] = np.array(cfg.ee_start) + rng.uniform(-0.01, 0.01, size=3)
            self.cube[i] = (nx + c.x, ny + c.y, self.half)
            self.cube_yaw[i] = c.yaw
            self.mu[i] = c.mu
            self._noise[i] = rng.normal(0.0, cfg.action_noise, size=self._noise.shape[1:])
        self.ee_yaw[:] = 0.0
        self.ee_vel[:] = 0.0
        self.width[:] = cfg.gripper_open_width
        self.cube_vel[:] = 0.0
        self.cube_tilt[:] = 0.0
        self.grasped[:] = False
        self.rel0[:] = 0.0
        self.f_eff[:] = 0.0
        self.at_limit[:] = 0.0
        self._t = 0
        self.joint_pos = self._joints()
        self.joint_vel[:] = 0.0
        return self._priv()

    # ------------------------------------------------------------------------------------
    def step(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        cfg, task, dt, half = self.cfg, self.task, self.dt, self.half
        a = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0).copy()
        t = min(self._t, self._noise.shape[1] - 1)
        a[:, :6] += self._noise[:, t]
        a[:, :6] = np.clip(a[:, :6], -1.0, 1.0)

        dpos = a[:, :3] * task.pos_scale
        dyaw = a[:, 5] * task.rot_scale
        ee_old = self.ee.copy()
        target = ee_old + dpos
        clipped = np.clip(target, self.lo, self.hi)
        self.at_limit = np.any(np.abs(clipped - target) > 1e-9, axis=1).astype(np.float64)
        new_ee = clipped

        offset = self.cube[:, :2] - new_ee[:, :2]
        dist = np.linalg.norm(offset, axis=1)
        yaw_err = wrap_sym90(self.cube_yaw - (self.ee_yaw + dyaw))
        cube_top = self.cube[:, 2] + half
        free = ~self.grasped

        # --- fingers colliding with the cube while not grasped (pushes / blocks) -----------
        open_enough = self.width >= 2 * half + 0.01
        straddle = open_enough & (dist < 2 * cfg.grasp_xy_tol) & (np.abs(yaw_err) < math.radians(25.0))
        low = new_ee[:, 2] < cube_top
        hit = free & low & (dist < half + 0.012) & ~straddle
        if np.any(hit):
            new_ee[hit, 2] = np.maximum(new_ee[hit, 2], cube_top[hit])
            push = np.zeros((self.num_envs, 2))
            push[hit] = 0.5 * dpos[hit, :2]
            away = np.where(dist[:, None] > 1e-6, -offset / np.maximum(dist[:, None], 1e-6), 0.0)
            push[hit] -= 0.002 * away[hit]  # offset points ee->cube; push the cube away from ee
            self.cube[hit, :2] += push[hit]
            self.cube_yaw[hit] += 0.02 * np.sign(yaw_err[hit] + 1e-9)

        self.ee_yaw = self.ee_yaw + dyaw

        # --- gripper ----------------------------------------------------------------------
        closing = a[:, 6] < 0.0
        w_target = np.where(closing, 0.0, cfg.gripper_open_width)
        new_width = self.width + np.clip(w_target - self.width, -cfg.gripper_speed, cfg.gripper_speed)

        z_ok = np.abs(new_ee[:, 2] - self.cube[:, 2]) < cfg.grasp_z_tol
        between = free & z_ok & (dist < cfg.grasp_xy_tol)
        aligned = np.abs(yaw_err) < math.radians(cfg.grasp_yaw_tol_deg)
        jam = np.abs(yaw_err) >= math.radians(cfg.jam_yaw_deg)
        touching = closing & between & (new_width <= 2 * half)
        # Moderately misaligned: the fingers rotate the cube toward alignment while closing.
        rot = closing & between & ~aligned & ~jam & (new_width <= 2 * half + 0.01)
        self.cube_yaw[rot] -= np.sign(yaw_err[rot]) * 0.04
        # Badly misaligned: jam on the corners, no grasp; the cube is shoved.
        jammed = touching & jam
        self.cube[jammed, :2] += 0.002 * np.stack([np.cos(self.ee_yaw[jammed]), np.sin(self.ee_yaw[jammed])], -1)
        new_grasp = touching & aligned
        # Off-centre closing: the fingers sweep the cube away instead of grasping it.
        sweep = free & closing & z_ok & (dist >= cfg.grasp_xy_tol) & (dist < 2 * cfg.grasp_xy_tol) & (new_width < dist + half)
        if np.any(sweep):
            away = offset[sweep] / np.maximum(dist[sweep, None], 1e-6)
            self.cube[sweep, :2] += 0.003 * away
        blocked = (new_grasp | (self.grasped & closing)) | (touching & ~jam & ~aligned)
        new_width = np.where(blocked, np.maximum(new_width, 2 * half), new_width)

        if np.any(new_grasp):
            yaw_frac = np.abs(yaw_err[new_grasp]) / math.radians(cfg.grasp_yaw_tol_deg)
            xy_frac = dist[new_grasp] / cfg.grasp_xy_tol
            self.f_eff[new_grasp] = cfg.grip_force * (1 - 0.6 * yaw_frac**2) * (1 - 0.5 * xy_frac**2)
            self.cube[new_grasp, :2] = new_ee[new_grasp, :2] + 0.3 * (self.cube[new_grasp, :2] - new_ee[new_grasp, :2])
            self.rel0[new_grasp] = self.cube[new_grasp, 2] - new_ee[new_grasp, 2]
            self.cube_vel[new_grasp] = 0.0
            self.grasped |= new_grasp
        released = self.grasped & ~closing
        self.grasped &= ~released
        self.f_eff[released] = 0.0

        # --- cube dynamics ------------------------------------------------------------------
        g, m = cfg.gravity, cfg.cube_mass
        ve = (new_ee[:, 2] - ee_old[:, 2]) / dt
        z0 = half
        gi = self.grasped
        if np.any(gi):
            vc = self.cube_vel[gi, 2]
            a_need = (ve[gi] - vc) / dt
            cap = 2.0 * self.mu[gi] * self.f_eff[gi] / m
            acc = np.clip(a_need + g, -cap, cap) - g  # friction can supply at most cap (either way)
            vc = vc + acc * dt
            cz = self.cube[gi, 2] + vc * dt
            on_table = cz <= z0
            cz = np.where(on_table, z0, cz)
            vc = np.where(on_table, np.maximum(vc, 0.0), vc)
            self.cube[gi, 2] = cz
            self.cube_vel[gi, 2] = vc
            self.cube[gi, :2] += new_ee[gi, :2] - ee_old[gi, :2]
            self.cube_yaw[gi] += dyaw[gi]
            slip = (cz - new_ee[gi, 2]) - self.rel0[gi]
            lost = np.abs(slip) > cfg.max_slip
            idx = np.flatnonzero(gi)[lost]
            self.grasped[idx] = False
            self.f_eff[idx] = 0.0
        fi = ~self.grasped
        if np.any(fi):
            vz = self.cube_vel[fi, 2] - g * dt
            cz = self.cube[fi, 2] + vz * dt
            landed = cz <= z0
            self.cube[fi, 2] = np.where(landed, z0, cz)
            self.cube_vel[fi, 2] = np.where(landed, 0.0, vz)

        self.width = new_width
        self.ee_vel = (new_ee - ee_old) / dt
        self.ee = new_ee
        jp = self._joints()
        self.joint_vel = (jp - self.joint_pos) / dt
        self.joint_pos = jp
        self._t += 1
        return self._priv()

    # ------------------------------------------------------------------------------------
    def _joints(self) -> np.ndarray:
        """Surrogate 9-D joint state (7 arm + 2 fingers): a smooth function of the EE pose."""
        x, y, z = self.ee[:, 0], self.ee[:, 1], self.ee[:, 2]
        r = np.hypot(x, y)
        arm = np.stack(
            [np.arctan2(y, x), r, z, self.ee_yaw, np.sin(self.ee_yaw) * r, z * r, x - y], axis=-1
        )
        fingers = np.stack([self.width / 2, self.width / 2], axis=-1)
        return np.concatenate([arm, fingers], axis=-1)

    def _priv(self) -> dict[str, np.ndarray]:
        n = self.num_envs
        nx, ny = self.task.nominal_cube_xy
        contact = np.where(self.grasped, self.f_eff, 0.0)
        on_table = self.cube[:, 2] <= self.half + 1e-6
        off = (np.abs(self.cube[:, 0] - nx) > 0.4) | (np.abs(self.cube[:, 1] - ny) > 0.6)
        return {
            "joint_pos": self.joint_pos.copy(),
            "joint_vel": self.joint_vel.copy(),
            "ee_pos": self.ee.copy(),
            "ee_quat": quat_xyzw_from_yaw(self.ee_yaw),
            "ee_lin_vel": self.ee_vel.copy(),
            "gripper_width": self.width.copy(),
            "cube_pos": self.cube.copy(),
            "cube_quat": quat_xyzw_from_yaw(self.cube_yaw),
            "cube_lin_vel": self.cube_vel.copy(),
            "contact_lr": np.stack([contact, contact], axis=-1),
            "contact_table": np.where(on_table, self.cfg.cube_mass * self.cfg.gravity, 0.0),
            "at_joint_limit": self.at_limit.copy(),
            "invalid": np.zeros(n),
            "off_table": off.astype(np.float64),
            "mu": self.mu.copy(),
            "cube_z0": np.full(n, self.half),
        }
