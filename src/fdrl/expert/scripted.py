"""State-feedback scripted grasp-lift expert (ARCHITECTURE AD6).

The phase is inferred from the current privileged state, never from an internal timer, so the
expert can label ANY state. That is what DAgger relabelling (S6) and the ensemble diagnosis need.

Phases, in priority order:
  LIFT     cube held by both fingers -> lift with a mu-aware velocity/acceleration profile
  RECOVER  gripper closed/closing without a grasp and not about to grasp -> open and back off
  CLOSE    aligned at grasp height -> close
  DESCEND  aligned above the cube -> descend to grasp height
  APPROACH otherwise -> go to the pre-grasp pose above the cube (rising first if too low)

Actions are normalized: Δpos/scale, Δyaw/scale, gripper +1 open / -1 close.
"""

from __future__ import annotations

import math

import numpy as np

from fdrl.config import ExpertCfg, TaskCfg
from fdrl.sim.obs import wrap_sym90, yaw_from_quat_xyzw

PHASES = ("lift", "recover", "close", "descend", "approach")


class ScriptedExpert:
    def __init__(self, cfg: ExpertCfg, task: TaskCfg, open_width: float = 0.08):
        self.cfg = cfg
        self.task = task
        self.open_width = open_width

    def lift_profile(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(target lift speed, max acceleration) as functions of friction."""
        c = self.cfg
        mu_eff = np.asarray(mu) if c.mu_aware else np.ones_like(mu)
        a_cap = c.grip_accel_per_mu * mu_eff - c.gravity
        a_max = np.maximum(c.accel_safety * a_cap, 0.05)
        frac = np.clip((mu_eff - c.mu_slow) / max(c.mu_full_speed - c.mu_slow, 1e-6), 0.0, 1.0)
        v = c.lift_vmin + (c.lift_vmax - c.lift_vmin) * frac
        return v, a_max

    def act(self, priv: dict[str, np.ndarray], return_phase: bool = False):
        c, task = self.cfg, self.task
        n = priv["ee_pos"].shape[0]
        ee, cube = priv["ee_pos"], priv["cube_pos"]
        ee_yaw = yaw_from_quat_xyzw(priv["ee_quat"])
        cube_yaw = yaw_from_quat_xyzw(priv["cube_quat"])
        yaw_err = wrap_sym90(cube_yaw - ee_yaw)
        xy_err = cube[:, :2] - ee[:, :2]
        dxy = np.linalg.norm(xy_err, axis=1)
        width = priv["gripper_width"]
        z0 = priv["cube_z0"]
        half = task.cube_half
        in_hand = priv["contact_lr"].min(axis=1) > 0.1

        target = cube.copy()
        target[:, 2] = cube[:, 2] + c.pregrasp_height
        grip = np.ones(n)
        move_xy = np.ones(n, dtype=bool)
        dz_override = np.full(n, np.nan)

        tight = (dxy < c.descend_xy_tol) & (np.abs(yaw_err) < math.radians(c.descend_yaw_tol_deg))
        below_pre = ee[:, 2] < cube[:, 2] + c.pregrasp_height - 0.01
        keep = below_pre & (dxy < c.descend_keep_xy_tol) & (np.abs(yaw_err) < math.radians(c.descend_keep_yaw_tol_deg))
        descend_ok = tight | keep
        at_grasp = tight & (np.abs(ee[:, 2] - cube[:, 2]) < c.close_z_tol)
        closing = width < self.open_width - 0.002
        still_closing = closing & at_grasp & (width > 2 * half - 0.002)

        phase = np.full(n, "approach", dtype=object)
        # APPROACH (default): rise first if low and far from the cube.
        low_far = (ee[:, 2] < cube[:, 2] + half + 0.02) & (dxy > 0.02)
        move_xy &= ~low_far
        # DESCEND
        d = descend_ok & ~closing
        target[d, 2] = cube[d, 2]
        # Drifted beyond the tight tolerance: hold height and correct xy/yaw before going lower.
        hold = d & ~tight
        target[hold, 2] = ee[hold, 2]
        phase[d] = "descend"
        # CLOSE
        cl = (at_grasp & ~closing) | still_closing
        target[cl, 2] = cube[cl, 2]
        grip[cl] = -1.0
        phase[cl] = "close"
        # RECOVER
        rec = closing & ~in_hand & ~still_closing
        target[rec, :2] = ee[rec, :2]
        target[rec, 2] = cube[rec, 2] + c.pregrasp_height
        grip[rec] = 1.0
        phase[rec] = "recover"
        # LIFT
        li = in_hand
        if np.any(li):
            v_t, a_max = self.lift_profile(priv["mu"][li])
            height = cube[li, 2] - z0[li]
            v_goal = np.where(height < c.lift_target_height, v_t, 0.0)
            v_now = priv["ee_lin_vel"][li, 2]
            v_cmd = v_now + np.clip(v_goal - v_now, -a_max * task.policy_dt, a_max * task.policy_dt)
            dz_override[li] = v_cmd * task.policy_dt
            target[li, :2] = ee[li, :2]
            grip[li] = -1.0
            phase[li] = "lift"

        act = np.zeros((n, 7))
        err = target - ee
        step = c.gain * err
        step[~move_xy, :2] = 0.0
        has_dz = ~np.isnan(dz_override)
        step[has_dz, 2] = dz_override[has_dz]
        act[:, :3] = step / task.pos_scale
        yaw_cmd = np.where(phase == "lift", 0.0, c.gain * yaw_err)
        act[:, 5] = yaw_cmd / task.rot_scale
        act[:, :6] = np.clip(act[:, :6], -1.0, 1.0)
        act[:, 6] = grip
        if return_phase:
            return act, phase
        return act

    __call__ = act
