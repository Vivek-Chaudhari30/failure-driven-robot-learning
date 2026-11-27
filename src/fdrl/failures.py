"""Failure taxonomy F1-F8 over privileged signals (EXPERIMENT_DESIGN section 6).

Predicates run over a recorded episode. The primary label is the earliest-triggered predicate;
ties are broken by the order F1..F8. All triggered flags and the failure timestep are kept.
This module labels *what happened* (detection). *Why* is the diagnosis module's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fdrl.config import FailureCfg, TaskCfg
from fdrl.sim.obs import tilt_from_quat_xyzw, wrap_sym90, yaw_from_quat_xyzw

CODES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")
NAMES = {
    "F1": "approach_failure",
    "F2": "misaligned_grasp",
    "F3": "missed_grasp",
    "F4": "slip",
    "F5": "drop",
    "F6": "disturbed_object",
    "F7": "unstable_motion",
    "F8": "timeout",
}


@dataclass
class FailureLabel:
    label: str | None  # primary code, None on success
    flags: list[str]  # all triggered codes (also recorded for successes, as near-miss info)
    t: int | None  # timestep of the primary failure
    diag: dict[str, float]


def _first(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def label_episode(
    priv: dict[str, np.ndarray],
    actions: np.ndarray,
    success: bool,
    task: TaskCfg,
    cfg: FailureCfg,
) -> FailureLabel:
    L = actions.shape[0]
    T = task.horizon
    ee = priv["ee_pos"].astype(np.float64)
    cube = priv["cube_pos"].astype(np.float64)
    z0 = float(priv["cube_z0"][0])
    lr = priv["contact_lr"]
    both = lr.min(axis=1) > cfg.contact_force_eps
    anyc = lr.max(axis=1) > cfg.contact_force_eps
    width = priv["gripper_width"]
    ee_yaw = yaw_from_quat_xyzw(priv["ee_quat"])
    cube_yaw = yaw_from_quat_xyzw(priv["cube_quat"])
    yaw_err = np.abs(wrap_sym90(cube_yaw - ee_yaw))
    dxy = np.linalg.norm(cube[:, :2] - ee[:, :2], axis=1)
    dz = np.abs(cube[:, 2] - ee[:, 2])
    trig: dict[str, int] = {}

    # F1 approach failure
    t_lim = min(int(cfg.approach_frac_T * T), L)
    near = (dxy < cfg.approach_xy) & (dz < cfg.approach_z)
    if not near[:t_lim].any() and (not success):
        trig["F1"] = max(t_lim - 1, 0)

    # gripper close events near the cube
    g = actions[:, 6] < 0
    close_ev = g & np.concatenate([[True], ~g[:-1]])
    near_close = close_ev & (dxy < 0.05) & (dz < 0.05)
    t_close = _first(near_close)
    diag: dict[str, float] = {
        "min_ee_cube_dist": float(np.min(np.hypot(dxy, dz))),
        "grasp_yaw_err_deg": float("nan"),
        "grasp_xy_err": float("nan"),
        "grasp_z_err": float("nan"),
    }
    if t_close is not None:
        diag["grasp_yaw_err_deg"] = float(math.degrees(yaw_err[t_close]))
        diag["grasp_xy_err"] = float(dxy[t_close])
        diag["grasp_z_err"] = float(dz[t_close])
        if (
            yaw_err[t_close] > math.radians(cfg.misalign_yaw_deg)
            or dxy[t_close] > cfg.misalign_xy
            or dz[t_close] > cfg.misalign_z
        ):
            trig["F2"] = t_close

    # F3 missed grasp: fully closed with no finger contact
    t3 = _first((width < cfg.closed_width_eps) & ~anyc)
    if t3 is not None:
        trig["F3"] = t3

    # F4 slip: in contact, relative displacement grows before reaching success height
    t_grasp = _first(both)
    max_slip = 0.0
    if t_grasp is not None:
        rel = (cube - ee) - (cube[t_grasp] - ee[t_grasp])
        disp = np.linalg.norm(rel, axis=1)
        was_contact = both | np.concatenate([[False], both[:-1]])
        below = cube[:, 2] < z0 + task.success_height
        valid = was_contact & (np.arange(L) >= t_grasp)
        max_slip = float(np.max(np.where(valid, disp, 0.0)))
        t4 = _first(valid & below & (disp > cfg.slip_disp))
        if t4 is not None:
            trig["F4"] = t4
    diag["max_slip"] = max_slip

    # F5 drop: rose above z0 + drop_height, then lost contact and fell
    height = cube[:, 2] - z0
    peak = np.maximum.accumulate(height)
    t5 = _first((peak > cfg.drop_height) & ~anyc & (height < peak - 0.02))
    if t5 is not None:
        trig["F5"] = t5
    diag["max_height"] = float(np.max(height))

    # F6 disturbed object before grasp
    pre = np.arange(L) < (t_grasp if t_grasp is not None else L)
    disp0 = np.linalg.norm(cube[:, :2] - cube[0, :2], axis=1)
    tilt = tilt_from_quat_xyzw(priv["cube_quat"])
    t6 = _first(pre & ((disp0 > cfg.disturb_disp) | (tilt > math.radians(cfg.disturb_tilt_deg))))
    if t6 is not None:
        trig["F6"] = t6
    diag["max_pregrasp_disp"] = float(np.max(np.where(pre, disp0, 0.0)))

    # F7 unstable motion: jerk / joint velocity / joint limit for more than N steps
    a = actions[:, :6].astype(np.float64)
    jerk = np.zeros(L)
    if L >= 3:
        jerk[2:] = np.linalg.norm(a[2:] - 2 * a[1:-1] + a[:-2], axis=1)
    jv = np.max(np.abs(priv["joint_vel"]), axis=1)
    bad = (jerk > cfg.jerk_limit) | (jv > cfg.joint_vel_limit) | (priv["at_joint_limit"] > 0.5)
    cum = np.cumsum(bad)
    t7 = _first(cum > cfg.unstable_steps)
    if t7 is not None:
        trig["F7"] = t7
    diag["mean_jerk"] = float(jerk.mean())
    diag["grasp_t"] = float(t_grasp) if t_grasp is not None else float("nan")

    flags = [c for c in CODES if c in trig]
    if success:
        return FailureLabel(None, flags, None, diag)
    if not trig:
        trig["F8"] = L - 1
        flags = ["F8"]
    order = {c: i for i, c in enumerate(CODES)}
    label = min(trig, key=lambda c: (trig[c], order[c]))
    return FailureLabel(label, flags, trig[label], diag)
