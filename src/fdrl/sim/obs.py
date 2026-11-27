"""Simulator-agnostic observation construction (EXPERIMENT_DESIGN section 2).

Every backend returns the same *privileged state* dictionary of batched numpy arrays. The policy
observation is built from it here, so the observation definition is identical across simulators.
Quaternions are XYZW everywhere in fdrl (Isaac Lab 3.0 convention); adapters convert at the
boundary.
"""

from __future__ import annotations

import math

import numpy as np

# Privileged-state keys every backend must provide (batched, leading dim = num_envs).
PRIV_KEYS: dict[str, tuple[int, ...]] = {
    "joint_pos": (9,),
    "joint_vel": (9,),
    "ee_pos": (3,),
    "ee_quat": (4,),  # xyzw
    "ee_lin_vel": (3,),
    "gripper_width": (),
    "cube_pos": (3,),
    "cube_quat": (4,),  # xyzw
    "cube_lin_vel": (3,),
    "contact_lr": (2,),  # finger-cube normal force magnitude, left/right (N)
    "contact_table": (),  # cube-table force magnitude (N)
    "at_joint_limit": (),  # 1.0 if any joint is at its limit
    "invalid": (),  # 1.0 on NaN / physics explosion
    "off_table": (),  # 1.0 if the cube left the table
    "mu": (),  # applied friction (latent; never in the policy observation by default)
    "cube_z0": (),  # cube resting height at reset
}

# Ordered observation layout: (key, dim). Total 41.
OBS_LAYOUT: list[tuple[str, int]] = [
    ("joint_pos", 9),
    ("joint_vel", 9),
    ("ee_pos", 3),
    ("ee_quat", 4),
    ("gripper_width", 1),
    ("cube_pos", 3),
    ("cube_yaw4", 2),  # (sin 4ψ, cos 4ψ): respects the cube's 90° symmetry
    ("cube_rel_ee", 3),
    ("prev_action", 7),
]
OBS_DIM = sum(d for _, d in OBS_LAYOUT)
ACTION_DIM = 7  # Δpos(3), Δrot axis-angle(3), gripper(+1 open / -1 close)


def obs_slices() -> dict[str, slice]:
    out, i = {}, 0
    for k, d in OBS_LAYOUT:
        out[k] = slice(i, i + d)
        i += d
    return out


def yaw_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_xyzw_from_yaw(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float64)
    q = np.zeros(yaw.shape + (4,), dtype=np.float64)
    q[..., 2] = np.sin(yaw / 2.0)
    q[..., 3] = np.cos(yaw / 2.0)
    return q


def tilt_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    """Angle (rad) between the body z-axis and world z."""
    x, y = q[..., 0], q[..., 1]
    cos_tilt = 1.0 - 2.0 * (x * x + y * y)
    return np.arccos(np.clip(cos_tilt, -1.0, 1.0))


def wrap_sym90(a: np.ndarray) -> np.ndarray:
    """Wrap an angle difference to [-45°, 45°) (cube 90° symmetry)."""
    q = math.pi / 2.0
    return (np.asarray(a) + q / 2.0) % q - q / 2.0


def build_obs(priv: dict[str, np.ndarray], prev_action: np.ndarray) -> np.ndarray:
    n = priv["ee_pos"].shape[0]
    yaw = yaw_from_quat_xyzw(priv["cube_quat"])
    parts = [
        priv["joint_pos"],
        priv["joint_vel"],
        priv["ee_pos"],
        priv["ee_quat"],
        priv["gripper_width"].reshape(n, 1),
        priv["cube_pos"],
        np.stack([np.sin(4 * yaw), np.cos(4 * yaw)], axis=-1),
        priv["cube_pos"] - priv["ee_pos"],
        prev_action,
    ]
    obs = np.concatenate([np.asarray(p, dtype=np.float32).reshape(n, -1) for p in parts], axis=-1)
    assert obs.shape[1] == OBS_DIM, obs.shape
    return obs


def latent_feature(mu: np.ndarray, mu_range: tuple[float, float] = (0.1, 1.0)) -> np.ndarray:
    """Normalized log friction in [0, 1], appended only for the privileged-observation probe."""
    lo, hi = math.log(mu_range[0]), math.log(mu_range[1])
    return ((np.log(np.asarray(mu, dtype=np.float64)) - lo) / (hi - lo)).astype(np.float32)


def append_latent(obs: np.ndarray, mu: np.ndarray, mu_range: tuple[float, float] = (0.1, 1.0)) -> np.ndarray:
    feat = latent_feature(mu, mu_range).reshape(-1, 1)
    return np.concatenate([obs, np.broadcast_to(feat, (obs.shape[0], 1))], axis=-1).astype(np.float32)


def split_obs(obs: np.ndarray) -> dict[str, np.ndarray]:
    return {k: obs[..., s] for k, s in obs_slices().items()}


def join_obs(parts: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(parts[k], dtype=np.float32) for k, _ in OBS_LAYOUT], axis=-1)
