"""Isaac Lab adapter for the Franka cube-lift task (ARCHITECTURE section 6, module 2).

STATUS: written against the Isaac Lab manager-based API but NOT YET RUN. It cannot run on macOS.
Every line marked ``VERIFY`` depends on an Isaac Lab detail that must be checked against the
version installed on the GPU machines (M1/M2 exit criterion: "a θ sweep visibly changes the scene;
logged θ matches applied θ"). Run ``scripts/isaac_smoke.py`` first.

This is the ONLY module in fdrl that imports Isaac Lab, and it does so lazily, after launching the
Kit app, as Isaac Lab requires.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from fdrl.conditions import Condition
from fdrl.config import IsaacCfg, TaskCfg
from fdrl.sim.obs import quat_xyzw_from_yaw


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:], q[..., :3]], axis=-1)


class IsaacLabLiftBackend:
    def __init__(self, task: TaskCfg, cfg: IsaacCfg, num_envs: int):
        self.task, self.cfg, self.num_envs = task, cfg, num_envs
        # 1) Launch the app BEFORE importing other isaaclab modules.
        from isaaclab.app import AppLauncher  # VERIFY: module path in the pinned version

        self._launcher = AppLauncher(headless=cfg.headless, device=cfg.device)
        self._app = self._launcher.app

        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401  registers tasks  # VERIFY: 3.0 contrib task package name
        import torch
        from isaaclab.sensors import ContactSensorCfg  # VERIFY
        from isaaclab_tasks.utils import parse_env_cfg  # VERIFY

        self.torch = torch
        env_cfg = parse_env_cfg(cfg.task, device=cfg.device, num_envs=num_envs)

        # 2) Derived config. Our rollout decides success/termination, so the env must never
        #    auto-reset mid-episode.
        env_cfg.episode_length_s = (task.horizon + 10) * task.policy_dt
        for name in ("object_dropping",):  # VERIFY: stock lift termination names
            if hasattr(env_cfg.terminations, name):
                setattr(env_cfg.terminations, name, None)
        # We place the cube ourselves after reset; disable the stock pose randomization.
        if hasattr(env_cfg.events, "reset_object_position"):  # VERIFY: stock event name
            env_cfg.events.reset_object_position = None
        # Contact sensing: fingers vs cube, cube vs table.
        env_cfg.scene.robot.spawn.activate_contact_sensors = True  # VERIFY
        env_cfg.scene.object.spawn.activate_contact_sensors = True  # VERIFY
        env_cfg.scene.contact_left = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        env_cfg.scene.contact_right = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        env_cfg.scene.contact_table = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],  # VERIFY: table prim path
        )
        # Action scaling: our normalized action * {pos,rot}_scale = metres/radians per step.
        self._ik_scale = float(getattr(env_cfg.actions.arm_action, "scale", 1.0))  # VERIFY

        self.env = gym.make(cfg.task, cfg=env_cfg)
        self.u = self.env.unwrapped
        self.device = self.u.device
        self._mu = np.ones(num_envs)
        self._z0 = np.full(num_envs, task.cube_half)
        self._wxyz = cfg.quat_order.lower() == "wxyz"

    # ------------------------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        out: dict[str, Any] = {"backend": "isaaclab", "task": self.cfg.task}
        try:
            import isaaclab

            out["isaaclab"] = getattr(isaaclab, "__version__", "unknown")
        except Exception:
            pass
        try:
            import isaacsim

            out["isaacsim"] = getattr(isaacsim, "__version__", "unknown")
        except Exception:
            pass
        return out

    def close(self) -> None:
        self.env.close()
        self._app.close()

    def reset(self, conditions: Sequence[Condition], seeds: Sequence[int]) -> dict[str, np.ndarray]:
        torch = self.torch
        n = self.num_envs
        if len(conditions) != n:
            raise ValueError(f"reset needs exactly {n} conditions")
        # GPU physics is not bitwise deterministic; the per-episode seed only seeds the env RNG.
        self.env.reset(seed=int(seeds[0]))
        scene = self.u.scene
        obj = scene["object"]
        env_ids = torch.arange(n, device=self.device)
        origins = scene.env_origins
        nx, ny = self.task.nominal_cube_xy
        xy = torch.tensor([[nx + c.x, ny + c.y] for c in conditions], device=self.device, dtype=torch.float32)
        z = torch.full((n, 1), self.task.cube_half, device=self.device)
        pos = origins + torch.cat([xy, z], dim=1)
        q = quat_xyzw_from_yaw(np.array([c.yaw for c in conditions]))
        if self._wxyz:
            q = _xyzw_to_wxyz(q)
        quat = torch.tensor(q, device=self.device, dtype=torch.float32)
        obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)  # VERIFY 3.0 name
        obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)  # VERIFY

        # Friction: set static = dynamic = mu on the cube AND the fingers so that PhysX's
        # material combine mode (average by default) yields exactly mu at the finger-cube contact.
        self._mu = np.array([c.mu for c in conditions], dtype=np.float64)
        mu_t = torch.tensor(self._mu, dtype=torch.float32)
        for entity in (obj, scene["robot"]):
            view = entity.root_physx_view  # VERIFY: PhysX tensor view in the pinned version
            mats = view.get_material_properties()  # [N, num_shapes, 3] (static, dynamic, restitution), CPU
            mats[:, :, 0] = mu_t[:, None]
            mats[:, :, 1] = mu_t[:, None]
            view.set_material_properties(mats, torch.arange(n))
        # One zero-motion step so the new state is reflected in the buffers.
        self._last_ee = None
        priv = self.step(np.concatenate([np.zeros((n, 6)), np.ones((n, 1))], axis=1).astype(np.float32))
        self._z0 = priv["cube_pos"][:, 2].copy()
        priv["cube_z0"] = self._z0.copy()
        return priv

    def step(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        torch = self.torch
        a = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        raw = np.zeros_like(a)
        raw[:, :3] = a[:, :3] * self.task.pos_scale / self._ik_scale
        raw[:, 3:6] = a[:, 3:6] * self.task.rot_scale / self._ik_scale
        raw[:, 6] = np.where(a[:, 6] < 0, -1.0, 1.0)  # binary gripper: negative closes  # VERIFY
        self.env.step(torch.tensor(raw, device=self.device))
        return self._priv()

    # ------------------------------------------------------------------------------------
    def _priv(self) -> dict[str, np.ndarray]:
        scene = self.u.scene
        robot, obj = scene["robot"], scene["object"]
        origins = scene.env_origins

        def np_(t):
            return t.detach().cpu().numpy().astype(np.float64)

        ee_frame = scene["ee_frame"]  # VERIFY: FrameTransformer name in the lift cfg
        ee_pos = np_(ee_frame.data.target_pos_w[:, 0, :] - origins)
        ee_quat = np_(ee_frame.data.target_quat_w[:, 0, :])
        cube_pos = np_(obj.data.root_pos_w - origins)
        cube_quat = np_(obj.data.root_quat_w)
        if self._wxyz:
            ee_quat, cube_quat = _wxyz_to_xyzw(ee_quat), _wxyz_to_xyzw(cube_quat)
        jp, jv = np_(robot.data.joint_pos), np_(robot.data.joint_vel)
        lim = np_(robot.data.soft_joint_pos_limits)  # [N, J, 2]  # VERIFY
        at_lim = ((jp <= lim[..., 0] + 1e-3) | (jp >= lim[..., 1] - 1e-3))[:, :7].any(axis=1)

        def force(name):
            s = scene[name]
            fm = s.data.force_matrix_w  # [N, bodies, filters, 3]  # VERIFY
            return np.linalg.norm(np_(fm), axis=-1).reshape(fm.shape[0], -1).max(axis=1)

        fl, fr, ft = force("contact_left"), force("contact_right"), force("contact_table")
        finite = np.isfinite(cube_pos).all(axis=1) & np.isfinite(jp).all(axis=1)
        return {
            "joint_pos": jp[:, :9],
            "joint_vel": jv[:, :9],
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "gripper_width": jp[:, 7] + jp[:, 8],
            "cube_pos": cube_pos,
            "cube_quat": cube_quat,
            "cube_lin_vel": np_(obj.data.root_lin_vel_w),
            "contact_lr": np.stack([fl, fr], axis=1),
            "contact_table": ft,
            "at_joint_limit": at_lim.astype(np.float64),
            "invalid": (~finite).astype(np.float64),
            "off_table": (cube_pos[:, 2] < self._z0 - 0.05).astype(np.float64),
            "mu": self._mu.copy(),
            "cube_z0": self._z0.copy(),
        } | self._ee_vel(ee_pos)

    _last_ee: np.ndarray | None = None

    def _ee_vel(self, ee_pos: np.ndarray) -> dict[str, np.ndarray]:
        v = np.zeros_like(ee_pos) if self._last_ee is None else (ee_pos - self._last_ee) / self.task.policy_dt
        self._last_ee = ee_pos.copy()
        return {"ee_lin_vel": v}
