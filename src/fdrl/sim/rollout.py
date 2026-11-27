"""Batched rollouts with a per-step recorder.

A *controller* maps (obs [N, 41], privileged state) -> actions [N, 7]. The scripted expert ignores
obs; a BC policy ignores privileged state (except the opt-in latent probe). An optional *labeler*
(the expert) records what the expert would have done at every visited state, which is used for
DAgger relabelling and for the diagnosis.

Success and termination are decided here from privileged state (EXPERIMENT_DESIGN section 2):
success = cube height >= z0 + success_height with both fingers in contact for
success_hold_steps consecutive steps; terminal failure = cube off table or invalid physics;
otherwise the episode runs to the horizon.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from fdrl.conditions import Condition
from fdrl.config import TaskCfg
from fdrl.sim.base import SimBackend
from fdrl.sim.obs import ACTION_DIM, build_obs

Controller = Callable[[np.ndarray, dict[str, np.ndarray]], np.ndarray]
Labeler = Callable[[dict[str, np.ndarray]], np.ndarray]


@dataclass
class EpisodeResult:
    condition: Condition
    seed: int
    length: int
    success: bool
    terminal: str | None  # None, "off_table", "invalid"
    obs: np.ndarray | None  # [L, 41]
    actions: np.ndarray  # [L, 7] executed
    expert_actions: np.ndarray | None  # [L, 7] labels
    privileged: dict[str, np.ndarray]  # each [L, ...], state BEFORE the action at that step


def rollout(
    backend: SimBackend,
    items: Sequence[tuple[Condition, int]],
    controller: Controller,
    task: TaskCfg,
    labeler: Labeler | None = None,
    record_obs: bool = True,
) -> list[EpisodeResult]:
    """Run every (condition, seed) to termination, num_envs at a time. Results keep input order."""
    results: list[EpisodeResult] = []
    n_env = backend.num_envs
    for start in range(0, len(items), n_env):
        chunk = list(items[start : start + n_env])
        k = len(chunk)
        pad = chunk + [chunk[-1]] * (n_env - k)
        results.extend(_rollout_chunk(backend, pad, controller, task, labeler, record_obs)[:k])
    return results


def _rollout_chunk(backend, items, controller, task, labeler, record_obs):
    n = backend.num_envs
    T = task.horizon
    conds = [c for c, _ in items]
    seeds = [s for _, s in items]
    priv = backend.reset(conds, seeds)
    prev_action = np.zeros((n, ACTION_DIM), dtype=np.float32)
    done = np.zeros(n, dtype=bool)
    length = np.full(n, T, dtype=np.int64)
    success = np.zeros(n, dtype=bool)
    terminal: list[str | None] = [None] * n
    hold = np.zeros(n, dtype=np.int64)
    rec_obs, rec_act, rec_lab = [], [], []
    rec_priv: dict[str, list[np.ndarray]] = {k: [] for k in priv}

    for t in range(T):
        obs = build_obs(priv, prev_action)
        act = np.asarray(controller(obs, priv), dtype=np.float32)
        if act.shape != (n, ACTION_DIM):
            raise ValueError(f"controller returned {act.shape}, expected {(n, ACTION_DIM)}")
        if record_obs:
            rec_obs.append(obs)
        rec_act.append(act)
        if labeler is not None:
            rec_lab.append(np.asarray(labeler(priv), dtype=np.float32))
        for k, v in priv.items():
            rec_priv[k].append(np.asarray(v, dtype=np.float32))

        priv = backend.step(act)
        prev_action = act

        # success / termination bookkeeping on the post-step state
        lifted = priv["cube_pos"][:, 2] >= priv["cube_z0"] + task.success_height
        held = priv["contact_lr"].min(axis=1) > 0.1
        hold = np.where(lifted & held, hold + 1, 0)
        newly_ok = (~done) & (hold >= task.success_hold_steps)
        bad_off = (~done) & (priv["off_table"] > 0.5)
        bad_inv = (~done) & ((priv["invalid"] > 0.5) | ~np.isfinite(priv["cube_pos"]).all(axis=1))
        for i in np.flatnonzero(newly_ok):
            success[i], done[i], length[i] = True, True, t + 1
        for i in np.flatnonzero(bad_off & ~newly_ok):
            terminal[i], done[i], length[i] = "off_table", True, t + 1
        for i in np.flatnonzero(bad_inv & ~newly_ok & ~bad_off):
            terminal[i], done[i], length[i] = "invalid", True, t + 1
        if done.all():
            break

    O = np.stack(rec_obs, 1) if record_obs else None
    A = np.stack(rec_act, 1)
    L = np.stack(rec_lab, 1) if labeler is not None else None
    P = {k: np.stack(v, 1) for k, v in rec_priv.items()}
    out = []
    for i in range(n):
        li = int(length[i])
        out.append(
            EpisodeResult(
                condition=conds[i],
                seed=int(seeds[i]),
                length=li,
                success=bool(success[i]),
                terminal=terminal[i],
                obs=None if O is None else O[i, :li].copy(),
                actions=A[i, :li].copy(),
                expert_actions=None if L is None else L[i, :li].copy(),
                privileged={k: v[i, :li].copy() for k, v in P.items()},
            )
        )
    return out
