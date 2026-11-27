"""Simulator boundary (ARCHITECTURE AD1).

Only modules inside ``fdrl.sim`` may import a simulator. Everything else talks to a ``SimBackend``
and receives plain numpy arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from fdrl.conditions import Condition
from fdrl.config import Config


@runtime_checkable
class SimBackend(Protocol):
    """Batched, fixed-size simulator.

    ``reset`` applies one condition θ and one seed per environment and returns the privileged
    state (see ``fdrl.sim.obs.PRIV_KEYS``). ``step`` takes normalized actions in [-1, 1]^7
    (gripper: +1 open, -1 close) and returns the next privileged state. The backend does not
    decide success or failure; ``fdrl.sim.rollout`` and ``fdrl.failures`` do, from privileged state.
    """

    num_envs: int

    def reset(self, conditions: Sequence[Condition], seeds: Sequence[int]) -> dict[str, np.ndarray]: ...

    def step(self, actions: np.ndarray) -> dict[str, np.ndarray]: ...

    def info(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def make_backend(cfg: Config, num_envs: int | None = None) -> SimBackend:
    n = num_envs or cfg.sim.num_envs
    if cfg.sim.backend == "toy":
        from fdrl.sim.toy_lift import ToyLiftBackend

        return ToyLiftBackend(cfg.task, cfg.sim.toy, n)
    if cfg.sim.backend == "isaaclab":
        from fdrl.sim.isaaclab_lift import IsaacLabLiftBackend  # imports Isaac Lab lazily

        return IsaacLabLiftBackend(cfg.task, cfg.sim.isaac, n)
    raise ValueError(f"unknown sim backend {cfg.sim.backend!r}")
