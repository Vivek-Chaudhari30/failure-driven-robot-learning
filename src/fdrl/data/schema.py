"""Episode records (ARCHITECTURE section 4).

``EpisodeSummary`` is one Parquet row, written for EVERY episode (demo, rejected attempt, DAgger
rollout, validation and test rollouts). ``Trajectory`` holds the per-step arrays stored in HDF5.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fdrl.conditions import Condition

SPLITS = ("calib", "acq", "val", "test")
SOURCES = ("expert", "policy", "dagger")


def deterministic_key(split: str, seed: int, cond: Condition, policy_id: str | None) -> str:
    s = json.dumps([split, int(seed), cond.to_dict(), policy_id], sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:24]


@dataclass
class EpisodeSummary:
    split: str
    seed: int
    condition: Condition
    cell: int
    source: str
    success: bool
    length: int
    sim_steps_charged: int
    accepted: bool
    policy_id: str | None = None
    strategy: str | None = None
    round: int | None = None
    pipeline_seed: int | None = None
    tag: str | None = None  # e.g. experiment / dataset name
    failure_label: str | None = None
    failure_flags: list[str] = field(default_factory=list)
    failure_t: int | None = None
    diag: dict[str, float] = field(default_factory=dict)
    traj_ref: str | None = None
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"bad split {self.split!r}")
        if self.source not in SOURCES:
            raise ValueError(f"bad source {self.source!r}")

    @property
    def det_key(self) -> str:
        return deterministic_key(self.split, self.seed, self.condition, self.policy_id)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "episode_id": self.episode_id,
            "det_key": self.det_key,
            "split": self.split,
            "seed": int(self.seed),
            "x": float(self.condition.x),
            "y": float(self.condition.y),
            "yaw": float(self.condition.yaw),
            "mu": float(self.condition.mu),
            "cell": int(self.cell),
            "source": self.source,
            "policy_id": self.policy_id,
            "strategy": self.strategy,
            "round": -1 if self.round is None else int(self.round),
            "pipeline_seed": -1 if self.pipeline_seed is None else int(self.pipeline_seed),
            "tag": self.tag,
            "success": bool(self.success),
            "failure_label": self.failure_label,
            "failure_flags": "|".join(self.failure_flags),
            "failure_t": -1 if self.failure_t is None else int(self.failure_t),
            "length": int(self.length),
            "sim_steps_charged": int(self.sim_steps_charged),
            "accepted": bool(self.accepted),
            "traj_ref": self.traj_ref,
        }
        for k, v in self.diag.items():
            row[f"diag_{k}"] = float(v) if v is not None else math.nan
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> EpisodeSummary:
        diag = {k[5:]: float(v) for k, v in row.items() if k.startswith("diag_")}

        def opt_int(v: Any) -> int | None:
            return None if v is None or int(v) < 0 else int(v)

        return cls(
            episode_id=row["episode_id"],
            split=row["split"],
            seed=int(row["seed"]),
            condition=Condition(float(row["x"]), float(row["y"]), float(row["yaw"]), float(row["mu"])),
            cell=int(row["cell"]),
            source=row["source"],
            policy_id=row.get("policy_id"),
            strategy=row.get("strategy"),
            round=opt_int(row.get("round")),
            pipeline_seed=opt_int(row.get("pipeline_seed")),
            tag=row.get("tag"),
            success=bool(row["success"]),
            failure_label=row.get("failure_label"),
            failure_flags=[f for f in (row.get("failure_flags") or "").split("|") if f],
            failure_t=opt_int(row.get("failure_t")),
            length=int(row["length"]),
            sim_steps_charged=int(row["sim_steps_charged"]),
            accepted=bool(row["accepted"]),
            diag=diag,
            traj_ref=row.get("traj_ref"),
        )


@dataclass
class Trajectory:
    actions: np.ndarray  # [T, 7] executed actions
    privileged: dict[str, np.ndarray]  # each [T, ...]
    obs: np.ndarray | None = None  # [T, 41] flat (stored per key in HDF5)
    expert_actions: np.ndarray | None = None  # [T, 7] expert labels (DAgger / diagnosis)

    def training_actions(self) -> np.ndarray:
        """Labels for BC: expert labels if relabelled, else the executed (expert) actions."""
        return self.expert_actions if self.expert_actions is not None else self.actions
