"""Dataset manifests (ARCHITECTURE AD4): an immutable list of episode IDs plus a content hash.

Datasets are never copied. A training set is ``DatasetManifest`` + ``EpisodeStore``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from fdrl.conditions import CellGrid, Condition
from fdrl.data.store import EpisodeStore
from fdrl.sim.obs import append_latent


def _sha(ids: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    episode_ids: tuple[str, ...]
    parent: str | None
    sha256: str
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @classmethod
    def create(cls, name: str, ids, parent: str | None = None, meta: dict | None = None) -> DatasetManifest:
        ids = tuple(ids)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate episode ids in manifest")
        return cls(name, ids, parent, _sha(ids), dict(meta or {}))

    def extend(self, name: str, new_ids, meta: dict | None = None) -> DatasetManifest:
        return DatasetManifest.create(name, self.episode_ids + tuple(new_ids), parent=self.sha256, meta=meta)

    def prefix(self, name: str, n: int) -> DatasetManifest:
        if n > len(self.episode_ids):
            raise ValueError(f"manifest {self.name} has only {len(self.episode_ids)} episodes, asked {n}")
        return DatasetManifest.create(name, self.episode_ids[:n], parent=self.sha256)

    def __len__(self) -> int:
        return len(self.episode_ids)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["episode_ids"] = list(self.episode_ids)
        Path(path).write_text(json.dumps(d, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> DatasetManifest:
        d = json.loads(Path(path).read_text())
        m = cls(d["name"], tuple(d["episode_ids"]), d.get("parent"), d["sha256"], d.get("meta", {}))
        if _sha(m.episode_ids) != m.sha256:
            raise ValueError(f"manifest {path} hash mismatch (edited by hand?)")
        return m


@dataclass
class TrainingArrays:
    obs: np.ndarray  # [M, d]
    actions: np.ndarray  # [M, 7]
    episode: np.ndarray  # [M] episode index into conditions
    cell: np.ndarray  # [M]
    conditions: list[Condition]


def load_training_arrays(
    store: EpisodeStore,
    manifest: DatasetManifest,
    grid: CellGrid,
    include_latent: bool = False,
) -> TrainingArrays:
    """Gather (obs, action-label) pairs for every episode in the manifest."""
    sums = store.summaries(manifest.episode_ids)
    for s in sums:
        if s.split != "acq" or not s.accepted:
            raise ValueError(f"episode {s.episode_id} is not an accepted acquisition episode")
        if s.traj_ref is None:
            raise ValueError(f"episode {s.episode_id} has no stored trajectory")
    trajs = store.load_trajectories([s.traj_ref for s in sums])  # type: ignore[misc]
    obs, act, ep, cell, conds = [], [], [], [], []
    for i, s in enumerate(sums):
        tr = trajs[s.traj_ref]  # type: ignore[index]
        if tr.obs is None:
            raise ValueError(f"episode {s.episode_id} has no observations")
        o = tr.obs
        if include_latent:
            o = append_latent(o, np.full(len(o), s.condition.mu), grid.cfg.mu_range)
        obs.append(o)
        act.append(tr.training_actions())
        ep.append(np.full(len(o), i))
        cell.append(np.full(len(o), s.cell))
        conds.append(s.condition)
    if not obs:
        raise ValueError("empty dataset")
    return TrainingArrays(
        obs=np.concatenate(obs).astype(np.float32),
        actions=np.concatenate(act).astype(np.float32),
        episode=np.concatenate(ep),
        cell=np.concatenate(cell),
        conditions=conds,
    )
