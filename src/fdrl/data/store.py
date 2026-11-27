"""Episode store: sharded HDF5 trajectories + Parquet index (ARCHITECTURE AD4/AD5).

Layout under ``root``:
    shards/<shard_id>.hdf5      robomimic-style: data/<episode_id>/{obs/<key>, actions, expert_actions, privileged/<key>}
    index/part-<shard_id>.parquet   one row per episode (EpisodeSummary.to_row)

Each ``write`` call creates one new shard and one new index part, so concurrent writers (different
processes) never touch the same file.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from fdrl.data.schema import EpisodeSummary, Trajectory
from fdrl.sim.obs import join_obs, split_obs


class EpisodeStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "shards").mkdir(parents=True, exist_ok=True)
        (self.root / "index").mkdir(parents=True, exist_ok=True)
        self._index_cache: pd.DataFrame | None = None
        self._index_parts: tuple[str, ...] = ()

    # ------------------------------------------------------------------ writing
    def write(
        self,
        summaries: Sequence[EpisodeSummary],
        trajectories: dict[str, Trajectory] | None = None,
    ) -> str:
        """Write summaries (all) and trajectories (subset, keyed by episode_id). Returns shard id."""
        trajectories = trajectories or {}
        shard_id = uuid.uuid4().hex[:16]
        if trajectories:
            path = self.root / "shards" / f"{shard_id}.hdf5"
            with h5py.File(path, "w") as f:
                data = f.create_group("data")
                for s in summaries:
                    tr = trajectories.get(s.episode_id)
                    if tr is None:
                        continue
                    g = data.create_group(s.episode_id)
                    g.attrs["num_samples"] = int(tr.actions.shape[0])
                    g.attrs["condition"] = json.dumps(s.condition.to_dict())
                    g.attrs["seed"] = int(s.seed)
                    g.create_dataset("actions", data=tr.actions, compression="gzip")
                    if tr.expert_actions is not None:
                        g.create_dataset("expert_actions", data=tr.expert_actions, compression="gzip")
                    if tr.obs is not None:
                        og = g.create_group("obs")
                        for k, v in split_obs(tr.obs).items():
                            og.create_dataset(k, data=v, compression="gzip")
                    pg = g.create_group("privileged")
                    for k, v in tr.privileged.items():
                        pg.create_dataset(k, data=v, compression="gzip")
                    s.traj_ref = f"{shard_id}:{s.episode_id}"
        rows = [s.to_row() for s in summaries]
        if rows:
            pd.DataFrame(rows).to_parquet(self.root / "index" / f"part-{shard_id}.parquet", index=False)
        self._index_cache = None
        return shard_id

    # ------------------------------------------------------------------ reading
    def index(self) -> pd.DataFrame:
        parts = tuple(sorted(str(p) for p in (self.root / "index").glob("part-*.parquet")))
        if self._index_cache is not None and parts == self._index_parts:
            return self._index_cache
        if not parts:
            df = pd.DataFrame()
        else:
            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        self._index_cache, self._index_parts = df, parts
        return df

    def summaries(self, episode_ids: Iterable[str]) -> list[EpisodeSummary]:
        df = self.index().set_index("episode_id", drop=False)
        return [EpisodeSummary.from_row(df.loc[e].to_dict()) for e in episode_ids]

    def load_trajectory(self, traj_ref: str) -> Trajectory:
        return self.load_trajectories([traj_ref])[traj_ref]

    def load_trajectories(self, traj_refs: Sequence[str]) -> dict[str, Trajectory]:
        by_shard: dict[str, list[str]] = defaultdict(list)
        for ref in traj_refs:
            shard, eid = ref.split(":", 1)
            by_shard[shard].append(eid)
        out: dict[str, Trajectory] = {}
        for shard, eids in by_shard.items():
            with h5py.File(self.root / "shards" / f"{shard}.hdf5", "r") as f:
                for eid in eids:
                    g = f["data"][eid]
                    obs = None
                    if "obs" in g:
                        obs = join_obs({k: np.asarray(g["obs"][k]) for k in g["obs"]})
                    out[f"{shard}:{eid}"] = Trajectory(
                        actions=np.asarray(g["actions"]),
                        expert_actions=np.asarray(g["expert_actions"]) if "expert_actions" in g else None,
                        obs=obs,
                        privileged={k: np.asarray(g["privileged"][k]) for k in g["privileged"]},
                    )
        return out
