"""Turn rollout results into stored records: summary (always) + trajectory (when kept)."""

from __future__ import annotations

from collections.abc import Sequence

from fdrl.conditions import CellGrid
from fdrl.config import FailureCfg, TaskCfg
from fdrl.data.schema import EpisodeSummary, Trajectory
from fdrl.failures import label_episode
from fdrl.sim.rollout import EpisodeResult


def to_records(
    results: Sequence[EpisodeResult],
    *,
    split: str,
    source: str,
    grid: CellGrid,
    task: TaskCfg,
    fail_cfg: FailureCfg,
    accepted: Sequence[bool] | None = None,
    charged: Sequence[int] | None = None,
    policy_id: str | None = None,
    strategy: str | None = None,
    round_idx: int | None = None,
    pipeline_seed: int | None = None,
    tag: str | None = None,
) -> list[tuple[EpisodeSummary, Trajectory]]:
    out = []
    for i, r in enumerate(results):
        fl = label_episode(r.privileged, r.actions, r.success, task, fail_cfg)
        s = EpisodeSummary(
            split=split,
            seed=r.seed,
            condition=r.condition,
            cell=grid.cell_of(r.condition),
            source=source,
            success=r.success,
            length=r.length,
            sim_steps_charged=int(charged[i]) if charged is not None else r.length,
            accepted=bool(accepted[i]) if accepted is not None else r.success,
            policy_id=policy_id,
            strategy=strategy,
            round=round_idx,
            pipeline_seed=pipeline_seed,
            tag=tag,
            failure_label=fl.label,
            failure_flags=fl.flags,
            failure_t=fl.t,
            diag=fl.diag,
        )
        tr = Trajectory(actions=r.actions, privileged=r.privileged, obs=r.obs, expert_actions=r.expert_actions)
        out.append((s, tr))
    return out
