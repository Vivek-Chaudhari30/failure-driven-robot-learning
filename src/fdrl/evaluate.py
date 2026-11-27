"""Validation / test evaluation -> per-cell EvalReport (ARCHITECTURE section 4).

Leakage guards (ARCHITECTURE section 5):
  * test evaluation requires ``frozen_analysis`` (the git SHA of the frozen analysis code) and an
    output directory with a ``test`` path component; the SHA is recorded in the report;
  * ``fdrl.intel`` only accepts reports with ``split == "val"``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from fdrl.conditions import CellGrid, LeakageError, eval_plan
from fdrl.config import Config
from fdrl.data.store import EpisodeStore
from fdrl.episodes import to_records
from fdrl.sim.base import SimBackend
from fdrl.sim.rollout import Controller, Labeler, rollout


@dataclass
class CellStats:
    n: int
    successes: int
    failure_counts: dict[str, int] = field(default_factory=dict)
    diag_means: dict[str, float] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else float("nan")


@dataclass
class EvalReport:
    split: str
    policy_id: str
    n_per_cell: int
    pipeline_seed: int
    per_cell: dict[int, CellStats]
    sim_steps: int
    frozen_analysis: str | None = None
    meta: dict = field(default_factory=dict)

    def success(self, cell: int) -> float:
        return self.per_cell[cell].rate

    def overall(self, cells=None) -> float:
        """Cell-balanced success over the given cells (default: all cells in the report)."""
        cs = list(self.per_cell) if cells is None else [c for c in cells if c in self.per_cell]
        return float(np.mean([self.per_cell[c].rate for c in cs])) if cs else float("nan")

    def failure_totals(self) -> dict[str, int]:
        tot: Counter = Counter()
        for st in self.per_cell.values():
            tot.update(st.failure_counts)
        return dict(tot)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_cell"] = {str(k): asdict(v) for k, v in self.per_cell.items()}
        return d

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> EvalReport:
        d = json.loads(Path(path).read_text())
        d["per_cell"] = {int(k): CellStats(**v) for k, v in d["per_cell"].items()}
        return cls(**d)


@dataclass
class EvalArtifacts:
    """In-memory states visited during validation, with expert labels (for the diagnosis)."""

    split: str
    obs: np.ndarray  # [M, 41]
    expert_actions: np.ndarray  # [M, 7]
    cell: np.ndarray  # [M]
    episode_success: np.ndarray  # [M] bool
    mu: np.ndarray  # [M]


def evaluate(
    controller: Controller,
    backend: SimBackend,
    grid: CellGrid,
    cfg: Config,
    *,
    split: str,
    pipeline_seed: int,
    policy_id: str,
    n_per_cell: int | None = None,
    store: EpisodeStore | None = None,
    out_dir: str | Path | None = None,
    frozen_analysis: str | None = None,
    expert: Labeler | None = None,
    collect_artifacts: bool = False,
    tag: str | None = None,
    strategy: str | None = None,
    round_idx: int | None = None,
) -> tuple[EvalReport, EvalArtifacts | None]:
    if split not in ("val", "test"):
        raise ValueError(split)
    if split == "test":
        if not frozen_analysis:
            raise LeakageError("test evaluation requires --frozen-analysis <git-sha>")
        if out_dir is None or "test" not in Path(out_dir).parts:
            raise LeakageError("test evaluation must write under a 'test/' directory")
        if collect_artifacts:
            raise LeakageError("test rollouts may not feed the diagnosis")
    if collect_artifacts and expert is None:
        raise ValueError("collect_artifacts needs the expert as labeler")
    n_per = n_per_cell or (cfg.eval.n_per_cell_val if split == "val" else cfg.eval.n_per_cell_test)
    plan = eval_plan(grid, split, pipeline_seed, n_per)
    items = [(p.condition, p.seed) for p in plan]
    results = rollout(backend, items, controller, cfg.task, labeler=expert if collect_artifacts else None,
                      record_obs=collect_artifacts)
    recs = to_records(results, split=split, source="policy", grid=grid, task=cfg.task, fail_cfg=cfg.failures,
                      accepted=[False] * len(results), policy_id=policy_id, strategy=strategy,
                      round_idx=round_idx, pipeline_seed=pipeline_seed, tag=tag)

    per_cell: dict[int, CellStats] = {}
    grouped: dict[int, list] = defaultdict(list)
    for s, _ in recs:
        grouped[s.cell].append(s)
    for cell, ss in grouped.items():
        fails = Counter(s.failure_label for s in ss if not s.success)
        keys = sorted({k for s in ss for k in s.diag})
        means = {}
        for k in keys:
            vals = np.array([s.diag.get(k, np.nan) for s in ss], dtype=np.float64)
            means[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        per_cell[cell] = CellStats(len(ss), sum(s.success for s in ss), dict(fails), means)
    report = EvalReport(split, policy_id, n_per, pipeline_seed, per_cell,
                        sim_steps=int(sum(r.length for r in results)), frozen_analysis=frozen_analysis,
                        meta={"tag": tag, "strategy": strategy, "round": round_idx})

    if store is not None:
        keep = _select_stored(recs, cfg.eval.max_traj_per_cell, cfg.eval.store_success_frac)
        trajs = {}
        for i in keep:
            s, tr = recs[i]
            tr.obs = None  # eval obs are reconstructable from privileged state; save space
            trajs[s.episode_id] = tr
        store.write([s for s, _ in recs], trajs)
    if out_dir is not None:
        report.save(Path(out_dir) / f"{split}_report.json")

    artifacts = None
    if collect_artifacts:
        artifacts = _artifacts(results, grid, cfg.diagnosis.max_states_per_cell, pipeline_seed)
    return report, artifacts


def _select_stored(recs, cap: int, success_frac: float) -> list[int]:
    by_cell: dict[int, list[int]] = defaultdict(list)
    for i, (s, _) in enumerate(recs):
        by_cell[s.cell].append(i)
    keep = []
    for idxs in by_cell.values():
        fails = [i for i in idxs if not recs[i][0].success]
        succ = [i for i in idxs if recs[i][0].success and (recs[i][0].seed % 1000) < success_frac * 1000]
        keep.extend((fails + succ)[:cap])
    return keep


def _artifacts(results, grid: CellGrid, cap: int, seed: int) -> EvalArtifacts:
    rng = np.random.default_rng(seed)
    obs, lab, cell, succ, mu = [], [], [], [], []
    for r in results:
        c = grid.cell_of(r.condition)
        obs.append(r.obs)
        lab.append(r.expert_actions)
        cell.append(np.full(r.length, c))
        succ.append(np.full(r.length, r.success))
        mu.append(np.full(r.length, r.condition.mu))
    O, A = np.concatenate(obs), np.concatenate(lab)
    C, S, M = np.concatenate(cell), np.concatenate(succ), np.concatenate(mu)
    keep = []
    for c in np.unique(C):
        idx = np.flatnonzero(C == c)
        if len(idx) > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False))
        keep.append(idx)
    k = np.concatenate(keep)
    return EvalArtifacts("val", O[k], A[k], C[k], S[k], M[k])
