"""Per-cell statistics shared by the diagnosis and the strategies (EXPERIMENT_DESIGN 7.1-7.2).

Leakage guard (ARCHITECTURE section 5, item 1): every function that takes an evaluation report
calls ``require_val`` first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.spatial import cKDTree

from fdrl.conditions import CellGrid, LeakageError
from fdrl.evaluate import EvalArtifacts, EvalReport
from fdrl.policy.bc import Ensemble


def require_val(report: EvalReport | EvalArtifacts) -> None:
    if report.split != "val":
        raise LeakageError(f"fdrl.intel only accepts validation data, got split={report.split!r}")


@dataclass
class BetaPosterior:
    mean: float
    std: float
    upper: float  # 90% quantile


def beta_posterior(successes: int, n: int, q: float = 0.9) -> BetaPosterior:
    a, b = 1 + successes, 1 + (n - successes)
    d = sps.beta(a, b)
    return BetaPosterior(float(d.mean()), float(d.std()), float(d.ppf(q)))


def success_posteriors(report: EvalReport, cells) -> dict[int, BetaPosterior]:
    require_val(report)
    out = {}
    for c in cells:
        st = report.per_cell.get(c)
        out[c] = beta_posterior(st.successes, st.n) if st else beta_posterior(0, 0)
    return out


def demo_density(index: pd.DataFrame, episode_ids, grid: CellGrid) -> dict[int, int]:
    """Number of training episodes per cell in a dataset."""
    if index.empty:
        return {c: 0 for c in range(grid.n_cells)}
    sub = index[index["episode_id"].isin(set(episode_ids))]
    counts = sub.groupby("cell").size().to_dict()
    return {c: int(counts.get(c, 0)) for c in range(grid.n_cells)}


@dataclass
class ExpertCellStats:
    attempts: int
    accepted: int
    success: float  # partially pooled expert success e_c
    mean_attempt_steps: float
    kappa: float  # expected sim steps per ACCEPTED demo


def expert_stats(index: pd.DataFrame, grid: CellGrid, pipeline_seed: int | None = None,
                 prior_strength: float = 5.0, default_steps: float = 100.0) -> dict[int, ExpertCellStats]:
    """Expert success and cost per cell from acquisition logs (rejected attempts included)."""
    if index.empty:
        rows = pd.DataFrame(columns=["cell", "accepted", "sim_steps_charged"])
    else:
        rows = index[(index["split"].isin(["acq", "calib"])) & (index["source"] == "expert")]
        if pipeline_seed is not None and "pipeline_seed" in rows:
            rows = rows[(rows["pipeline_seed"] == pipeline_seed) | (rows["split"] == "calib")]
    glob_rate = float(rows["accepted"].mean()) if len(rows) else 0.9
    glob_steps = float(rows["sim_steps_charged"].mean()) if len(rows) else default_steps
    a0, b0 = prior_strength * glob_rate, prior_strength * (1 - glob_rate)
    out = {}
    g = rows.groupby("cell") if len(rows) else None
    for c in range(grid.n_cells):
        if g is not None and c in g.groups:
            r = g.get_group(c)
            att, acc = len(r), int(r["accepted"].sum())
            steps = float(r["sim_steps_charged"].mean())
        else:
            att, acc, steps = 0, 0, glob_steps
        e = (acc + a0) / (att + a0 + b0)
        out[c] = ExpertCellStats(att, acc, float(e), steps, float(steps / max(e, 1e-3)))
    return out


@dataclass
class EnsembleSignal:
    disagreement: float  # epistemic: mean variance across members
    error: float  # squared error of the ensemble mean to the expert label
    aleatoric_share: float  # error / (error + disagreement)


def ensemble_decomposition(ensemble: Ensemble, art: EvalArtifacts) -> dict[int, EnsembleSignal]:
    require_val(art)
    preds = ensemble.predict_pose(art.obs)  # [K, M, 6]
    var = preds.var(axis=0).mean(axis=1)
    err = ((preds.mean(axis=0) - art.expert_actions[:, :6]) ** 2).mean(axis=1)
    out = {}
    for c in np.unique(art.cell):
        m = art.cell == c
        d, e = float(var[m].mean()), float(err[m].mean())
        out[int(c)] = EnsembleSignal(d, e, e / (e + d + 1e-12))
    return out


def state_coverage_gap(art: EvalArtifacts, train_obs: np.ndarray, obs_mean: np.ndarray, obs_std: np.ndarray,
                       max_train: int = 50_000, seed: int = 0) -> dict[int, float]:
    """Median normalized kNN distance from FAILING-rollout states to the nearest training state,
    per cell, divided by the same quantity for successful-rollout states (global)."""
    require_val(art)
    rng = np.random.default_rng(seed)
    tr = train_obs
    if len(tr) > max_train:
        tr = tr[rng.choice(len(tr), max_train, replace=False)]
    std = np.where(obs_std < 1e-6, 1.0, obs_std)
    tree = cKDTree((tr - obs_mean) / std)
    dist, _ = tree.query((art.obs - obs_mean) / std, k=1)
    ref = np.median(dist[art.episode_success]) if art.episode_success.any() else np.median(dist)
    out = {}
    for c in np.unique(art.cell):
        m = (art.cell == c) & ~art.episode_success
        out[int(c)] = float(np.median(dist[m]) / max(ref, 1e-9)) if m.any() else 1.0
    return out
