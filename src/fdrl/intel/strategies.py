"""Acquisition strategies S0-S7 (+S3b, S4b, S2+, shuffled-S5). EXPERIMENT_DESIGN section 7.

A strategy turns an ``AcquisitionContext`` into an ``AcquisitionPlan``: what to collect and how.
The loop (``fdrl.loop``) executes the plan under a strict cost budget (expert sim steps,
rejected attempts included; policy rollouts charged for S3b and S6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fdrl.conditions import (
    CellGrid,
    CellMixture,
    Condition,
    ConditionDistribution,
    LeakageError,
    NominalDistribution,
    uniform_train,
)
from fdrl.config import ConditionSpaceCfg, StrategyCfg
from fdrl.evaluate import EvalReport
from fdrl.intel.diagnose import Diagnosis
from fdrl.intel.stats import ExpertCellStats, beta_posterior, require_val

MODES = ("none", "sampler", "sequence", "failed_replay", "dagger")
STRATEGIES = ("S0", "S1", "S2", "S2+", "S3", "S3b", "S4", "S4b", "S5", "S5-shuffled", "S6", "S7")
NEEDS_VAL = {"S3", "S5", "S5-shuffled"}
NEEDS_DIAGNOSIS = {"S5", "S5-shuffled"}
NEEDS_POLICY = {"S3b", "S6"}


@dataclass
class AcquisitionPlan:
    strategy: str
    mode: str
    budget: float  # cost units (expert/policy sim steps)
    distribution: ConditionDistribution | None = None
    sequence: list[Condition] | None = None
    cell_weights: dict[int, float] | None = None
    notes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(self.mode)


@dataclass
class AcquisitionContext:
    grid: CellGrid
    cond_cfg: ConditionSpaceCfg
    cfg: StrategyCfg
    regime: str  # "R1" | "R2"
    budget: float  # cost units
    pipeline_seed: int
    round_idx: int
    index: pd.DataFrame  # episode index (for densities, expert stats)
    dataset_ids: tuple[str, ...]
    val_report: EvalReport | None = None
    diagnosis: Diagnosis | None = None
    expert: dict[int, ExpertCellStats] | None = None
    oracle_report: EvalReport | None = None  # S7 only (test split, leakage by design)
    allow_oracle: bool = False
    gain_history: list[dict] | None = None  # realized gain per demo by class (S5 rounds >= 2)


def make_plan(name: str, ctx: AcquisitionContext) -> AcquisitionPlan:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; choose from {STRATEGIES}")
    if name in NEEDS_VAL:
        if ctx.val_report is None:
            raise ValueError(f"{name} needs a validation report")
        require_val(ctx.val_report)
    fn = {
        "S0": _s0, "S1": _s1, "S2": _s2, "S2+": _s2plus, "S3": _s3, "S3b": _s3b, "S4": _s4,
        "S4b": _s4b, "S5": _s5, "S5-shuffled": _s5_shuffled, "S6": _s6, "S7": _s7,
    }[name]
    return fn(ctx)


# ---------------------------------------------------------------------- simple baselines

def _s0(ctx):
    return AcquisitionPlan("S0", "none", 0.0)


def _s1(ctx):
    dist = NominalDistribution(ctx.cond_cfg) if ctx.regime == "R1" else uniform_train(ctx.grid)
    return AcquisitionPlan("S1", "sampler", ctx.budget, distribution=dist)


def _s2(ctx, name="S2", extra=0.0):
    dist = uniform_train(ctx.grid)
    return AcquisitionPlan(name, "sampler", ctx.budget + extra, distribution=dist, cell_weights=dist.as_dict())


def _s2plus(ctx):
    return _s2(ctx, "S2+", ctx.cfg.extra_budget)


def _with_floor(weights: dict[int, float], cells, floor: float) -> dict[int, float]:
    w = np.array([max(weights.get(c, 0.0), 0.0) for c in cells], dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.full(len(cells), 1.0 / len(cells))
    w = (1 - floor) * w + floor / len(cells)
    return {c: float(v) for c, v in zip(cells, w, strict=True)}


def _s3(ctx):
    rep = ctx.val_report
    cells = ctx.grid.train_cells
    raw = {}
    for c in cells:
        st = rep.per_cell.get(c)
        n, s = (st.n, st.successes) if st else (0, 0)
        raw[c] = (n - s + 1) / (n + 2)
    w = _with_floor(raw, cells, ctx.cfg.floor)
    return AcquisitionPlan("S3", "sampler", ctx.budget, distribution=CellMixture(w), cell_weights=w)


def _s3b(ctx):
    dist = uniform_train(ctx.grid)
    return AcquisitionPlan("S3b", "failed_replay", ctx.budget, distribution=dist,
                           notes={"max_policy_rollouts": ctx.cfg.s3b_max_policy_rollouts})


def _s6(ctx):
    dist = uniform_train(ctx.grid)
    return AcquisitionPlan("S6", "dagger", ctx.budget, distribution=dist)


def _s4(ctx):
    """Greedy k-center in normalized Θ against existing demo conditions."""
    grid = ctx.grid
    idx = ctx.index
    existing = np.zeros((0, 4))
    if not idx.empty and ctx.dataset_ids:
        sub = idx[idx["episode_id"].isin(set(ctx.dataset_ids))]
        existing = sub[["x", "y", "yaw", "mu"]].to_numpy(dtype=np.float64)
    pool = []
    for i in range(ctx.cfg.kcenter_pool):
        rng = np.random.default_rng([ctx.pipeline_seed, ctx.round_idx, i, 4])
        cell = int(rng.choice(grid.train_cells))
        pool.append(grid.sample_in_cell(cell, rng))
    P = grid.normalize(pool)
    if len(existing):
        E = grid.normalize(existing)
        d = np.min(((P[:, None, :] - E[None, :, :]) ** 2).sum(-1), axis=1)
    else:
        d = np.full(len(P), np.inf)
    order: list[int] = []
    taken = np.zeros(len(P), dtype=bool)
    max_pick = min(len(P), 2000)
    for _ in range(max_pick):
        j = int(np.argmax(np.where(taken, -1.0, d)))
        order.append(j)
        taken[j] = True
        d = np.minimum(d, ((P - P[j]) ** 2).sum(-1))
    seq = [pool[j] for j in order]
    return AcquisitionPlan("S4", "sequence", ctx.budget, sequence=seq)


def _s4b(ctx):
    """Factored coverage: vary one factor at a time around a base condition (non-adaptive)."""
    grid, cfg = ctx.grid, ctx.cond_cfg
    rng = np.random.default_rng([ctx.pipeline_seed, ctx.round_idx, 44])
    base = Condition(0.0, 0.0, 0.0, float(np.sqrt(cfg.mu_range[0] * cfg.mu_range[1])))
    seq: list[Condition] = []
    for i in range(4000):
        f = i % 4
        x, y, yaw, mu = base.x, base.y, base.yaw, base.mu
        if f == 0:
            x = rng.uniform(*cfg.x_range)
        elif f == 1:
            y = rng.uniform(*cfg.y_range)
        elif f == 2:
            yaw = np.radians(rng.uniform(*cfg.yaw_range_deg))
        else:
            mu = float(np.exp(rng.uniform(np.log(cfg.mu_range[0]), np.log(cfg.mu_range[1]))))
        c = Condition(float(x), float(y), float(yaw), float(mu))
        if grid.cell_of(c) not in grid.holdout:
            seq.append(c)
    return AcquisitionPlan("S4b", "sequence", ctx.budget, sequence=seq)


# ---------------------------------------------------------------------- S5 (proposed)

def s5_priorities(ctx: AcquisitionContext, report: EvalReport) -> tuple[dict[int, float], dict[int, dict]]:
    """p_c = w_c (1 - s_c) f_c / kappa_c + beta * sigma_c   (EXPERIMENT_DESIGN 7.1)."""
    cfg = ctx.cfg
    cells = ctx.grid.train_cells
    fix = _fixability(ctx)
    kappas = {c: (ctx.expert[c].kappa if ctx.expert and c in ctx.expert else 1.0) for c in cells}
    med_kappa = float(np.median(list(kappas.values()))) or 1.0
    pri, detail = {}, {}
    for c in cells:
        st = report.per_cell.get(c)
        post = beta_posterior(st.successes, st.n) if st else beta_posterior(0, 0)
        f = fix.get(c, 1.0) if cfg.use_diagnosis else 1.0
        infeasible = (cfg.use_feasibility and ctx.diagnosis is not None and c in ctx.diagnosis.cells
                      and ctx.diagnosis.cells[c].label == "expert_infeasible")
        k = kappas[c] / med_kappa if cfg.use_cost else 1.0
        w = 1.0 / len(cells)
        # Feasibility filter: no priority at all (not even exploration) where the expert can't succeed.
        p = 0.0 if infeasible else w * (1 - post.mean) * f / max(k, 1e-6) + cfg.beta * w * post.std
        pri[c] = float(p)
        detail[c] = {"s_mean": post.mean, "s_std": post.std, "fixability": f, "kappa_rel": k, "priority": p}
    return pri, detail


def _fixability(ctx: AcquisitionContext) -> dict[int, float]:
    if ctx.diagnosis is None:
        return {}
    priors = ctx.cfg.fixability_priors
    if not ctx.gain_history:
        return {c: d.fixability for c, d in ctx.diagnosis.cells.items()}
    # Rounds >= 2: partially pooled realized gain-per-demo by diagnosis class.
    by_label: dict[str, list[float]] = {}
    for h in ctx.gain_history:
        by_label.setdefault(h["label"], []).append(h["gain_per_demo"])
    all_g = [g for v in by_label.values() for g in v]
    gmax = max(max(all_g), 1e-9) if all_g else 1.0
    out = {}
    for c, d in ctx.diagnosis.cells.items():
        obs = by_label.get(d.label, [])
        k = len(obs)
        emp = float(np.clip(np.mean(obs) / gmax, 0, 1)) if k else priors.get(d.label, 0.5)
        lam = k / (k + 3.0)  # shrink toward the class prior
        out[c] = lam * emp + (1 - lam) * priors.get(d.label, 0.5)
    return out


def _s5(ctx, name="S5", shuffle=False):
    if ctx.diagnosis is None and ctx.cfg.use_diagnosis:
        raise ValueError("S5 needs a diagnosis (or set strategy.use_diagnosis=false)")
    pri, detail = s5_priorities(ctx, ctx.val_report)
    cells = list(ctx.grid.train_cells)
    if shuffle:
        rng = np.random.default_rng([ctx.pipeline_seed, ctx.round_idx, 55])
        vals = np.array([pri[c] for c in cells])
        rng.shuffle(vals)
        pri = dict(zip(cells, vals.tolist(), strict=True))
    w = _with_floor(pri, cells, ctx.cfg.floor)
    return AcquisitionPlan(name, "sampler", ctx.budget, distribution=CellMixture(w), cell_weights=w,
                           notes={"detail": {str(c): v for c, v in detail.items()}})


def _s5_shuffled(ctx):
    return _s5(ctx, "S5-shuffled", shuffle=True)


def _s7(ctx):
    """Oracle: S5 allocation using TEST per-cell success. Non-deployable upper bound."""
    if not ctx.allow_oracle:
        raise LeakageError("S7 uses test data by design; pass allow_oracle=True explicitly")
    rep = ctx.oracle_report
    if rep is None or rep.split != "test":
        raise ValueError("S7 needs a test-split oracle report")
    pri, detail = s5_priorities(ctx, rep)
    w = _with_floor(pri, list(ctx.grid.train_cells), ctx.cfg.floor)
    return AcquisitionPlan("S7", "sampler", ctx.budget, distribution=CellMixture(w), cell_weights=w,
                           notes={"oracle": True})
