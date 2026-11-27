"""Condition space Θ, cells, hold-out set H, seed streams and samplers q(θ).

EXPERIMENT_DESIGN sections 3-5. Leakage guards (ARCHITECTURE section 5, item 3):
  * seed streams are disjoint integer ranges, checked on every allocation;
  * acquisition/validation sampling refuses any condition that falls in a hold-out cell.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from fdrl.config import ConditionSpaceCfg

SPLITS = ("calib", "acq", "val", "test")

# Disjoint seed ranges [lo, hi). EXPERIMENT_DESIGN section 5.
SEED_STREAMS: dict[str, tuple[int, int]] = {
    "calib": (0, 10_000),
    "acq": (1_000_000, 2_000_000),
    "val": (2_000_000, 3_000_000),
    "test": (3_000_000, 4_000_000),
}
# Layout inside each stream: lo + pipeline_seed * ps_stride + block * block_stride + i.
_STRIDES: dict[str, tuple[int, int]] = {
    "calib": (1_000, 1_000),  # (ps_stride, block_stride); calib uses block 0 only
    "acq": (50_000, 10_000),  # up to 5 blocks (rounds incl. D0) of 10k attempts per seed
    "val": (10_000, 10_000),
    "test": (20_000, 20_000),
}


class LeakageError(RuntimeError):
    """Raised when code tries to use data or seeds it must not touch."""


@dataclass(frozen=True)
class Condition:
    """One point θ ∈ Θ. x, y are offsets (m) from the nominal cube position; yaw in radians."""

    x: float
    y: float
    yaw: float
    mu: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.mu], dtype=np.float64)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def seed_for(stream: str, pipeline_seed: int, block: int, i: int) -> int:
    """Deterministic seed in a disjoint stream. Raises LeakageError if it would leave its range."""
    if stream not in SEED_STREAMS:
        raise ValueError(f"unknown seed stream {stream!r}")
    lo, hi = SEED_STREAMS[stream]
    ps_stride, block_stride = _STRIDES[stream]
    if i < 0 or i >= block_stride or block < 0 or pipeline_seed < 0:
        raise LeakageError(f"seed index out of block: stream={stream} block={block} i={i}")
    if block_stride * (block + 1) > ps_stride:
        raise LeakageError(f"block {block} overflows pipeline-seed stride in stream {stream}")
    s = lo + pipeline_seed * ps_stride + block * block_stride + i
    if not (lo <= s < hi):
        raise LeakageError(f"seed {s} outside stream {stream} range [{lo}, {hi})")
    return s


def stream_of(seed: int) -> str:
    for name, (lo, hi) in SEED_STREAMS.items():
        if lo <= seed < hi:
            return name
    raise ValueError(f"seed {seed} belongs to no stream")


class CellGrid:
    """Position quadrant (2x2) × yaw bins × log-spaced friction bins."""

    def __init__(self, cfg: ConditionSpaceCfg):
        self.cfg = cfg
        self.n_yaw = cfg.n_yaw_bins
        self.n_mu = cfg.n_mu_bins
        self.n_cells = 4 * self.n_yaw * self.n_mu
        lo, hi = (math.radians(v) for v in cfg.yaw_range_deg)
        self.yaw_edges = np.linspace(lo, hi, self.n_yaw + 1)
        self.logmu_edges = np.linspace(math.log(cfg.mu_range[0]), math.log(cfg.mu_range[1]), self.n_mu + 1)
        self.holdout: frozenset[int] = frozenset(self.index(*map(int, h)) for h in cfg.holdout)
        self.train_cells: tuple[int, ...] = tuple(c for c in range(self.n_cells) if c not in self.holdout)
        self._check_holdout()

    # --- indexing -------------------------------------------------------------------------
    def index(self, quadrant: int, yaw_bin: int, mu_bin: int) -> int:
        if not (0 <= quadrant < 4 and 0 <= yaw_bin < self.n_yaw and 0 <= mu_bin < self.n_mu):
            raise ValueError("cell bin out of range")
        return (quadrant * self.n_yaw + yaw_bin) * self.n_mu + mu_bin

    def bins(self, cell: int) -> tuple[int, int, int]:
        mu_bin = cell % self.n_mu
        yaw_bin = (cell // self.n_mu) % self.n_yaw
        quadrant = cell // (self.n_mu * self.n_yaw)
        return quadrant, yaw_bin, mu_bin

    def cell_of(self, c: Condition) -> int:
        quadrant = (2 if c.x >= 0 else 0) + (1 if c.y >= 0 else 0)
        yb = int(np.clip(np.searchsorted(self.yaw_edges, c.yaw, side="right") - 1, 0, self.n_yaw - 1))
        mb = int(np.clip(np.searchsorted(self.logmu_edges, math.log(c.mu), side="right") - 1, 0, self.n_mu - 1))
        return self.index(quadrant, yb, mb)

    def cells_of(self, conds: Sequence[Condition]) -> np.ndarray:
        return np.array([self.cell_of(c) for c in conds], dtype=np.int64)

    # --- geometry -------------------------------------------------------------------------
    def sample_in_cell(self, cell: int, rng: np.random.Generator) -> Condition:
        q, yb, mb = self.bins(cell)
        x_lo, x_hi = self.cfg.x_range
        y_lo, y_hi = self.cfg.y_range
        xr = (0.0, x_hi) if q >= 2 else (x_lo, 0.0)
        yr = (0.0, y_hi) if q % 2 == 1 else (y_lo, 0.0)
        x = rng.uniform(*xr)
        y = rng.uniform(*yr)
        yaw = rng.uniform(self.yaw_edges[yb], self.yaw_edges[yb + 1])
        mu = math.exp(rng.uniform(self.logmu_edges[mb], self.logmu_edges[mb + 1]))
        return Condition(float(x), float(y), float(yaw), float(mu))

    def normalize(self, conds: Sequence[Condition] | np.ndarray) -> np.ndarray:
        """Map θ to [0,1]^4 (log friction) for distance-based strategies (S4)."""
        a = np.asarray([c.as_array() for c in conds]) if not isinstance(conds, np.ndarray) else conds
        a = np.atleast_2d(a).astype(np.float64)
        lo = np.array([self.cfg.x_range[0], self.cfg.y_range[0], self.yaw_edges[0], self.logmu_edges[0]])
        hi = np.array([self.cfg.x_range[1], self.cfg.y_range[1], self.yaw_edges[-1], self.logmu_edges[-1]])
        b = a.copy()
        b[:, 3] = np.log(b[:, 3])
        return (b - lo) / (hi - lo)

    def _check_holdout(self) -> None:
        """Every single-factor bin must still appear in at least one training cell."""
        seen_q, seen_y, seen_m = set(), set(), set()
        for c in self.train_cells:
            q, y, m = self.bins(c)
            seen_q.add(q)
            seen_y.add(y)
            seen_m.add(m)
        if len(seen_q) < 4 or len(seen_y) < self.n_yaw or len(seen_m) < self.n_mu:
            raise ValueError("hold-out set H removes an entire factor bin; only combinations may be held out")


# --- distributions q(θ) -------------------------------------------------------------------

class ConditionDistribution:
    """Base class: deterministic sampling from a seed."""

    def sample(self, grid: CellGrid, seed: int) -> Condition:  # pragma: no cover - interface
        raise NotImplementedError


class CellMixture(ConditionDistribution):
    """Mixture over cells, uniform within each cell. Weights over cells need not be normalized."""

    def __init__(self, weights: dict[int, float]):
        cells = sorted(c for c, w in weights.items() if w > 0)
        if not cells:
            raise ValueError("CellMixture needs at least one cell with positive weight")
        w = np.array([weights[c] for c in cells], dtype=np.float64)
        self.cells = np.array(cells)
        self.probs = w / w.sum()

    def sample(self, grid: CellGrid, seed: int) -> Condition:
        rng = np.random.default_rng(seed)
        cell = int(rng.choice(self.cells, p=self.probs))
        return grid.sample_in_cell(cell, rng)

    def as_dict(self) -> dict[int, float]:
        return {int(c): float(p) for c, p in zip(self.cells, self.probs, strict=True)}


def uniform_train(grid: CellGrid) -> CellMixture:
    return CellMixture({c: 1.0 for c in grid.train_cells})


class NominalDistribution(ConditionDistribution):
    """p_nom for regime R1: narrow, human-like collection habits. Rejects hold-out cells."""

    def __init__(self, cfg: ConditionSpaceCfg):
        self.cfg = cfg

    def sample(self, grid: CellGrid, seed: int) -> Condition:
        rng = np.random.default_rng(seed)
        cfg = self.cfg
        for _ in range(1000):
            x = _truncnorm(rng, cfg.nom_pos_std, *cfg.x_range)
            y = _truncnorm(rng, cfg.nom_pos_std, *cfg.y_range)
            yaw = math.radians(_truncnorm(rng, cfg.nom_yaw_std_deg, *cfg.yaw_range_deg))
            mu = math.exp(rng.uniform(math.log(cfg.nom_mu_min), math.log(cfg.mu_range[1])))
            c = Condition(x, y, yaw, mu)
            if grid.cell_of(c) not in grid.holdout:
                return c
        raise RuntimeError("p_nom could not avoid hold-out cells; check the configuration")


def _truncnorm(rng: np.random.Generator, std: float, lo: float, hi: float) -> float:
    for _ in range(1000):
        v = rng.normal(0.0, std)
        if lo <= v <= hi:
            return float(v)
    return float(np.clip(0.0, lo, hi))


def sample_condition(grid: CellGrid, dist: ConditionDistribution, seed: int, split: str) -> Condition:
    """Sample θ for a split, enforcing the hold-out guard for acquisition and validation."""
    if split not in SPLITS:
        raise ValueError(split)
    if stream_of(seed) != split:
        raise LeakageError(f"seed {seed} is from stream {stream_of(seed)}, not {split}")
    c = dist.sample(grid, seed)
    if split in ("acq", "val") and grid.cell_of(c) in grid.holdout:
        raise LeakageError(f"refusing to sample hold-out cell {grid.cell_of(c)} for split {split}")
    return c


@dataclass(frozen=True)
class EpisodePlanItem:
    cell: int
    seed: int
    condition: Condition


def eval_plan(grid: CellGrid, split: str, pipeline_seed: int, n_per_cell: int) -> list[EpisodePlanItem]:
    """Identical evaluation episodes for every strategy within a pipeline seed (pairing).

    Validation covers training cells only; test covers all cells including H.
    """
    if split not in ("val", "test"):
        raise ValueError(split)
    cells = grid.train_cells if split == "val" else tuple(range(grid.n_cells))
    plan: list[EpisodePlanItem] = []
    i = 0
    for cell in cells:
        for _ in range(n_per_cell):
            s = seed_for(split, pipeline_seed, 0, i)
            i += 1
            cond = grid.sample_in_cell(cell, np.random.default_rng(s))
            plan.append(EpisodePlanItem(cell, s, cond))
    return plan
