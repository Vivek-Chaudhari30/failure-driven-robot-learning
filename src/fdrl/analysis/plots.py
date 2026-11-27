"""Plots listed in EXPERIMENT_DESIGN section 11. Every function writes a PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fdrl.conditions import CellGrid  # noqa: E402
from fdrl.failures import CODES, NAMES  # noqa: E402


def _save(fig, path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def scaling_curve(curves: dict[str, dict[int, list[float]]], path) -> str:
    """curves[label][N] = per-seed success values. Success vs log N with per-seed dots."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for label, d in curves.items():
        ns = sorted(d)
        means = [np.mean(d[n]) for n in ns]
        ax.plot(ns, means, marker="o", label=label)
        for n in ns:
            ax.scatter([n] * len(d[n]), d[n], s=8, alpha=0.35, color=ax.lines[-1].get_color())
    ax.set_xscale("log")
    ax.set_xlabel("number of demonstrations")
    ax.set_ylabel("cell-balanced success")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def strategy_comparison(values: dict[str, list[float]], path, title: str = "") -> str:
    """values[strategy] = per-seed success. Bars = mean, dots = seeds."""
    names = list(values)
    fig, ax = plt.subplots(figsize=(max(4, 0.7 * len(names)), 3.5))
    means = [np.mean(values[n]) for n in names]
    ax.bar(range(len(names)), means, color="#7a9cc6")
    for i, n in enumerate(names):
        ax.scatter([i] * len(values[n]), values[n], s=10, color="k", zorder=3)
    ax.set_xticks(range(len(names)), names, rotation=30)
    ax.set_ylabel("cell-balanced success")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    return _save(fig, path)


def coverage_heatmap(grid: CellGrid, counts: dict[int, float], path, title: str = "") -> str:
    """Per-cell counts laid out as (quadrant x yaw bin) rows by friction-bin columns."""
    rows = 4 * grid.n_yaw
    M = np.full((rows, grid.n_mu), np.nan)
    for c in range(grid.n_cells):
        q, y, m = grid.bins(c)
        M[q * grid.n_yaw + y, m] = counts.get(c, 0)
    fig, ax = plt.subplots(figsize=(3.5, 6))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    for c in grid.holdout:
        q, y, m = grid.bins(c)
        ax.text(m, q * grid.n_yaw + y, "H", ha="center", va="center", color="w")
    ax.set_xlabel("friction bin (low → high)")
    ax.set_ylabel("quadrant × yaw bin")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return _save(fig, path)


def failure_distribution(before: dict[str, int], after: dict[str, int], path) -> str:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(CODES))
    ax.bar(x - 0.2, [before.get(c, 0) for c in CODES], 0.4, label="before")
    ax.bar(x + 0.2, [after.get(c, 0) for c in CODES], 0.4, label="after")
    ax.set_xticks(x, [f"{c}\n{NAMES[c][:10]}" for c in CODES], fontsize=7)
    ax.set_ylabel("episodes")
    ax.legend()
    return _save(fig, path)


def iteration_curves(curves: dict[str, tuple[list[float], list[list[float]]]], path, xlabel: str = "cumulative cost") -> str:
    """curves[strategy] = (x per round, [per-seed success per round])."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, (xs, ys) in curves.items():
        Y = np.asarray(ys)
        ax.plot(xs, Y.mean(axis=0), marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cell-balanced success")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def confusion_matrix(labels: list[str], truth: list[str], pred: list[str], path) -> str:
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(truth, pred, strict=True):
        M[idx[t], idx[p]] += 1
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(M, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, M[i, j], ha="center", va="center")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("diagnosis")
    ax.set_ylabel("intervention ground truth")
    return _save(fig, path)
