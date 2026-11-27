"""Statistics for reporting (EXPERIMENT_DESIGN section 11).

Unit of replication = pipeline seed. Paired differences, bootstrap CIs over seeds, Holm-Bonferroni
across the primary family, IQM alongside the mean, N*(τ) and ρ(τ) for data efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass
class CI:
    mean: float
    lo: float
    hi: float
    p_value: float | None = None
    n: int = 0


def bootstrap_ci(x, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0, stat=np.mean) -> CI:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return CI(float("nan"), float("nan"), float("nan"), None, 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.apply_along_axis(stat, 1, x[idx]) if stat is not np.mean else x[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return CI(float(stat(x)), float(lo), float(hi), None, len(x))


def paired_bootstrap(a, b, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> CI:
    """CI and two-sided bootstrap p-value for mean(a - b), paired by seed (same index)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arrays must align by pipeline seed")
    d = a - b
    ci = bootstrap_ci(d, n_boot, alpha, seed)
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    ci.p_value = float(min(1.0, max(p, 1.0 / n_boot)))
    return ci


def stratified_bootstrap_ci(values_by_stratum: dict, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> CI:
    """Resample seeds within each stratum (e.g. regime or task) and average the strata means."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(v, dtype=np.float64) for v in values_by_stratum.values()]
    point = float(np.mean([a.mean() for a in arrays]))
    boots = np.zeros(n_boot)
    for a in arrays:
        idx = rng.integers(0, len(a), size=(n_boot, len(a)))
        boots += a[idx].mean(axis=1)
    boots /= len(arrays)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return CI(point, float(lo), float(hi), None, sum(len(a) for a in arrays))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone)."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def iqm(x) -> float:
    """Interquartile mean (rliable's recommended aggregate)."""
    x = np.asarray(x, dtype=np.float64)
    return float(sps.trim_mean(x, 0.25)) if len(x) else float("nan")


def n_star(sizes, success, tau: float) -> float:
    """Smallest dataset size reaching success τ, by linear interpolation in log N.
    Returns inf if τ is never reached, and sizes[0] if already reached at the smallest size."""
    s = np.asarray(sizes, dtype=np.float64)
    y = np.asarray(success, dtype=np.float64)
    order = np.argsort(s)
    s, y = s[order], y[order]
    if y[0] >= tau:
        return float(s[0])
    for i in range(1, len(s)):
        if y[i] >= tau:
            ls0, ls1 = np.log(s[i - 1]), np.log(s[i])
            f = (tau - y[i - 1]) / max(y[i] - y[i - 1], 1e-12)
            return float(np.exp(ls0 + f * (ls1 - ls0)))
    return float("inf")


def rho(uniform_sizes, uniform_success_by_seed: np.ndarray, method_n: np.ndarray, method_success: np.ndarray,
        tau: float, n_boot: int = 10_000, seed: int = 0) -> CI:
    """ρ(τ) = N*_uniform(τ) / N*_method(τ), bootstrap over seeds.

    uniform_success_by_seed: [n_seeds, n_sizes]; method_n / method_success: [n_seeds, n_points]
    (e.g. cumulative demos and success along an iterative-acquisition curve).
    """
    U = np.asarray(uniform_success_by_seed, dtype=np.float64)
    MN, MS = np.asarray(method_n, dtype=np.float64), np.asarray(method_success, dtype=np.float64)

    def ratio(idx):
        nu = n_star(uniform_sizes, U[idx].mean(axis=0), tau)
        nm = n_star(MN[idx].mean(axis=0), MS[idx].mean(axis=0), tau)
        return nu / nm if np.isfinite(nm) and nm > 0 else float("nan")

    k = U.shape[0]
    point = ratio(np.arange(k))
    rng = np.random.default_rng(seed)
    boots = np.array([ratio(rng.integers(0, k, size=k)) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return CI(point, float("nan"), float("nan"), None, k)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return CI(float(point), float(lo), float(hi), None, k)


def required_seeds(sd_paired_diff: float, effect: float = 0.05, alpha: float = 0.05, power: float = 0.8) -> int:
    """Seeds needed for a paired two-sided test to detect `effect` (normal approximation)."""
    if sd_paired_diff <= 0:
        return 2
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    return int(np.ceil((z * sd_paired_diff / effect) ** 2))
