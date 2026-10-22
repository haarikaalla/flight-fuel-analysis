"""Cluster bootstrap, paired tests, effect sizes and multiplicity correction."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def cluster_bootstrap_difference(values_a: np.ndarray, values_b: np.ndarray,
                                 clusters: np.ndarray, *, repetitions: int = 2000,
                                 seed: int = 20251022) -> dict:
    """Paired B-A mean difference; resample aircraft or route clusters."""
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    clusters = np.asarray(clusters).astype(str)
    if not (len(a) == len(b) == len(clusters)):
        raise ValueError("Paired statistics inputs must align")
    names = np.unique(clusters)
    if len(names) < 2:
        raise ValueError("Cluster bootstrap requires at least two clusters")
    cluster_sum = np.array([(b[clusters == name] - a[clusters == name]).sum() for name in names])
    cluster_n = np.array([(clusters == name).sum() for name in names])
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for iteration in range(repetitions):
        sampled = rng.integers(0, len(names), size=len(names))
        draws[iteration] = cluster_sum[sampled].sum() / cluster_n[sampled].sum()
    delta = b - a
    return {
        "mean_difference_b_minus_a": float(delta.mean()),
        "cluster_bootstrap_95pct_ci": np.quantile(draws, [.025, .975]).tolist(),
        "clusters": int(len(names)), "repetitions": repetitions,
        "standardized_effect": float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) else 0.0,
    }


def cluster_bootstrap_mean(values: np.ndarray, clusters: np.ndarray, *,
                           repetitions: int = 2000, seed: int = 20251022) -> dict:
    values, clusters = np.asarray(values, float), np.asarray(clusters).astype(str)
    names = np.unique(clusters)
    if len(names) < 2:
        raise ValueError("Cluster bootstrap requires at least two clusters")
    sums = np.array([values[clusters == name].sum() for name in names])
    counts = np.array([(clusters == name).sum() for name in names])
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for iteration in range(repetitions):
        sampled = rng.integers(0, len(names), size=len(names))
        draws[iteration] = sums[sampled].sum() / counts[sampled].sum()
    return {"mean": float(values.mean()),
            "cluster_bootstrap_95pct_ci": np.quantile(draws, [.025, .975]).tolist(),
            "clusters": int(len(names)), "repetitions": repetitions}


def paired_wilcoxon(values_a: np.ndarray, values_b: np.ndarray) -> dict:
    difference = np.asarray(values_b, float) - np.asarray(values_a, float)
    statistic, p_value = wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
    nonzero = difference[difference != 0]
    rank_biserial = 0.0 if not len(nonzero) else float((np.sum(nonzero > 0) - np.sum(nonzero < 0)) / len(nonzero))
    return {"statistic": float(statistic), "p_value": float(p_value),
            "rank_biserial_direction": rank_biserial}


def adjust_pvalues(p_values: np.ndarray, method: str = "holm") -> np.ndarray:
    p = np.asarray(p_values, float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0,1]")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    if method == "holm":
        ordered = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    elif method == "bh":
        raw = len(p) * p[order] / np.arange(1, len(p) + 1)
        ordered = np.minimum.accumulate(raw[::-1])[::-1]
    else:
        raise ValueError("method must be holm or bh")
    adjusted[order] = np.minimum(1.0, ordered)
    return adjusted

