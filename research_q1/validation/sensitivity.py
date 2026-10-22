"""Pre-specified sensitivity sweeps for calibration and risk-budget choices."""

from __future__ import annotations

import numpy as np

from research_q1.novel.conformal import finite_quantile
from research_q1.novel.selection import exact_risk_budget


def calibration_size_sweep(scores: np.ndarray, alphas=(.05, .1, .2),
                           fractions=(.25, .5, .75, 1.0)) -> list[dict]:
    """Chronological prefixes only; never random calibration subsamples."""
    scores = np.asarray(scores, float)
    rows = []
    for fraction in fractions:
        n = max(2, int(np.floor(fraction * len(scores))))
        for alpha in alphas:
            rows.append({"fraction": fraction, "n": n, "alpha": alpha,
                         "quantile": finite_quantile(scores[:n], alpha)})
    return rows


def risk_budget_sweep(utility: np.ndarray, risk: np.ndarray, minimum_coverage: float,
                      multipliers=(1.0, 1.25, 1.5)) -> list[dict]:
    risk = np.asarray(risk, float)
    minimum = int(np.ceil(minimum_coverage * len(risk)))
    base = np.sort(risk)[:minimum].sum()
    rows = []
    for multiplier in multipliers:
        selected, audit = exact_risk_budget(utility, risk, float(multiplier * base), minimum_coverage)
        rows.append({"multiplier": multiplier, "selected_n": int(selected.sum()), **audit})
    return rows


def risk_perturbation_sweep(utility: np.ndarray, risk: np.ndarray, budget: float,
                            minimum_coverage: float, noise_fractions=(.1, .25, .5),
                            repetitions: int = 100, seed: int = 20251022) -> list[dict]:
    base, _ = exact_risk_budget(utility, risk, budget, minimum_coverage)
    rng, scale = np.random.default_rng(seed), float(np.std(risk))
    rows = []
    for fraction in noise_fractions:
        jaccard, utility_values = [], []
        for _ in range(repetitions):
            perturbed = np.maximum(0, risk + rng.normal(0, fraction * scale, len(risk)))
            try:
                selected, _ = exact_risk_budget(utility, perturbed, budget, minimum_coverage)
            except ValueError:
                continue
            union = np.sum(base | selected)
            jaccard.append(np.sum(base & selected) / union if union else 1.0)
            utility_values.append(np.asarray(utility)[selected].sum())
        rows.append({"noise_fraction_of_risk_sd": fraction, "successful_repetitions": len(jaccard),
                     "mean_jaccard": float(np.mean(jaccard)) if jaccard else None,
                     "mean_utility": float(np.mean(utility_values)) if utility_values else None})
    return rows

