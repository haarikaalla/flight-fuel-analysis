"""Prediction, interval, calibration and selective-decision metrics."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    error = p - y
    denominator = np.sum((y - y.mean()) ** 2)
    return {
        "n": int(len(y)),
        "mae_kg": float(np.mean(np.abs(error))),
        "rmse_kg": float(np.sqrt(np.mean(error ** 2))),
        "bias_kg": float(np.mean(error)),
        "r2": float(1 - np.sum(error ** 2) / denominator) if denominator else None,
    }


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                     alpha: float = 0.1) -> dict:
    y, lo, hi = map(lambda value: np.asarray(value, float), (y_true, lower, upper))
    if np.any(hi < lo) or not 0 < alpha < 1:
        raise ValueError("Intervals and alpha are invalid")
    width = hi - lo
    miss_low, miss_high = y < lo, y > hi
    score = width + (2 / alpha) * (lo - y) * miss_low + (2 / alpha) * (y - hi) * miss_high
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width_kg": float(width.mean()),
        "median_width_kg": float(np.median(width)),
        "winkler_score": float(score.mean()),
    }


def calibration_metrics(probability: np.ndarray, outcome: np.ndarray, bins: int = 10) -> dict:
    p, y = np.asarray(probability, float), np.asarray(outcome, float)
    if np.any((p < 0) | (p > 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError("Calibration metrics require probabilities and binary outcomes")
    edges = np.linspace(0, 1, bins + 1)
    ids = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    ece = 0.0
    rows = []
    for index in range(bins):
        member = ids == index
        if member.any():
            row = {"n": int(member.sum()), "predicted": float(p[member].mean()),
                   "observed": float(y[member].mean())}
            ece += member.mean() * abs(row["predicted"] - row["observed"])
            rows.append(row)
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece), "bins": rows}


def selective_metrics(y_true: np.ndarray, y_pred: np.ndarray, selected: np.ndarray,
                      risk: np.ndarray, utility: np.ndarray | None = None,
                      risk_budget: float | None = None) -> dict:
    y, p, z, r = (np.asarray(y_true, float), np.asarray(y_pred, float),
                  np.asarray(selected, bool), np.asarray(risk, float))
    if not z.any():
        raise ValueError("At least one item must be selected")
    utility = np.ones(len(y)) if utility is None else np.asarray(utility, float)
    used = float(r[z].sum())
    return {
        "selected_n": int(z.sum()), "selected_fraction": float(z.mean()),
        "selected_mae_kg": float(np.abs(p[z] - y[z]).mean()),
        "selected_risk": used, "selected_utility": float(utility[z].sum()),
        "risk_budget_violation": None if risk_budget is None else float(max(0, used - risk_budget)),
    }


def conditional_interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                                 groups: np.ndarray, alpha: float = 0.1) -> list[dict]:
    groups = np.asarray(groups).astype(str)
    rows = []
    for name in sorted(np.unique(groups)):
        member = groups == name
        rows.append({"group": name, "n": int(member.sum()),
                     **interval_metrics(np.asarray(y_true)[member], np.asarray(lower)[member],
                                        np.asarray(upper)[member], alpha)})
    return rows

