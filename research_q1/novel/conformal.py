"""Conformal interval and selection primitives used by PI-MCS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


def finite_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, float)
    if not len(scores) or not np.isfinite(scores).all() or not 0 < alpha < 1:
        raise ValueError("Finite conformal scores and alpha in (0,1) are required")
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * (1 - alpha))))
    return float(np.partition(scores, rank - 1)[rank - 1])


def split_interval(prediction: np.ndarray, sigma: np.ndarray, calibration_scores: np.ndarray,
                   *, alpha: float = 0.1, lower_bound: float | np.ndarray = 0.0,
                   upper_bound: float | np.ndarray = np.inf) -> tuple[np.ndarray, np.ndarray, dict]:
    q = finite_quantile(calibration_scores, alpha)
    prediction, sigma = np.asarray(prediction, float), np.asarray(sigma, float)
    half = q * np.maximum(sigma, 1e-9)
    return (np.maximum(lower_bound, prediction - half),
            np.minimum(upper_bound, prediction + half),
            {"quantile": q, "calibration_n": int(len(calibration_scores)),
             "nominal_coverage": 1 - alpha})


class HierarchicalMondrian:
    """Aircraft×phase calibration with aircraft, phase and global fallbacks."""

    def __init__(self, alpha: float = 0.1, min_group_n: int = 50):
        self.alpha = alpha
        self.min_group_n = min_group_n

    def fit(self, scores: np.ndarray, aircraft: np.ndarray, phase: np.ndarray) -> "HierarchicalMondrian":
        scores = np.asarray(scores, float)
        aircraft, phase = np.asarray(aircraft).astype(str), np.asarray(phase).astype(str)
        if not (len(scores) == len(aircraft) == len(phase)):
            raise ValueError("Mondrian calibration arrays must have equal lengths")
        self.global_ = finite_quantile(scores, self.alpha)
        self.pair_: dict[tuple[str, str], tuple[float, int]] = {}
        self.aircraft_: dict[str, tuple[float, int]] = {}
        self.phase_: dict[str, tuple[float, int]] = {}
        for name in np.unique(aircraft):
            member = aircraft == name
            if member.sum() >= self.min_group_n:
                self.aircraft_[name] = (finite_quantile(scores[member], self.alpha), int(member.sum()))
        for name in np.unique(phase):
            member = phase == name
            if member.sum() >= self.min_group_n:
                self.phase_[name] = (finite_quantile(scores[member], self.alpha), int(member.sum()))
        for a, p in sorted(set(zip(aircraft, phase))):
            member = (aircraft == a) & (phase == p)
            if member.sum() >= self.min_group_n:
                self.pair_[(a, p)] = (finite_quantile(scores[member], self.alpha), int(member.sum()))
        self.calibration_n_ = len(scores)
        return self

    def multipliers(self, aircraft: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, list[str]]:
        if not hasattr(self, "global_"):
            raise RuntimeError("HierarchicalMondrian must be fit first")
        values, levels = [], []
        for a, p in zip(np.asarray(aircraft).astype(str), np.asarray(phase).astype(str)):
            if (a, p) in self.pair_:
                q, level = self.pair_[(a, p)][0], "aircraft+phase"
            elif a in self.aircraft_:
                q, level = self.aircraft_[a][0], "aircraft"
            elif p in self.phase_:
                q, level = self.phase_[p][0], "phase"
            else:
                q, level = self.global_, "global"
            values.append(q)
            levels.append(level)
        return np.asarray(values), levels

    def interval(self, prediction: np.ndarray, sigma: np.ndarray, aircraft: np.ndarray,
                 phase: np.ndarray, tank_capacity_kg: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        multipliers, levels = self.multipliers(aircraft, phase)
        half = multipliers * np.maximum(np.asarray(sigma, float), 1e-9)
        prediction = np.asarray(prediction, float)
        unique, counts = np.unique(levels, return_counts=True)
        return (np.maximum(0.0, prediction - half),
                np.minimum(np.asarray(tank_capacity_kg, float), prediction + half),
                {"fallback_counts": dict(zip(unique.tolist(), counts.astype(int).tolist())),
                 "calibration_n": self.calibration_n_, "nominal_coverage": 1 - self.alpha})


def weighted_quantile(scores: np.ndarray, calibration_weights: np.ndarray,
                      test_weight: float, alpha: float) -> float:
    """Weighted conformal quantile including the test point's mass at infinity."""
    scores, weights = np.asarray(scores, float), np.asarray(calibration_weights, float)
    if len(scores) != len(weights) or np.any(weights < 0) or test_weight < 0:
        raise ValueError("Weighted conformal inputs are invalid")
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[order])
    target = (1 - alpha) * (weights.sum() + test_weight)
    index = np.searchsorted(cumulative, target, side="left")
    return float(scores[order[index]]) if index < len(scores) else float("inf")


def weighted_intervals(prediction: np.ndarray, sigma: np.ndarray, calibration_scores: np.ndarray,
                       calibration_weights: np.ndarray, test_weights: np.ndarray,
                       *, alpha: float = 0.1, tank_capacity_kg: np.ndarray | None = None):
    q = np.array([weighted_quantile(calibration_scores, calibration_weights, value, alpha)
                  for value in np.asarray(test_weights, float)])
    half = q * np.maximum(np.asarray(sigma, float), 1e-9)
    prediction = np.asarray(prediction, float)
    cap = np.full(len(prediction), np.inf) if tank_capacity_kg is None else np.asarray(tank_capacity_kg, float)
    return np.maximum(0.0, prediction - half), np.minimum(cap, prediction + half), q


def cqr_interval(lower_prediction: np.ndarray, upper_prediction: np.ndarray,
                 calibration_y: np.ndarray, calibration_lower: np.ndarray,
                 calibration_upper: np.ndarray, *, alpha: float = 0.1,
                 tank_capacity_kg: np.ndarray | None = None):
    scores = np.maximum(np.asarray(calibration_lower) - np.asarray(calibration_y),
                        np.asarray(calibration_y) - np.asarray(calibration_upper))
    q = finite_quantile(scores, alpha)
    lo = np.maximum(0.0, np.asarray(lower_prediction, float) - q)
    hi = np.asarray(upper_prediction, float) + q
    if tank_capacity_kg is not None:
        hi = np.minimum(hi, np.asarray(tank_capacity_kg, float))
    return lo, hi, {"correction_kg": q, "calibration_n": int(len(scores))}


@dataclass
class AdaptiveConformal:
    calibration_scores: np.ndarray
    alpha: float = 0.1
    gamma: float = 0.01
    alpha_state: float | None = None

    def __post_init__(self):
        self.calibration_scores = np.asarray(self.calibration_scores, float)
        self.alpha_state = self.alpha if self.alpha_state is None else self.alpha_state

    def current_quantile(self) -> float:
        local_alpha = float(np.clip(self.alpha_state, 1 / (len(self.calibration_scores) + 1), 0.999))
        return finite_quantile(self.calibration_scores, local_alpha)

    def update(self, missed: bool) -> float:
        self.alpha_state = float(np.clip(
            self.alpha_state + self.gamma * (self.alpha - float(missed)),
            1 / (len(self.calibration_scores) + 1), 0.999,
        ))
        return self.alpha_state


def conformal_p_values(calibration_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    calibration, test = np.asarray(calibration_scores, float), np.asarray(test_scores, float)
    return np.array([(1 + np.sum(calibration >= value)) / (len(calibration) + 1) for value in test])


def benjamini_hochberg(p_values: np.ndarray, q: float) -> np.ndarray:
    """BH rejection set; FDR needs independent/PRDS-valid conformal p-values."""
    p = np.asarray(p_values, float)
    order = np.argsort(p, kind="stable")
    passed = p[order] <= q * np.arange(1, len(p) + 1) / len(p)
    selected = np.zeros(len(p), dtype=bool)
    if passed.any():
        selected[order[:np.flatnonzero(passed)[-1] + 1]] = True
    return selected


def clopper_pearson_upper(errors: int, n: int, delta: float) -> float:
    if not 0 <= errors <= n or not 0 < delta < 1:
        raise ValueError("Binomial-bound inputs are invalid")
    return 1.0 if errors == n else float(beta.ppf(1 - delta, errors + 1, n - errors))


def bonferroni_risk_control_threshold(scores: np.ndarray, losses: np.ndarray,
                                      candidates: np.ndarray, target_risk: float,
                                      delta: float = 0.05) -> dict:
    """Select largest fixed threshold with simultaneous binomial risk UCB.

    Guarantee: under iid calibration examples, binary monotone loss, and a
    candidate grid fixed independently of calibration labels, the selected
    threshold's population risk is at most target_risk with probability at
    least 1-delta. This is not a conditional or post-shift guarantee.
    """
    scores, losses, candidates = map(np.asarray, (scores, losses, candidates))
    if np.any((losses != 0) & (losses != 1)):
        raise ValueError("Risk control currently supports binary losses")
    rows = []
    corrected = delta / len(candidates)
    for threshold in sorted(map(float, candidates)):
        selected = scores <= threshold
        if selected.any():
            upper = clopper_pearson_upper(int(losses[selected].sum()), int(selected.sum()), corrected)
        else:
            upper = 1.0
        rows.append({"threshold": threshold, "selected_n": int(selected.sum()), "risk_ucb": upper})
    feasible = [row for row in rows if row["risk_ucb"] <= target_risk]
    chosen = max(feasible, key=lambda row: row["threshold"]) if feasible else None
    return {"chosen": chosen, "candidates": rows, "confidence": 1 - delta,
            "guarantee_scope": "iid calibration, fixed candidate grid, binary monotone loss"}

