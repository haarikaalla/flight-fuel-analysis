"""Exact, robust and approximate selection under a calibrated risk budget."""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix


def _validate(utility, risk, budget, minimum_coverage):
    utility, risk = np.asarray(utility, float), np.asarray(risk, float)
    if len(utility) != len(risk) or not len(risk) or np.any(risk < 0):
        raise ValueError("Utility and nonnegative risk arrays must align")
    if budget < 0 or not 0 < minimum_coverage <= 1:
        raise ValueError("Budget and minimum coverage are invalid")
    return utility, risk, int(math.ceil(minimum_coverage * len(risk)))


def exact_risk_budget(utility: np.ndarray, risk: np.ndarray, budget: float,
                      minimum_coverage: float) -> tuple[np.ndarray, dict]:
    utility, risk, minimum = _validate(utility, risk, budget, minimum_coverage)
    n, started = len(risk), time.perf_counter()
    matrix = csr_matrix(np.vstack([risk, np.ones(n)]))
    constraint = LinearConstraint(matrix, [-np.inf, minimum], [budget, np.inf])
    result = milp(-utility, integrality=np.ones(n), bounds=Bounds(0, 1),
                  constraints=constraint, options={"presolve": True})
    if not result.success or result.x is None:
        raise ValueError(f"Risk-budget MILP is infeasible: {result.message}")
    selected = result.x > 0.5
    return selected, {"method": "exact_milp", "solve_time_s": time.perf_counter() - started,
                      "solver_status": int(result.status), "solver_message": result.message,
                      "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
                      "risk_used": float(risk[selected].sum()), "utility": float(utility[selected].sum())}


def robust_risk_budget(utility: np.ndarray, risk: np.ndarray, deviation: np.ndarray,
                       budget: float, minimum_coverage: float, gamma: float) -> tuple[np.ndarray, dict]:
    """Bertsimas-Sim budgeted-risk MILP with Gamma uncertain coefficients."""
    utility, risk, minimum = _validate(utility, risk, budget, minimum_coverage)
    deviation = np.asarray(deviation, float)
    if len(deviation) != len(risk) or np.any(deviation < 0) or gamma < 0:
        raise ValueError("Risk deviations and gamma are invalid")
    n, variables = len(risk), 2 * len(risk) + 1
    objective = np.r_[-utility, 0.0, np.zeros(n)]
    # risk*z + Gamma*theta + sum(p) <= B; sum(z) >= minimum
    # Build the n+2 constraints with O(n) storage, including for full test sets.
    indices = np.arange(n)
    row_index = np.r_[np.zeros(n, int), np.zeros(n, int),
                      np.array([0]), np.ones(n, int), 2 + indices,
                      2 + indices, 2 + indices]
    col_index = np.r_[indices, n + 1 + indices, np.array([n]),
                      indices, indices, np.full(n, n), n + 1 + indices]
    values = np.r_[risk, np.ones(n), np.array([gamma]),
                   np.ones(n), deviation, -np.ones(n), -np.ones(n)]
    matrix = coo_matrix((values, (row_index, col_index)), shape=(n + 2, variables)).tocsr()
    lower = np.r_[-np.inf, minimum, np.full(n, -np.inf)]
    upper = np.r_[budget, np.inf, np.zeros(n)]
    started = time.perf_counter()
    result = milp(objective, integrality=np.r_[np.ones(n), np.zeros(n + 1)],
                  bounds=Bounds(np.zeros(variables), np.r_[np.ones(n), np.inf, np.full(n, np.inf)]),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={"presolve": True})
    if not result.success or result.x is None:
        raise ValueError(f"Robust risk-budget MILP is infeasible: {result.message}")
    selected = result.x[:n] > 0.5
    worst_extra = float(np.sort(deviation[selected])[::-1][:int(math.floor(gamma))].sum())
    fraction = gamma - math.floor(gamma)
    ordered = np.sort(deviation[selected])[::-1]
    if fraction and len(ordered) > math.floor(gamma):
        worst_extra += fraction * ordered[math.floor(gamma)]
    return selected, {"method": "robust_milp", "gamma": gamma,
                      "solve_time_s": time.perf_counter() - started,
                      "solver_status": int(result.status), "solver_message": result.message,
                      "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
                      "nominal_risk": float(risk[selected].sum()),
                      "robust_risk_bound": float(risk[selected].sum() + worst_extra),
                      "utility": float(utility[selected].sum())}


def risk_ranked(utility: np.ndarray, risk: np.ndarray, budget: float,
                minimum_coverage: float) -> tuple[np.ndarray, dict]:
    utility, risk, minimum = _validate(utility, risk, budget, minimum_coverage)
    order = np.lexsort((-utility, risk))
    selected = np.zeros(len(risk), dtype=bool)
    used = 0.0
    for index in order:
        if used + risk[index] <= budget:
            selected[index] = True
            used += risk[index]
    if selected.sum() < minimum:
        raise ValueError("Risk-ranked selection cannot meet coverage within budget")
    return selected, {"method": "risk_ranked", "risk_used": used,
                      "utility": float(utility[selected].sum())}


def greedy_utility_ratio(utility: np.ndarray, risk: np.ndarray, budget: float,
                         minimum_coverage: float) -> tuple[np.ndarray, dict]:
    utility, risk, minimum = _validate(utility, risk, budget, minimum_coverage)
    ratio = utility / np.maximum(risk, 1e-12)
    order = np.lexsort((risk, -ratio))
    selected, used = np.zeros(len(risk), bool), 0.0
    for index in order:
        if used + risk[index] <= budget:
            selected[index], used = True, used + risk[index]
    if selected.sum() < minimum:
        raise ValueError("Greedy selection cannot meet coverage within budget")
    return selected, {"method": "greedy_utility_risk", "risk_used": used,
                      "utility": float(utility[selected].sum())}


def lagrangian_relaxation(utility: np.ndarray, risk: np.ndarray, budget: float,
                          minimum_coverage: float, iterations: int = 80) -> tuple[np.ndarray, dict]:
    utility, risk, minimum = _validate(utility, risk, budget, minimum_coverage)
    low, high = 0.0, float(np.max(utility / np.maximum(risk, 1e-12)) * 2)
    best = None
    for _ in range(iterations):
        penalty = (low + high) / 2
        score = utility - penalty * risk
        order = np.argsort(-score, kind="stable")
        selected = score > 0
        if selected.sum() < minimum:
            selected[:] = False
            selected[order[:minimum]] = True
        used = risk[selected].sum()
        if used <= budget:
            best, high = selected.copy(), penalty
        else:
            low = penalty
    if best is None:
        raise ValueError("Lagrangian relaxation cannot meet coverage within budget")
    return best, {"method": "lagrangian", "lambda": high,
                  "risk_used": float(risk[best].sum()), "utility": float(utility[best].sum())}

