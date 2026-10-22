"""Advanced coverage-constrained analysis and replay-dashboard generator.

This module operates on the committed, pre-scored September benchmark rows.
It adds an exact mixed-integer selection baseline, decision explanations and
an offline streaming replay. It does not claim that replayed records are live
aircraft data, and it never uses fuel labels to select intervals.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from .eligibility import DURATION_LABELS, describe, duration_groups, select

ROW_COLUMNS = (
    "aircraft_type",
    "duration_min",
    "risk_score_kg",
    "predicted_fuel_kg",
    "reference_fuel_kg",
    "adsb_position_count",
    "stable_tie",
)


def load_demo_payload(path: Path) -> dict:
    """Read the JSON payload embedded in the generated offline dashboard."""
    source = path.read_text()
    marker = "const DATA="
    start = source.find(marker)
    if start < 0:
        raise ValueError(f"No embedded DATA payload in {path}")
    payload, _ = json.JSONDecoder().raw_decode(source[start + len(marker):])
    if not isinstance(payload, dict) or "rows" not in payload:
        raise ValueError("Dashboard payload must contain rows")
    return payload


def rows_frame(payload: dict) -> pd.DataFrame:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dashboard payload rows must be a nonempty list")
    frame = pd.DataFrame(rows, columns=ROW_COLUMNS)
    numeric = [name for name in ROW_COLUMNS if name != "aircraft_type"]
    if frame[numeric].isna().any().any():
        raise ValueError("Advanced analysis requires finite committed demo rows")
    frame["flight_id"] = [f"replay-{i:05d}" for i in range(len(frame))]
    frame["fuel_kg"] = frame.reference_fuel_kg.astype(float)
    return frame


def _constraint_members(
    types: np.ndarray,
    duration: np.ndarray,
    min_type_n: int,
) -> list[tuple[str, np.ndarray]]:
    groups = duration_groups(duration)
    constraints = [
        (f"duration:{name}", groups == name)
        for name in DURATION_LABELS
        if np.any(groups == name)
    ]
    names, counts = np.unique(types.astype(str), return_counts=True)
    constraints.extend(
        (f"aircraft:{name}", types.astype(str) == name)
        for name, count in zip(names, counts)
        if count >= min_type_n
    )
    return constraints


def constrained_select(
    scores: np.ndarray,
    types: np.ndarray,
    duration: np.ndarray,
    ties: np.ndarray,
    target: float,
    floor: float,
    min_type_n: int = 50,
    *,
    time_limit_s: float = 60,
) -> tuple[np.ndarray, dict]:
    """Minimize total risk subject to overall and subgroup coverage constraints.

    A large per-row count penalty first minimizes the number of selected rows;
    predicted risk then breaks ties among equal-size feasible selections. This
    allows floors to exceed the nominal budget while avoiding unnecessary rows.
    """
    scores, types, duration, ties = (np.asarray(x) for x in (scores, types, duration, ties))
    n = len(scores)
    if not n or any(len(x) != n for x in (types, duration, ties)):
        raise ValueError("Nonempty, equally sized arrays required")
    if not np.isfinite(scores).all() or not np.isfinite(ties).all():
        raise ValueError("Scores and ties must be finite")
    if not (0 < target <= 1 and 0 <= floor <= 1 and min_type_n >= 1):
        raise ValueError("Invalid target, floor or minimum aircraft group size")

    span = float(np.ptp(scores))
    normalized = np.zeros(n) if span == 0 else (scores - float(scores.min())) / span
    count_penalty = float(n + 1)
    objective = count_penalty + normalized + ties.astype(float) * 1e-9

    members = _constraint_members(types, duration, min_type_n)
    rows = [np.ones(n, dtype=float)]
    lower = [math.ceil(target * n)]
    names = ["overall"]
    for name, mask in members:
        rows.append(mask.astype(float))
        lower.append(math.ceil(floor * int(mask.sum())))
        names.append(name)
    matrix = csr_matrix(np.vstack(rows))
    result = milp(
        c=objective,
        integrality=np.ones(n, dtype=np.uint8),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(matrix, np.asarray(lower, dtype=float), np.full(len(rows), np.inf)),
        options={"time_limit": float(time_limit_s)},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Constrained selector failed: {result.message}")
    selected = result.x >= 0.5
    achieved = matrix @ selected.astype(float)
    details = {
        "method": "mixed_integer_minimum_risk",
        "solver": "scipy.optimize.milp (HiGHS)",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "target_count": int(lower[0]),
        "selected_count": int(selected.sum()),
        "floor_forced_over_budget": bool(selected.sum() > lower[0]),
        "objective_predicted_risk_sum_kg": float(scores[selected].sum()),
        "constraints": {
            name: {"required": int(required), "achieved": int(actual)}
            for name, required, actual in zip(names, lower, achieved)
        },
    }
    return selected, details


def decision_reasons(
    selected: np.ndarray,
    risk_only: np.ndarray,
    types: np.ndarray,
    duration: np.ndarray,
    min_type_n: int,
) -> list[str]:
    """Create honest, non-causal explanations for selection decisions."""
    selected, risk_only = np.asarray(selected, dtype=bool), np.asarray(risk_only, dtype=bool)
    types = np.asarray(types).astype(str)
    durations = duration_groups(np.asarray(duration, dtype=float))
    names, counts = np.unique(types, return_counts=True)
    protected = {name for name, count in zip(names, counts) if count >= min_type_n}
    reasons = []
    for keep, low_risk, aircraft, duration_name in zip(selected, risk_only, types, durations):
        if not keep:
            reasons.append("not selected at this coverage target")
        elif low_risk:
            reasons.append("selected by low predicted-error rank")
        else:
            memberships = [f"duration coverage ({duration_name})"]
            if aircraft in protected:
                memberships.append(f"aircraft coverage ({aircraft})")
            reasons.append("coverage-constrained optimum; " + " and ".join(memberships))
    return reasons


def _policy_summary(
    frame: pd.DataFrame,
    mask: np.ndarray,
    predicted: np.ndarray,
    risk: np.ndarray,
    policy: dict,
) -> dict:
    summary = describe(frame, predicted, risk, mask, policy)
    duration_coverages = [group["coverage"] for group in summary["groups"]["duration"].values()]
    protected = [
        group["coverage"]
        for group in summary["groups"]["aircraft_type"].values()
        if group["n"] >= int(policy.get("min_type_n", 50))
    ]
    summary["mean_predicted_risk_kg"] = float(np.mean(risk[mask]))
    summary["duration_coverage_gap"] = float(max(duration_coverages) - min(duration_coverages))
    summary["protected_aircraft_coverage_gap"] = (
        float(max(protected) - min(protected)) if protected else None
    )
    return summary


def analyze(
    payload: dict,
    *,
    target: float = 0.75,
    floor: float = 0.5,
    min_type_n: int = 50,
) -> tuple[dict, dict]:
    frame = rows_frame(payload)
    scores = frame.risk_score_kg.to_numpy(dtype=float)
    types = frame.aircraft_type.to_numpy(dtype=str)
    duration = frame.duration_min.to_numpy(dtype=float)
    ties = frame.stable_tie.to_numpy(dtype=float)
    predicted = frame.predicted_fuel_kg.to_numpy(dtype=float)

    risk_mask, risk_details = select(scores, types, duration, ties, target, 0, min_type_n)
    greedy_mask, greedy_details = select(scores, types, duration, ties, target, floor, min_type_n)
    optimized_mask, optimized_details = constrained_select(
        scores, types, duration, ties, target, floor, min_type_n
    )
    policies = {
        "risk_only": _policy_summary(
            frame,
            risk_mask,
            predicted,
            scores,
            {"method": "risk_rank", "floor": 0, "min_type_n": min_type_n, **risk_details},
        ),
        "greedy_floor": _policy_summary(
            frame,
            greedy_mask,
            predicted,
            scores,
            {"method": "duration_then_aircraft_greedy", "floor": floor,
             "min_type_n": min_type_n, **greedy_details},
        ),
        "optimized_floor": _policy_summary(
            frame,
            optimized_mask,
            predicted,
            scores,
            {"floor": floor, "min_type_n": min_type_n, **optimized_details},
        ),
    }

    pareto = []
    for coverage in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0):
        baseline, _ = select(scores, types, duration, ties, coverage, 0, min_type_n)
        constrained, detail = constrained_select(
            scores, types, duration, ties, coverage, floor, min_type_n
        )
        for name, mask in (("risk_only", baseline), ("optimized_floor", constrained)):
            truth = frame.fuel_kg.to_numpy(dtype=float)
            pareto.append({
                "policy": name,
                "target": coverage,
                "actual_coverage": float(mask.mean()),
                "accepted_mae_kg": float(np.abs(predicted[mask] - truth[mask]).mean()),
                "labeled_fuel_share": float(truth[mask].sum() / truth.sum()),
                "mean_predicted_risk_kg": float(scores[mask].mean()),
                "floor_forced_over_budget": bool(detail["floor_forced_over_budget"])
                if name == "optimized_floor" else False,
            })

    # Generate explanations once as an integrity check. The browser derives the
    # same wording from compact selection bitsets and the source demo rows.
    decision_reasons(optimized_mask, risk_mask, types, duration, min_type_n)
    report = {
        "status": "Advanced exploratory extension on committed September benchmark rows",
        "research_claim": (
            "Exact coverage-constrained selection is compared with risk-only and the existing greedy rule; "
            "labels are used only after selection for evaluation."
        ),
        "limitations": [
            "This is a historical replay, not live aircraft telemetry.",
            "Optimization improves the selection rule but does not make the fuel model physically validated.",
            "A later untouched dataset is still required for confirmatory publication claims.",
        ],
        "target": target,
        "floor": floor,
        "min_aircraft_group_n": min_type_n,
        "rows": len(frame),
        "policies": policies,
        "pareto": pareto,
    }
    dashboard_payload = {
        "period": payload.get("period", "Committed validation benchmark"),
        "source_demo": "index.html",
        "row_schema": list(ROW_COLUMNS),
        "selection_bits": {
            "risk": "".join("1" if value else "0" for value in risk_mask),
            "greedy": "".join("1" if value else "0" for value in greedy_mask),
            "optimized": "".join("1" if value else "0" for value in optimized_mask),
        },
        "report": report,
    }
    return report, dashboard_payload


def render(template: str, payload: dict) -> str:
    marker = "/*__ADVANCED_DATA__*/"
    if template.count(marker) != 1:
        raise ValueError("Expected exactly one advanced-dashboard data marker")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return template.replace(marker, "const DATA=" + data.replace("<", "\\u003c") + ";")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-demo", type=Path, default=Path("demo/index.html"))
    parser.add_argument("--template", type=Path, default=Path("demo/advanced-template.html"))
    parser.add_argument("--dashboard", type=Path, default=Path("demo/advanced.html"))
    parser.add_argument("--report", type=Path, default=Path("reports/advanced-results.json"))
    parser.add_argument("--target", type=float, default=0.75)
    parser.add_argument("--floor", type=float, default=0.5)
    parser.add_argument("--min-type-n", type=int, default=50)
    args = parser.parse_args()
    payload = load_demo_payload(args.source_demo)
    report, dashboard = analyze(
        payload, target=args.target, floor=args.floor, min_type_n=args.min_type_n
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.dashboard.parent.mkdir(parents=True, exist_ok=True)
    args.dashboard.write_text(render(args.template.read_text(), dashboard))
    print(f"Wrote {args.report} and {args.dashboard}")


if __name__ == "__main__":
    main()
