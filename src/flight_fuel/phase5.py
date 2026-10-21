"""Audit published Phase 4 aggregates and build an offline results demo.

This command does not retrain a model or read row-level labels. It checks
internal consistency and computes explicitly descriptive sensitivity results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DURATION_ORDER = ("<5 min", "5–15 min", "15–60 min")
PERIODS = ("validation", "final")


def _close(actual: float, expected: float, description: str, tolerance: float = 1e-7) -> None:
    if abs(actual - expected) > tolerance:
        raise ValueError(f"{description}: {actual} differs from {expected}")


def duration_sensitivity(period: dict) -> dict:
    """Recover fallback MAE per duration from disjoint label-lattice strata."""
    source = period["label_granularity"]["duration_bin"]
    if set(source) != set(DURATION_ORDER):
        raise ValueError("Missing or extra duration groups")
    segments = {}
    for group in DURATION_ORDER:
        values = source[group]
        n, share = values["n"], values["lattice_fraction"]
        if n <= 0 or not 0 <= share <= 1:
            raise ValueError("Invalid duration group")
        mae = share * values["lattice_mae_kg"] + (1 - share) * values["other_mae_kg"]
        segments[group] = {"n": n, "fallback_mae_kg": mae}
    total = sum(item["n"] for item in segments.values())
    error = sum(item["n"] * item["fallback_mae_kg"] for item in segments.values())
    _close(error / total, period["models"]["fallback"]["mae_kg"], "duration reconstruction")
    if total != period["models"]["fallback"]["n_intervals"]:
        raise ValueError("Duration counts differ from model count")
    for item in segments.values():
        item["fraction"] = item["n"] / total
    longer_n = total - segments["<5 min"]["n"]
    longer_error = error - segments["<5 min"]["n"] * segments["<5 min"]["fallback_mae_kg"]
    return {"duration_groups": segments, "at_least_5_min": {
        "n": longer_n, "fallback_mae_kg": longer_error / longer_n}}


def check_selection(period: dict, name: str) -> None:
    total = period["models"]["fallback"]["n_intervals"]
    for selector, points in period["selectors"].items():
        previous = -1
        for target in ("0.5", "0.75", "0.9", "1.0"):
            point = points[target]
            count = point["accepted_intervals"]
            if count < previous or count > total:
                raise ValueError(f"Nonmonotone/invalid acceptance: {name} {selector} {target}")
            previous = count
            _close(count / total, point["accepted_fraction"], "acceptance fraction")
            for group, members in point["acceptance_by_group"].items():
                if sum(item["n"] for item in members.values()) != total:
                    raise ValueError(f"Incomplete {group} partition")
                weighted = sum(item["n"] * item["accepted_fraction"] for item in members.values())
                _close(weighted, count, f"{group} accepted count")
        full = points["1.0"]
        if full["accepted_intervals"] != total:
            raise ValueError("100% policy does not accept the whole period")
        for metric in ("mae_kg", "rmse_kg", "flight_macro_mae_kg"):
            _close(full["accepted_score"][metric], period["models"]["fallback"][metric],
                   f"100% {metric}")


def audit(report: dict) -> dict:
    by_period = {}
    for name in PERIODS:
        period = report["periods"][name]
        check_selection(period, name)
        by_period[name] = duration_sensitivity(period)
        primary = period["selectors"]["learned_risk"]["0.75"]
        long = primary["acceptance_by_group"]["duration_bin"]["15–60 min"]
        by_period[name]["risk_75_long_interval_acceptance"] = {
            "n": long["n"], "fraction": long["accepted_fraction"],
            "below_10_percent": long["accepted_fraction"] < 0.1}

    val = by_period["validation"]["duration_groups"]
    final = by_period["final"]["duration_groups"]
    standardized_final = sum(val[g]["fraction"] * final[g]["fallback_mae_kg"] for g in DURATION_ORDER)
    standardized_validation = sum(final[g]["fraction"] * val[g]["fallback_mae_kg"] for g in DURATION_ORDER)
    result = {"basis": "Reconstructed from published disjoint aggregates; no raw labels, predictions or refitting",
              "periods": by_period,
              "mix_standardization": {
                  "final_mae_with_validation_duration_mix_kg": standardized_final,
                  "validation_mae_with_final_duration_mix_kg": standardized_validation,
                  "description": "Descriptive direct standardization of fallback absolute error across three duration bins; no causal or inferential claim"},
              "checks": "Duration reconstruction, selector partitions, monotonic acceptance, and 100% identity passed"}
    return result


def dashboard_payload(report: dict, robustness: dict) -> dict:
    return {"protocol": {"primary_coverage": report["protocol"]["primary_coverage"],
                         "fallback": report["protocol"]["fallback"]},
            "periods": {name: {"counts": report["block_counts"][name],
                               "models": report["periods"][name]["models"],
                               "selectors": report["periods"][name]["selectors"],
                               "sensitivity": robustness["periods"][name]}
                        for name in PERIODS},
            "outage": report["outage_experiment"],
            "mix_standardization": robustness["mix_standardization"]}


def render(template: str, payload: dict) -> str:
    marker = "/*__PHASE5_DATA__*/"
    if template.count(marker) != 1:
        raise ValueError("Dashboard template must contain one data marker")
    # Escaping < prevents an embedded string from closing the script element.
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return template.replace(marker, "const DATA = " + data.replace("<", "\\u003c") + ";")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("reports/phase-4-results.json"))
    parser.add_argument("--audit-output", type=Path, default=Path("reports/phase-5-robustness.json"))
    parser.add_argument("--template", type=Path, default=Path("demo/template.html"))
    parser.add_argument("--demo-output", type=Path, default=Path("demo/index.html"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    result = audit(report)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.demo_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    args.demo_output.write_text(render(args.template.read_text(), dashboard_payload(report, result)))
    print(f"Verified {', '.join(PERIODS)}; wrote {args.audit_output} and {args.demo_output}")


if __name__ == "__main__":
    main()
