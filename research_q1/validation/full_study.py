"""End-to-end PI-MCS study runner for an approved contract-compliant table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_q1.baselines.intervals import standard_split_conformal
from research_q1.baselines.predictors import BaselineSuite
from research_q1.novel.calibration import RiskCalibrator
from research_q1.novel.conformal import HierarchicalMondrian, cqr_interval
from research_q1.novel.physics import PhysicsPrior
from research_q1.novel.residual import HybridResidualModel
from research_q1.novel.selection import (exact_risk_budget, greedy_utility_ratio,
                                          lagrangian_relaxation, risk_ranked,
                                          robust_risk_budget)
from research_q1.stats.inference import cluster_bootstrap_mean
from research_q1.validation.data_contract import (frame_fingerprint, read_table,
                                                    validate_frame)
from research_q1.validation.metrics import (calibration_metrics,
                                             conditional_interval_metrics,
                                             interval_metrics,
                                             regression_metrics,
                                             selective_metrics)
from research_q1.validation.shifts import domain_classifier_auc
from research_q1.validation.splits import Split, grouped_holdout, route_holdout, temporal_split


def _split(frame: pd.DataFrame, kind: str, config: dict) -> Split:
    if kind == "temporal":
        bounds = config["temporal"]
        return temporal_split(frame, bounds["train_before"], bounds["calibration_before"], bounds["test_before"])
    if kind == "route":
        return route_holdout(frame, seed=config["seed"])
    column = {"aircraft": "aircraft_type", "aircraft_family": "aircraft_family",
              "seasonal": "season"}.get(kind)
    if column is None:
        raise ValueError("split kind must be temporal, route, aircraft, aircraft_family or seasonal")
    return grouped_holdout(frame, column, seed=config["seed"])


def _arrays(frame: pd.DataFrame, mask: np.ndarray, features: list[str]):
    part = frame.loc[mask].reset_index(drop=True)
    return part, part[features].to_numpy(float), part.reference_fuel_kg.to_numpy(float)


def _selection_results(y, prediction, calibrated_risk, config):
    n = len(y)
    utility = np.log1p(np.maximum(prediction, 0.0))
    minimum = int(np.ceil(config["minimum_coverage"] * n))
    minimum_risk = float(np.sort(calibrated_risk)[:minimum].sum())
    budget = config["risk_budget_multiplier_over_minimum_feasible"] * minimum_risk
    deviation = config["robust_deviation_fraction"] * calibrated_risk
    methods = {
        "risk_ranked": lambda: risk_ranked(utility, calibrated_risk, budget, config["minimum_coverage"]),
        "greedy": lambda: greedy_utility_ratio(utility, calibrated_risk, budget, config["minimum_coverage"]),
        "lagrangian": lambda: lagrangian_relaxation(utility, calibrated_risk, budget, config["minimum_coverage"]),
        "exact_milp": lambda: exact_risk_budget(utility, calibrated_risk, budget, config["minimum_coverage"]),
        "robust_milp": lambda: robust_risk_budget(
            utility, calibrated_risk, deviation, budget, config["minimum_coverage"], config["robust_gamma"]),
    }
    output = {"risk_budget": budget, "minimum_feasible_nominal_risk": minimum_risk,
              "utility_definition": "sum log(1+predicted fuel kg); labels are not selection inputs",
              "methods": {}}
    for name, function in methods.items():
        selected, audit = function()
        output["methods"][name] = {"optimizer": audit,
                                   "observed": selective_metrics(y, prediction, selected,
                                                                 calibrated_risk, utility, budget)}
    return output


def run(frame: pd.DataFrame, config: dict, split_kind: str) -> dict:
    manifest = validate_frame(frame)
    split = _split(frame, split_kind, config)
    features = config["numeric_features"]
    train, x_train, y_train = _arrays(frame, split.train, features)
    calibration, x_cal, y_cal = _arrays(frame, split.calibration, features)
    test, x_test, y_test = _arrays(frame, split.test, features)
    if min(len(train), len(calibration), len(test)) == 0:
        raise ValueError("Selected grouped split has an empty train/calibration/test partition")

    physics = PhysicsPrior().fit(train)
    p_train, p_cal, p_test = physics.predict(train), physics.predict(calibration), physics.predict(test)
    hybrid = HybridResidualModel(seed=config["seed"]).fit(
        x_train, y_train, p_train, train.flight_id.to_numpy())
    cal_prediction, cal_sigma = hybrid.predict(x_cal, p_cal)
    test_prediction, test_sigma = hybrid.predict(x_test, p_test)

    normalized_scores = np.abs(y_cal - cal_prediction) / np.maximum(cal_sigma, 1e-9)
    mondrian = HierarchicalMondrian(config["alpha"], config["mondrian_minimum_n"]).fit(
        normalized_scores, calibration.aircraft_type, calibration.flight_phase)
    pi_lower, pi_upper, pi_audit = mondrian.interval(
        test_prediction, test_sigma, test.aircraft_type, test.flight_phase, test.tank_capacity_kg)

    evaluation_x = np.vstack([x_cal, x_test])
    suite = BaselineSuite(config["seed"], config["model_iterations"])
    baselines = suite.fit_predict(x_train, y_train, evaluation_x,
                                  physics_test=np.r_[p_cal, p_test])
    n_cal = len(calibration)
    baseline_results = {}
    for name, bundle in baselines.items():
        cal_p, test_p = bundle.point[:n_cal], bundle.point[n_cal:]
        lower, upper, audit = standard_split_conformal(
            y_cal, cal_p, test_p, alpha=config["alpha"],
            tank_capacity_kg=test.tank_capacity_kg.to_numpy())
        baseline_results[name] = {
            "regression": regression_metrics(y_test, test_p),
            "interval": {**interval_metrics(y_test, lower, upper, config["alpha"]), **audit},
        }
        if name == "quantile_regression":
            lo, hi, cqr_audit = cqr_interval(
                bundle.lower[n_cal:], bundle.upper[n_cal:], y_cal,
                bundle.lower[:n_cal], bundle.upper[:n_cal], alpha=config["alpha"],
                tank_capacity_kg=test.tank_capacity_kg.to_numpy())
            baseline_results[name]["cqr"] = {
                **interval_metrics(y_test, lo, hi, config["alpha"]), **cqr_audit}

    threshold = config["risk_event_absolute_error_kg"]
    cal_event = (np.abs(y_cal - cal_prediction) > threshold).astype(int)
    risk_calibration, test_risks = {}, {}
    for method in ("isotonic", "platt", "venn_abers"):
        calibrator = RiskCalibrator(method, config["seed"]).fit(cal_sigma, cal_event)
        probability_cal, probability_test = calibrator.predict(cal_sigma), calibrator.predict(test_sigma)
        risk_calibration[method] = calibration_metrics(probability_cal, cal_event)
        test_risks[method] = probability_test

    test_route = test.origin_icao.astype(str) + "-" + test.destination_icao.astype(str)
    pi_interval = interval_metrics(y_test, pi_lower, pi_upper, config["alpha"])
    result = {
        "status": "FULL_STUDY_RESULT",
        "split": split_kind, "config": config, "dataset": manifest,
        "dataset_sha256": frame_fingerprint(frame),
        "counts": {"train": len(train), "calibration": len(calibration), "test": len(test),
                   "excluded": int(split.excluded.sum())},
        "pi_mcs": {
            "regression": regression_metrics(y_test, test_prediction),
            "interval": {**pi_interval, **pi_audit},
            "conditional_aircraft": conditional_interval_metrics(
                y_test, pi_lower, pi_upper, test.aircraft_type, config["alpha"]),
            "conditional_phase": conditional_interval_metrics(
                y_test, pi_lower, pi_upper, test.flight_phase, config["alpha"]),
            "mae_route_cluster_ci": cluster_bootstrap_mean(
                np.abs(y_test - test_prediction), test_route, repetitions=config["bootstrap_repetitions"],
                seed=config["seed"]),
            "physics_fit_audit": physics.fit_audit_, "residual_fit_audit": hybrid.oof_audit_,
        },
        "baselines": baseline_results,
        "risk_calibration": risk_calibration,
        "selection": _selection_results(y_test, test_prediction, test_risks["isotonic"], config),
        "shift": domain_classifier_auc(
            x_train, x_test, train.flight_id.to_numpy(), test.flight_id.to_numpy(),
            seed=config["seed"]),
        "guarantee_boundary": "Empirical grouped evaluation; conformal guarantees require the assumptions documented in paper/claims.md",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("research_q1/configs/full_study.json"))
    parser.add_argument("--split", choices=["temporal", "route", "aircraft", "aircraft_family", "seasonal"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = read_table(args.data, access_log=Path("research_q1/external/future-confirmation-access.log"))
    result = run(frame, json.loads(args.config.read_text()), args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

