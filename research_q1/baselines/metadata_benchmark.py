"""Real-data development benchmark using legitimately downloaded PRC tables.

This is a metadata-only benchmark. It supplies five-plus established prediction
baselines and route-cluster confidence intervals, but it is not PI-MCS evidence
because trajectory, weather and physics fields are absent. October 2025 was
previously inspected by the project and is explicitly exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from research_q1.baselines.intervals import standard_split_conformal
from research_q1.baselines.predictors import BaselineSuite, PredictionBundle
from research_q1.novel.conformal import cqr_interval
from research_q1.stats.inference import cluster_bootstrap_difference, cluster_bootstrap_mean
from research_q1.validation.metrics import interval_metrics, regression_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_period(raw_dir: Path, split: str) -> pd.DataFrame:
    flights = pd.read_parquet(raw_dir / f"flightlist_{split}.parquet")
    labels = pd.read_parquet(raw_dir / f"fuel_{split}.parquet")
    frame = labels.merge(flights, on="flight_id", validate="many_to_one")
    frame["duration_min"] = (frame.end - frame.start).dt.total_seconds() / 60
    frame["flight_duration_min"] = (frame.landed - frame.takeoff).dt.total_seconds() / 60
    frame["elapsed_from_takeoff_min"] = (frame.start - frame.takeoff).dt.total_seconds() / 60
    frame["remaining_to_landing_min"] = (frame.landed - frame.end).dt.total_seconds() / 60
    frame["start_hour_sin"] = np.sin(2 * np.pi * frame.start.dt.hour / 24)
    frame["start_hour_cos"] = np.cos(2 * np.pi * frame.start.dt.hour / 24)
    frame["route_id"] = frame.origin_icao.astype(str) + "-" + frame.destination_icao.astype(str)
    frame["published_split"] = split
    return frame


NUMERIC = ["duration_min", "flight_duration_min", "elapsed_from_takeoff_min",
           "remaining_to_landing_min", "start_hour_sin", "start_hour_cos"]


def design(train: pd.DataFrame, evaluation: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_type = pd.get_dummies(train.aircraft_type, prefix="type", dtype=float)
    eval_type = pd.get_dummies(evaluation.aircraft_type, prefix="type", dtype=float).reindex(
        columns=train_type.columns, fill_value=0)
    a = pd.concat([train[NUMERIC].reset_index(drop=True), train_type.reset_index(drop=True)], axis=1)
    b = pd.concat([evaluation[NUMERIC].reset_index(drop=True), eval_type.reset_index(drop=True)], axis=1)
    return a.to_numpy(float), b.to_numpy(float), a.columns.tolist()


def type_rate(train: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    rates = train.fuel_kg / train.duration_min
    mapping = rates.groupby(train.aircraft_type).median()
    rate = evaluation.aircraft_type.map(mapping).fillna(float(rates.median())).to_numpy()
    return np.maximum(0.0, rate * evaluation.duration_min.to_numpy())


def run(raw_dir: Path, *, seed: int = 20251022, bootstrap_repetitions: int = 2000,
        max_iterations: int = 300) -> dict:
    train, calibration, test = (load_period(raw_dir, name) for name in ("train", "rank", "final"))
    if set(train.flight_id) & set(calibration.flight_id) or set(train.flight_id) & set(test.flight_id) or set(calibration.flight_id) & set(test.flight_id):
        raise ValueError("Published flight identifiers overlap")
    evaluation = pd.concat([calibration, test], ignore_index=True)
    x_train, x_evaluation, columns = design(train, evaluation)
    suite = BaselineSuite(seed=seed, max_iterations=max_iterations)
    bundles = suite.fit_predict(x_train, train.fuel_kg.to_numpy(), x_evaluation)
    bundles["aircraft_type_median_rate"] = PredictionBundle(type_rate(train, evaluation))
    n_cal = len(calibration)
    truth_cal, truth_test = calibration.fuel_kg.to_numpy(), test.fuel_kg.to_numpy()
    route = test.route_id.to_numpy()
    result = {
        "status": "REAL_DATA_EXPLORATORY_METADATA_BENCHMARK",
        "evidence_boundary": "October 2025 labels were previously inspected; no confirmatory or PI-MCS claim",
        "protocol": "train April-August 2025; calibrate September 2025; evaluate October 2025",
        "seed": seed, "feature_columns": columns,
        "counts": {"train_intervals": len(train), "calibration_intervals": len(calibration),
                   "test_intervals": len(test), "train_flights": int(train.flight_id.nunique()),
                   "calibration_flights": int(calibration.flight_id.nunique()),
                   "test_flights": int(test.flight_id.nunique()), "test_routes": int(test.route_id.nunique())},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "source_sha256": {path.name: sha256(path) for path in sorted(raw_dir.glob("flightlist_*.parquet")) + sorted(raw_dir.glob("fuel_*.parquet"))},
        "models": {}, "paired_against_hist_gradient_boosting": {},
    }
    reference_error = None
    for name, bundle in bundles.items():
        cal_prediction, test_prediction = bundle.point[:n_cal], bundle.point[n_cal:]
        absolute_error = np.abs(test_prediction - truth_test)
        lo, hi, conformal_audit = standard_split_conformal(
            truth_cal, cal_prediction, test_prediction, alpha=.1)
        interval = interval_metrics(truth_test, lo, hi, .1)
        model_result = {
            "regression": regression_metrics(truth_test, test_prediction),
            "mae_route_cluster_ci": cluster_bootstrap_mean(
                absolute_error, route, repetitions=bootstrap_repetitions, seed=seed),
            "split_conformal_90": {**interval, **conformal_audit},
        }
        if name == "quantile_regression" and bundle.lower is not None:
            cqr_lo, cqr_hi, audit = cqr_interval(
                bundle.lower[n_cal:], bundle.upper[n_cal:], truth_cal,
                bundle.lower[:n_cal], bundle.upper[:n_cal], alpha=.1)
            model_result["cqr_90"] = {**interval_metrics(truth_test, cqr_lo, cqr_hi, .1), **audit}
        result["models"][name] = model_result
        if name == "hist_gradient_boosting":
            reference_error = absolute_error
    if reference_error is None:
        raise RuntimeError("Reference baseline was not fit")
    for name, bundle in bundles.items():
        if name == "hist_gradient_boosting":
            continue
        error = np.abs(bundle.point[n_cal:] - truth_test)
        result["paired_against_hist_gradient_boosting"][name] = cluster_bootstrap_difference(
            reference_error, error, route, repetitions=bootstrap_repetitions, seed=seed)
    unseen = ~test.aircraft_type.isin(set(train.aircraft_type))
    result["unseen_aircraft_types"] = sorted(test.loc[unseen, "aircraft_type"].unique().tolist())
    result["unseen_aircraft_intervals"] = int(unseen.sum())
    if unseen.any():
        result["unseen_aircraft_metrics"] = {
            name: regression_metrics(truth_test[unseen.to_numpy()], bundle.point[n_cal:][unseen.to_numpy()])
            for name, bundle in bundles.items()
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("research_q1/outputs/metadata-baselines.json"))
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--max-iterations", type=int, default=300)
    args = parser.parse_args()
    result = run(args.raw_dir, bootstrap_repetitions=args.bootstrap_repetitions,
                 max_iterations=args.max_iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, values in result["models"].items():
        metric = values["regression"]
        print(name, "MAE", round(metric["mae_kg"], 3), "RMSE", round(metric["rmse_kg"], 3))


if __name__ == "__main__":
    main()

