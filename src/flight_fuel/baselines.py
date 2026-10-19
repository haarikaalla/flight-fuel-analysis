"""Fixed Phase 3 baselines on chronological train and ranking periods.

The final period is intentionally inaccessible to this command. These are
retrospective interval estimates, not operational fuel-flow measurements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURE_COLS

NUMERIC_METADATA = ["duration_min"]
TRAJECTORY_PREDICTORS = [
    "adsb_altitude_median_raw", "adsb_altitude_p10_raw", "adsb_altitude_p90_raw",
    "adsb_speed_median_raw", "adsb_speed_p10_raw", "adsb_speed_p90_raw",
    "adsb_vertical_rate_abs_median_raw", "adsb_vertical_rate_abs_p90_raw",
    "adsb_position_count", "adsb_speed_count",
]


def load_split(raw_dir: Path, split: str, feature_dir: Path | None = None) -> pd.DataFrame:
    if split not in ("train", "rank"):
        raise ValueError("Phase 3 only reads train and rank; final is locked")
    flights = pd.read_parquet(raw_dir / f"flightlist_{split}.parquet", columns=["flight_id", "aircraft_type"])
    intervals = pd.read_parquet(raw_dir / f"fuel_{split}.parquet", columns=["idx", "flight_id", "start", "end", "fuel_kg"])
    frame = intervals.merge(flights, on="flight_id", validate="many_to_one", indicator=True)
    if not frame._merge.eq("both").all() or len(frame) != len(intervals):
        raise ValueError(f"Missing or duplicated flight metadata in {split}")
    frame = frame.drop(columns="_merge")
    frame["duration_min"] = (frame.end - frame.start).dt.total_seconds() / 60
    if frame.duration_min.le(0).any() or frame.fuel_kg.isna().any():
        raise ValueError(f"Invalid labels or durations in {split}")
    if feature_dir is not None:
        features = pd.read_parquet(feature_dir / f"features_{split}.parquet")
        if set(features.columns) != {"idx", "flight_id", *FEATURE_COLS}:
            raise ValueError("Unexpected feature column; audit feature provenance")
        frame = frame.merge(features, on=["idx", "flight_id"], how="left", validate="one_to_one", indicator=True)
        if len(frame) != len(intervals) or not frame._merge.eq("both").all():
            raise ValueError(f"Missing/extra trajectory features in {split}")
        frame = frame.drop(columns="_merge")
    return frame


def type_rate_predictions(train: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    """Duration × median observed kg/min by type, global fallback for unknowns."""
    rates = train.fuel_kg / train.duration_min
    by_type = rates.groupby(train.aircraft_type).median().to_dict()
    global_rate = float(rates.median())
    return holdout.duration_min.to_numpy() * holdout.aircraft_type.map(by_type).fillna(global_rate).to_numpy()


def design(train: pd.DataFrame, holdout: pd.DataFrame, with_trajectory: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    numeric = NUMERIC_METADATA + (TRAJECTORY_PREDICTORS if with_trajectory else [])
    # Dummy columns are defined on train only. Unknown types become all-zero.
    a = pd.get_dummies(train.aircraft_type, prefix="type", dtype=float)
    b = pd.get_dummies(holdout.aircraft_type, prefix="type", dtype=float).reindex(columns=a.columns, fill_value=0)
    train_data = pd.concat([train[numeric].reset_index(drop=True), a.reset_index(drop=True)], axis=1)
    holdout_data = pd.concat([holdout[numeric].reset_index(drop=True), b.reset_index(drop=True)], axis=1)
    return (train_data.to_numpy(dtype=np.float32), holdout_data.to_numpy(dtype=np.float32), list(train_data.columns))


def score(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    truth = frame.fuel_kg.to_numpy()
    error = predicted - truth
    by_flight = pd.DataFrame({"flight_id": frame.flight_id.to_numpy(), "ae": abs(error), "se": error**2}).groupby("flight_id").mean()
    return {
        "n_intervals": len(frame), "n_flights": int(frame.flight_id.nunique()),
        "mae_kg": float(np.mean(abs(error))),
        "rmse_kg": float(np.sqrt(np.mean(error**2))),
        "flight_macro_mae_kg": float(by_flight.ae.mean()),
        "flight_macro_rmse_kg": float(np.sqrt(by_flight.se.mean())),
        "mean_signed_error_kg": float(error.mean()),
    }


def grouped_scores(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    grouped = frame[["aircraft_type", "duration_min", "flight_id", "fuel_kg"]].copy()
    grouped["prediction"] = predicted
    grouped["duration_bin"] = pd.cut(grouped.duration_min, [0, 5, 15, 60], labels=["<5 min", "5–15 min", "15–60 min"], include_lowest=True, right=False).astype(str)
    if "adsb_position_count" in frame:
        grouped["adsb_input"] = np.where(frame.adsb_position_count.eq(0), "zero ADS-B", "at least one ADS-B")
    result = {}
    for label in ("aircraft_type", "duration_bin", "adsb_input"):
        if label not in grouped:
            continue
        result[label] = {}
        for value, subset in grouped.groupby(label):
            p = subset.prediction.to_numpy()
            result[label][str(value)] = score(subset, p)
    return result


def paired_flight_bootstrap(frame: pd.DataFrame, first: np.ndarray, second: np.ndarray, repetitions: int = 1000) -> dict:
    """Second minus first, resampling flights and retaining their full intervals."""
    a = first - frame.fuel_kg.to_numpy()
    b = second - frame.fuel_kg.to_numpy()
    per_flight = pd.DataFrame({
        "flight_id": frame.flight_id.to_numpy(), "n": 1,
        "abs_a": abs(a), "abs_b": abs(b), "sq_a": a**2, "sq_b": b**2,
    }).groupby("flight_id", sort=True).sum()
    table = per_flight.to_numpy()
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(table), size=(repetitions, len(table)))
    samples = table[draws].sum(axis=1)
    delta_mae = (samples[:, 2] - samples[:, 1]) / samples[:, 0]
    delta_rmse = np.sqrt(samples[:, 4] / samples[:, 0]) - np.sqrt(samples[:, 3] / samples[:, 0])
    return {
        "second_minus_first_mae_kg": float(np.mean(abs(b)) - np.mean(abs(a))),
        "mae_95pct_flight_bootstrap_ci_kg": [float(x) for x in np.quantile(delta_mae, [.025, .975])],
        "second_minus_first_rmse_kg": float(np.sqrt(np.mean(b**2)) - np.sqrt(np.mean(a**2))),
        "rmse_95pct_flight_bootstrap_ci_kg": [float(x) for x in np.quantile(delta_rmse, [.025, .975])],
        "bootstrap_repetitions": repetitions,
    }


def run(raw_dir: Path, feature_dir: Path | None, config: dict) -> dict:
    manifest = None
    if feature_dir is not None:
        manifest = json.loads((feature_dir / "feature_manifest.json").read_text())
        if any(s not in manifest or manifest[s]["subset"] for s in ("train", "rank")):
            raise ValueError("Full train/rank feature manifests are required")
    train = load_split(raw_dir, "train", feature_dir)
    rank = load_split(raw_dir, "rank", feature_dir)
    if manifest is not None and (manifest["train"]["intervals"] != len(train) or manifest["rank"]["intervals"] != len(rank)):
        raise ValueError("Feature manifest interval counts differ from joined data")
    if set(train.flight_id) & set(rank.flight_id):
        raise ValueError("Train/rank flight overlap")
    results = {"protocol": "train April–August 2025; rank September 2025; October final unopened",
               "features": "metadata and ADS-B only" if feature_dir else "metadata only",
               "feature_manifest": manifest,
               "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
               "configuration": config,
               "train_intervals": len(train), "rank_intervals": len(rank),
               "train_flights": int(train.flight_id.nunique()), "rank_flights": int(rank.flight_id.nunique()),
               "unseen_rank_types": sorted(set(rank.aircraft_type) - set(train.aircraft_type)),
               "models": {}}
    models = {"median_type_rate": type_rate_predictions(train, rank)}
    for name, use_trajectory in [("histgb_metadata", False), ("histgb_adsb", True)]:
        if use_trajectory and feature_dir is None:
            continue
        x_train, x_rank, columns = design(train, rank, use_trajectory)
        estimator = HistGradientBoostingRegressor(**config["hist_gradient_boosting"])
        estimator.fit(x_train, train.fuel_kg.to_numpy())
        models[name] = np.maximum(0, estimator.predict(x_rank))
        results["models"][name] = {"predictor_columns": columns}
    for name, predicted in models.items():
        results["models"].setdefault(name, {})
        results["models"][name].update({"overall": score(rank, predicted),
                                        "subgroups": grouped_scores(rank, predicted)})
    comparisons = [("median_type_rate", "histgb_metadata")]
    if feature_dir is not None:
        comparisons.extend([("histgb_metadata", "histgb_adsb"), ("median_type_rate", "histgb_adsb")])
    results["paired_comparisons"] = {
        f"{first}__vs__{second}": paired_flight_bootstrap(rank, models[first], models[second])
        for first, second in comparisons
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--feature-dir", type=Path, default=None,
                        help="Require full verified features_train/rank.parquet; omit for metadata-only run")
    parser.add_argument("--output", type=Path, default=Path("reports/phase-3-rank-baselines.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/phase-3-baselines.json"))
    args = parser.parse_args()
    result = run(args.raw_dir, args.feature_dir, json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, info in result["models"].items():
        print(name, {k: round(info["overall"][k], 2) for k in ("mae_kg", "rmse_kg", "flight_macro_mae_kg")})


if __name__ == "__main__":
    main()
