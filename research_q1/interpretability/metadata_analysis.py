"""Real SHAP, permutation and refit-ablation analysis for the PRC metadata model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from research_q1.baselines.metadata_benchmark import design, load_period
from research_q1.interpretability.analysis import feature_group_ablation, permutation_importance
from research_q1.validation.metrics import regression_metrics


def run(raw_dir: Path, *, seed: int = 20251022, shap_sample: int = 2000) -> dict:
    train, test = load_period(raw_dir, "train"), load_period(raw_dir, "final")
    x_train, x_test, feature_names = design(train, test)

    def model(parameters=None):
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=.05, max_leaf_nodes=31,
            random_state=seed, early_stopping=False, **(parameters or {}))

    fitted = model().fit(x_train, train.fuel_kg.to_numpy())
    permutation = permutation_importance(
        fitted, x_test, test.fuel_kg.to_numpy(), feature_names, repetitions=10, seed=seed)
    groups = {
        "interval_duration": ["duration_min"],
        "whole_flight_timing": ["flight_duration_min", "elapsed_from_takeoff_min", "remaining_to_landing_min"],
        "time_of_day": ["start_hour_sin", "start_hour_cos"],
        "aircraft_type": [name for name in feature_names if name.startswith("type_")],
    }
    ablation = feature_group_ablation(
        lambda a, b, c: np.maximum(0.0, model().fit(a, b).predict(c)),
        x_train, train.fuel_kg.to_numpy(), x_test, test.fuel_kg.to_numpy(),
        feature_names, groups)

    import shap
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(test), size=min(shap_sample, len(test)), replace=False))
    explainer = shap.TreeExplainer(fitted, feature_perturbation="tree_path_dependent")
    values = np.asarray(explainer.shap_values(x_test[chosen], check_additivity=True), float)
    global_rows = sorted([
        {"feature": name, "mean_absolute_shap_kg": float(np.mean(np.abs(values[:, index]))),
         "mean_signed_shap_kg": float(np.mean(values[:, index]))}
        for index, name in enumerate(feature_names)
    ], key=lambda row: -row["mean_absolute_shap_kg"])
    subgroup = {}
    selected_types = test.iloc[chosen].aircraft_type.astype(str).to_numpy()
    for aircraft_type in sorted(np.unique(selected_types)):
        member = selected_types == aircraft_type
        if member.sum() >= 20:
            subgroup[aircraft_type] = {
                feature_names[index]: float(np.mean(np.abs(values[member, index])))
                for index in range(len(feature_names))
            }
    return {
        "status": "REAL_DATA_EXPLORATORY_METADATA_INTERPRETABILITY",
        "evidence_boundary": "No physics, weather, raw trajectory or data-quality features are present in this metadata model",
        "test_regression": regression_metrics(test.fuel_kg.to_numpy(), np.maximum(0.0, fitted.predict(x_test))),
        "permutation_importance": permutation,
        "refit_feature_group_ablation": ablation,
        "shap": {"sample_n": int(len(chosen)), "explainer": "exact tree-path-dependent TreeSHAP",
                 "global": global_rows, "mean_absolute_by_aircraft": subgroup},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path,
                        default=Path("research_q1/outputs/metadata-interpretability.json"))
    args = parser.parse_args()
    result = run(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
