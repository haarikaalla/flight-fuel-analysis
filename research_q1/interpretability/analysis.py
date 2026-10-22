"""Interpretability methods evaluated on untouched grouped test folds."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from research_q1.validation.metrics import regression_metrics


def permutation_importance(model, x: np.ndarray, y: np.ndarray, feature_names: list[str],
                           *, repetitions: int = 20, seed: int = 20251022) -> list[dict]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape[1] != len(feature_names):
        raise ValueError("Feature names must match design columns")
    baseline = regression_metrics(y, model.predict(x))["mae_kg"]
    rng = np.random.default_rng(seed)
    rows = []
    for column, name in enumerate(feature_names):
        changes = []
        for _ in range(repetitions):
            perturbed = x.copy()
            perturbed[:, column] = rng.permutation(perturbed[:, column])
            changes.append(regression_metrics(y, model.predict(perturbed))["mae_kg"] - baseline)
        rows.append({"feature": name, "delta_mae_kg": float(np.mean(changes)),
                     "delta_mae_sd_kg": float(np.std(changes, ddof=1))})
    return sorted(rows, key=lambda row: -row["delta_mae_kg"])


def feature_group_ablation(fit_predict, x_train: np.ndarray, y_train: np.ndarray,
                           x_test: np.ndarray, y_test: np.ndarray,
                           feature_names: list[str], feature_groups: dict[str, list[str]]) -> list[dict]:
    """Refit for every ablation; never zero features only at test time."""
    lookup = {name: index for index, name in enumerate(feature_names)}
    rows = []
    all_prediction = fit_predict(x_train, y_train, x_test)
    baseline = regression_metrics(y_test, all_prediction)["mae_kg"]
    rows.append({"ablation": "none", "features": len(feature_names), "mae_kg": baseline,
                 "delta_mae_kg": 0.0})
    for group, names in feature_groups.items():
        omitted = set(names)
        keep = [lookup[name] for name in feature_names if name not in omitted]
        if not keep:
            raise ValueError(f"Ablation {group} removes every feature")
        prediction = fit_predict(x_train[:, keep], y_train, x_test[:, keep])
        mae = regression_metrics(y_test, prediction)["mae_kg"]
        rows.append({"ablation": f"remove_{group}", "features": len(keep),
                     "mae_kg": mae, "delta_mae_kg": mae - baseline})
    return rows


def shap_values(model, x_background: np.ndarray, x_explain: np.ndarray):
    """Compute SHAP values when the separately licensed optional package exists."""
    try:
        import shap
    except ImportError as error:
        raise RuntimeError("Install and record a compatible SHAP version before SHAP analysis") from error
    explainer = shap.Explainer(model, np.asarray(x_background, float))
    return explainer(np.asarray(x_explain, float))


COMPONENT_ABLATIONS = {
    "full_pi_mcs": [],
    "remove_physics_prior": ["physics_prior"],
    "remove_mondrian": ["mondrian"],
    "remove_risk_calibration": ["risk_calibration"],
    "replace_exact_with_greedy": ["exact_optimizer"],
}


def importance_stability(fold_importances: list[np.ndarray]) -> dict:
    """Pairwise Spearman stability of feature rankings across grouped folds."""
    if len(fold_importances) < 2:
        raise ValueError("At least two fold-importance vectors are required")
    arrays = [np.asarray(values, float) for values in fold_importances]
    if len({len(values) for values in arrays}) != 1:
        raise ValueError("Fold-importance vectors must align")
    correlations = [float(spearmanr(arrays[i], arrays[j]).statistic)
                    for i in range(len(arrays)) for j in range(i + 1, len(arrays))]
    return {"pairwise_spearman": correlations, "mean_spearman": float(np.nanmean(correlations)),
            "minimum_spearman": float(np.nanmin(correlations))}

