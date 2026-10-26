"""Established point and quantile prediction baselines on identical splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class PredictionBundle:
    point: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    ensemble_standard_deviation: np.ndarray | None = None


class BaselineSuite:
    """Fit baselines with one shared, logged tuning budget.

    Hyperparameters are fixed before test evaluation. This class does not use
    test labels and accepts only already encoded numeric features.
    """

    def __init__(self, seed: int = 20251022, max_iterations: int = 300,
                 ensemble_members: int = 5):
        self.seed = seed
        self.max_iterations = max_iterations
        self.ensemble_members = ensemble_members

    def fit_predict(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                    *, physics_test: np.ndarray | None = None,
                    existing_test: np.ndarray | None = None,
                    alpha: float = 0.1) -> dict[str, PredictionBundle]:
        x_train, y_train, x_test = np.asarray(x_train, float), np.asarray(y_train, float), np.asarray(x_test, float)
        results: dict[str, PredictionBundle] = {}
        models = {
            "ridge": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0)),
            "lasso": make_pipeline(SimpleImputer(), StandardScaler(),
                                   Lasso(alpha=0.01, max_iter=5000, random_state=self.seed)),
            "hist_gradient_boosting": HistGradientBoostingRegressor(
                max_iter=self.max_iterations, learning_rate=.05, max_leaf_nodes=31,
                random_state=self.seed, early_stopping=False),
            "mlp": make_pipeline(SimpleImputer(), StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=self.max_iterations,
                random_state=self.seed, early_stopping=False)),
        }
        for name, model in models.items():
            model.fit(x_train, y_train)
            results[name] = PredictionBundle(np.maximum(0.0, model.predict(x_test)))
        members = []
        for index in range(self.ensemble_members):
            model = HistGradientBoostingRegressor(
                max_iter=self.max_iterations, learning_rate=.05, max_leaf_nodes=31,
                l2_regularization=.1, random_state=self.seed + index, early_stopping=False,
            ).fit(x_train, y_train)
            members.append(model.predict(x_test))
        member_array = np.vstack(members)
        results["histgb_ensemble"] = PredictionBundle(
            np.maximum(0.0, member_array.mean(axis=0)),
            ensemble_standard_deviation=member_array.std(axis=0, ddof=1),
        )
        quantiles = []
        for q in (alpha / 2, 1 - alpha / 2):
            model = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=self.max_iterations,
                learning_rate=.05, max_leaf_nodes=31, random_state=self.seed,
                early_stopping=False,
            ).fit(x_train, y_train)
            quantiles.append(np.maximum(0.0, model.predict(x_test)))
        lower, upper = np.minimum(quantiles[0], quantiles[1]), np.maximum(quantiles[0], quantiles[1])
        results["quantile_regression"] = PredictionBundle((lower + upper) / 2, lower, upper)
        if physics_test is not None:
            results["physics_only"] = PredictionBundle(np.asarray(physics_test, float))
        if existing_test is not None:
            results["existing_project"] = PredictionBundle(np.asarray(existing_test, float))
        return results

