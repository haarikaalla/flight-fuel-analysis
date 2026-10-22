"""Cross-fitted hybrid residual and local-scale learner for PI-MCS."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_q1.validation.splits import grouped_folds


class HybridResidualModel:
    """Physics + mean(HGB residual, MLP residual), with OOF local error scale."""

    def __init__(self, *, seed: int = 20251022, folds: int = 5,
                 hgb_parameters: dict | None = None, mlp_parameters: dict | None = None):
        self.seed = seed
        self.folds = folds
        self.hgb_parameters = {"max_iter": 200, "max_leaf_nodes": 31,
                               "learning_rate": 0.05, "random_state": seed,
                               "early_stopping": False, **(hgb_parameters or {})}
        self.mlp_parameters = {"hidden_layer_sizes": (64, 32), "max_iter": 500,
                               "early_stopping": False, "random_state": seed,
                               **(mlp_parameters or {})}

    def _models(self):
        hgb = HistGradientBoostingRegressor(**self.hgb_parameters)
        mlp = make_pipeline(SimpleImputer(), StandardScaler(),
                            MLPRegressor(**self.mlp_parameters))
        return hgb, mlp

    def fit(self, x: np.ndarray, y: np.ndarray, physics_prediction: np.ndarray,
            flight_groups: np.ndarray) -> "HybridResidualModel":
        x, y, physics = np.asarray(x, float), np.asarray(y, float), np.asarray(physics_prediction, float)
        if not (len(x) == len(y) == len(physics) == len(flight_groups)):
            raise ValueError("Residual-model inputs must have equal lengths")
        fold_frame = pd.DataFrame({"flight_id": np.asarray(flight_groups).astype(str)})
        fold_id = grouped_folds(fold_frame, "flight_id", self.folds, self.seed)
        target = y - physics
        oof = np.empty(len(y), dtype=float)
        for fold in range(self.folds):
            held = fold_id == fold
            train = ~held
            hgb, mlp = self._models()
            hgb.fit(x[train], target[train])
            mlp.fit(x[train], target[train])
            oof[held] = 0.5 * (hgb.predict(x[held]) + mlp.predict(x[held]))
        self.hgb_, self.mlp_ = self._models()
        self.hgb_.fit(x, target)
        self.mlp_.fit(x, target)
        absolute_oof_error = np.abs(target - oof)
        self.scale_ = HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=20, learning_rate=0.05,
            random_state=self.seed, early_stopping=False,
        ).fit(x, absolute_oof_error)
        positive = absolute_oof_error[absolute_oof_error > 0]
        self.scale_floor_ = float(np.quantile(positive, 0.05)) if len(positive) else 1.0
        self.oof_audit_ = {
            "folds": self.folds, "grouped_by": "flight_id",
            "oof_residual_mae_kg": float(absolute_oof_error.mean()),
            "scale_floor_kg": self.scale_floor_,
        }
        return self

    def predict(self, x: np.ndarray, physics_prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "scale_"):
            raise RuntimeError("HybridResidualModel must be fit before prediction")
        x, physics = np.asarray(x, float), np.asarray(physics_prediction, float)
        residual = 0.5 * (self.hgb_.predict(x) + self.mlp_.predict(x))
        prediction = np.maximum(0.0, physics + residual)
        sigma = np.maximum(self.scale_floor_, self.scale_.predict(x))
        return prediction, sigma

