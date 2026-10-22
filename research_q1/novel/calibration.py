"""Isotonic, Platt and inductive Venn-Abers risk calibration."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class RiskCalibrator:
    def __init__(self, method: str = "isotonic", seed: int = 20251022):
        if method not in {"isotonic", "platt", "venn_abers"}:
            raise ValueError("Unknown risk-calibration method")
        self.method, self.seed = method, seed

    def fit(self, score: np.ndarray, event: np.ndarray) -> "RiskCalibrator":
        x, y = np.asarray(score, float), np.asarray(event, int)
        if np.any((y != 0) & (y != 1)):
            raise ValueError("Risk calibration requires a binary event")
        if self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(x, y)
        elif self.method == "platt":
            self.model_ = LogisticRegression(random_state=self.seed).fit(x.reshape(-1, 1), y)
        else:
            self.x_, self.y_ = x, y
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        x = np.asarray(score, float)
        if self.method == "isotonic":
            return self.model_.predict(x)
        if self.method == "platt":
            return self.model_.predict_proba(x.reshape(-1, 1))[:, 1]
        probabilities = []
        for value in x:
            models = [IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
                np.r_[self.x_, value], np.r_[self.y_, label]) for label in (0, 1)]
            low, high = (float(model.predict([value])[0]) for model in models)
            denominator = 1 - low + high
            probabilities.append(high / denominator if denominator else 0.5)
        return np.asarray(probabilities)

