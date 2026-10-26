"""Isotonic, Platt and inductive Venn-Abers risk calibration."""

from __future__ import annotations

import numpy as np
from scipy.special import expit
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
            from venn_abers import VennAbers
            self.location_ = float(np.median(x))
            q25, q75 = np.quantile(x, [.25, .75])
            self.scale_ = float(max(q75 - q25, np.std(x), 1e-9))
            base = expit((x - self.location_) / self.scale_)
            self.model_ = VennAbers().fit(p_cal=np.c_[1 - base, base], y_cal=y, precision=6)
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        x = np.asarray(score, float)
        if self.method == "isotonic":
            return self.model_.predict(x)
        if self.method == "platt":
            return self.model_.predict_proba(x.reshape(-1, 1))[:, 1]
        base = expit((x - self.location_) / self.scale_)
        probabilities, self.last_probability_bounds_ = self.model_.predict_proba(
            p_test=np.c_[1 - base, base])
        return probabilities[:, 1]

