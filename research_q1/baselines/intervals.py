"""Uncertainty baselines sharing the same calibration/test partitions."""

from __future__ import annotations

import numpy as np

from research_q1.novel.conformal import (AdaptiveConformal, HierarchicalMondrian,
                                          cqr_interval, split_interval,
                                          weighted_intervals)


def standard_split_conformal(calibration_y, calibration_prediction, test_prediction,
                             *, alpha=.1, test_sigma=None, tank_capacity_kg=None):
    calibration_y = np.asarray(calibration_y, float)
    calibration_prediction = np.asarray(calibration_prediction, float)
    sigma = np.ones(len(test_prediction)) if test_sigma is None else np.asarray(test_sigma, float)
    scores = np.abs(calibration_y - calibration_prediction)
    cap = np.inf if tank_capacity_kg is None else tank_capacity_kg
    return split_interval(test_prediction, sigma, scores, alpha=alpha, upper_bound=cap)


def ensemble_variance_interval(point, standard_deviation, multiplier=1.96,
                               tank_capacity_kg=None):
    point, scale = np.asarray(point, float), np.asarray(standard_deviation, float)
    cap = np.full(len(point), np.inf) if tank_capacity_kg is None else np.asarray(tank_capacity_kg, float)
    return np.maximum(0, point - multiplier * scale), np.minimum(cap, point + multiplier * scale)


__all__ = ["AdaptiveConformal", "HierarchicalMondrian", "cqr_interval",
           "ensemble_variance_interval", "standard_split_conformal", "weighted_intervals"]

