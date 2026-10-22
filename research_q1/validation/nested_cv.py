"""Group-aware nested cross-validation with a fixed candidate grid."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from .splits import grouped_folds


def grouped_nested_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray, model_factory,
                      parameter_grid: dict, *, outer_folds: int = 5,
                      inner_folds: int = 3, seed: int = 20251022) -> dict:
    """Return leakage-safe outer-fold predictions and selected parameters."""
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups).astype(str)
    if not (len(x) == len(y) == len(groups)):
        raise ValueError("Nested-CV arrays must align")
    candidates = list(ParameterGrid(parameter_grid))
    if not candidates:
        raise ValueError("Parameter grid is empty")
    outer = grouped_folds(pd.DataFrame({"group": groups}), "group", outer_folds, seed)
    prediction = np.empty(len(y), dtype=float)
    audits = []
    for fold in range(outer_folds):
        test = outer == fold
        development = ~test
        dev_groups = groups[development]
        inner = grouped_folds(pd.DataFrame({"group": dev_groups}), "group",
                              inner_folds, seed + fold + 1)
        scores = []
        for parameters in candidates:
            errors = []
            for inner_fold in range(inner_folds):
                validate = inner == inner_fold
                train = ~validate
                model = model_factory(parameters)
                model.fit(x[development][train], y[development][train])
                errors.extend(np.abs(model.predict(x[development][validate]) - y[development][validate]))
            scores.append(float(np.mean(errors)))
        best_index = int(np.argmin(scores))
        model = model_factory(candidates[best_index])
        model.fit(x[development], y[development])
        prediction[test] = model.predict(x[test])
        audits.append({"outer_fold": fold, "test_groups": int(len(np.unique(groups[test]))),
                       "best_parameters": candidates[best_index], "inner_mae": scores[best_index]})
    return {"oof_prediction": prediction, "outer_folds": outer, "audit": audits,
            "group_overlap": False, "candidate_count": len(candidates)}

