"""Group-aware covariate-shift diagnostics."""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .splits import grouped_folds


def domain_classifier_auc(x_source: np.ndarray, x_target: np.ndarray,
                          source_groups: np.ndarray, target_groups: np.ndarray,
                          *, folds: int = 5, seed: int = 20251022) -> dict:
    """Out-of-group AUC for distinguishing source from target covariates."""
    x = np.vstack([np.asarray(x_source, float), np.asarray(x_target, float)])
    y = np.r_[np.zeros(len(x_source), dtype=int), np.ones(len(x_target), dtype=int)]
    groups = np.r_[np.char.add("S:", np.asarray(source_groups).astype(str)),
                   np.char.add("T:", np.asarray(target_groups).astype(str))]
    import pandas as pd
    frame = pd.DataFrame({"group": groups})
    fold_id = grouped_folds(frame, "group", n_splits=folds, seed=seed)
    probability = np.empty(len(y), dtype=float)
    for fold in range(folds):
        test = fold_id == fold
        train = ~test
        model = make_pipeline(SimpleImputer(), StandardScaler(),
                              LogisticRegression(max_iter=2000, random_state=seed))
        model.fit(x[train], y[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
    auc = float(roc_auc_score(y, probability))
    odds = np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1)
    prior_ratio = len(x_source) / len(x_target)
    return {"auc": auc, "density_ratio": odds * prior_ratio,
            "interpretation": "0.5 is indistinguishable; larger AUC indicates measured covariate shift"}

