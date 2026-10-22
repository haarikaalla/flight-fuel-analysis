"""Reference-fuel provenance, quantization and independent-label checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import regression_metrics


POUND_KG = 0.45359237


def label_audit(frame: pd.DataFrame, independent_column: str | None = None,
                physics_reference: np.ndarray | None = None) -> dict:
    y = frame.reference_fuel_kg.to_numpy(float)
    pounds = y / POUND_KG
    lattice = np.isclose(pounds, np.round(pounds), atol=1e-4, rtol=0)
    result = {
        "n": int(len(frame)), "label_sources": frame.label_source.value_counts().to_dict(),
        "zero_fraction": float(np.mean(y == 0)), "integer_pound_lattice_fraction": float(lattice.mean()),
        "quantiles_kg": {str(q): float(np.quantile(y, q)) for q in (0, .01, .5, .99, 1)},
        "assumption": "Fuel-on-board differences include reporting, unit-inference and timing error unless label_source states an independently validated measurement.",
    }
    if independent_column is not None:
        if independent_column not in frame:
            raise ValueError(f"Independent reference column is absent: {independent_column}")
        valid = frame[independent_column].notna().to_numpy()
        if not valid.any():
            raise ValueError("Independent reference column has no observed values")
        result["independent_reference"] = {
            "column": independent_column, "n": int(valid.sum()),
            **regression_metrics(frame.loc[valid, independent_column].to_numpy(float), y[valid]),
        }
    if physics_reference is not None:
        result["physics_reference"] = regression_metrics(y, np.asarray(physics_reference, float))
    return result

