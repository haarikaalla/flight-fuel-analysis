"""Grouped validation splits that never randomize individual flights."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    excluded: np.ndarray
    kind: str


def _mask(frame: pd.DataFrame, flights: set[str]) -> np.ndarray:
    return frame.flight_id.astype(str).isin(flights).to_numpy()


def _flight_times(frame: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame({
        "flight_id": frame.flight_id.astype(str),
        "start": pd.to_datetime(frame.interval_start_utc, utc=True),
        "end": pd.to_datetime(frame.interval_end_utc, utc=True),
    })
    return table.groupby("flight_id", sort=True).agg(first=("start", "min"), last=("end", "max"))


def temporal_split(frame: pd.DataFrame, train_before: str, calibration_before: str,
                   test_before: str | None = None) -> Split:
    """Forward split; flights crossing a boundary are excluded, never divided."""
    a, b = pd.Timestamp(train_before), pd.Timestamp(calibration_before)
    a = a.tz_localize("UTC") if a.tz is None else a.tz_convert("UTC")
    b = b.tz_localize("UTC") if b.tz is None else b.tz_convert("UTC")
    c = pd.Timestamp.max.tz_localize("UTC") if test_before is None else pd.Timestamp(test_before)
    c = c.tz_localize("UTC") if c.tz is None else c.tz_convert("UTC")
    if not a < b < c:
        raise ValueError("Temporal boundaries must be strictly increasing")
    times = _flight_times(frame)
    crossing = ((times["first"] < a) & (times["last"] >= a)) | ((times["first"] < b) & (times["last"] >= b)) | ((times["first"] < c) & (times["last"] >= c))
    eligible = times.loc[~crossing]
    train_f = set(eligible.index[eligible["last"] < a])
    cal_f = set(eligible.index[(eligible["first"] >= a) & (eligible["last"] < b)])
    test_f = set(eligible.index[(eligible["first"] >= b) & (eligible["last"] < c)])
    excluded_f = set(times.index) - train_f - cal_f - test_f
    result = Split(_mask(frame, train_f), _mask(frame, cal_f), _mask(frame, test_f),
                   _mask(frame, excluded_f), "temporal")
    assert_split_integrity(frame, result, group_column="flight_id")
    return result


def _stable_group_partition(values: pd.Series, test_fraction: float, seed: int) -> set[str]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    groups = sorted(values.astype(str).unique())
    scored = sorted(groups, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    n_test = min(len(scored) - 1, max(1, int(np.ceil(test_fraction * len(scored)))))
    return set(scored[:n_test])


def grouped_holdout(frame: pd.DataFrame, group_column: str, *, test_fraction: float = 0.2,
                    calibration_fraction: float = 0.2, seed: int = 20251022) -> Split:
    """Disjoint group holdout with deterministic group hashing, never row sampling."""
    if group_column not in frame:
        raise ValueError(f"Missing group column: {group_column}")
    groups = frame[group_column].astype(str)
    test_groups = _stable_group_partition(groups, test_fraction, seed)
    remaining = frame.loc[~groups.isin(test_groups), group_column].astype(str)
    calibration_groups = _stable_group_partition(remaining, calibration_fraction, seed + 1)
    train = ~groups.isin(test_groups | calibration_groups)
    calibration = groups.isin(calibration_groups)
    test = groups.isin(test_groups)
    result = Split(train.to_numpy(), calibration.to_numpy(), test.to_numpy(),
                   np.zeros(len(frame), dtype=bool), f"{group_column}_holdout")
    assert_split_integrity(frame, result, group_column=group_column)
    return result


def route_holdout(frame: pd.DataFrame, **kwargs) -> Split:
    work = frame.copy()
    work["route_id"] = work.origin_icao.astype(str) + "-" + work.destination_icao.astype(str)
    return grouped_holdout(work, "route_id", **kwargs)


def grouped_folds(frame: pd.DataFrame, group_column: str, n_splits: int = 5,
                  seed: int = 20251022) -> np.ndarray:
    """Assign entire named groups to deterministic, approximately balanced folds."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    sizes = frame.assign(_group=frame[group_column].astype(str)).groupby("_group", dropna=False).size()
    if len(sizes) < n_splits:
        raise ValueError("Fewer groups than requested folds")
    ordered = sorted(sizes.index, key=lambda value: (
        -int(sizes.loc[value]), hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    ))
    totals = np.zeros(n_splits, dtype=int)
    assignment: dict[str, int] = {}
    for name in ordered:
        fold = int(np.argmin(totals))
        assignment[name] = fold
        totals[fold] += int(sizes.loc[name])
    return frame[group_column].astype(str).map(assignment).to_numpy(dtype=int)


def assert_split_integrity(frame: pd.DataFrame, split: Split, group_column: str) -> None:
    arrays = (split.train, split.calibration, split.test, split.excluded)
    if any(np.asarray(mask).dtype != bool or len(mask) != len(frame) for mask in arrays):
        raise AssertionError("Every split mask must be a row-aligned boolean array")
    membership = sum(mask.astype(int) for mask in arrays)
    if not np.all(membership == 1):
        raise AssertionError("Every row must occur in exactly one split partition")
    groups = frame[group_column].astype(str)
    named = [set(groups[mask]) for mask in arrays[:3]]
    if (named[0] & named[1]) or (named[0] & named[2]) or (named[1] & named[2]):
        raise AssertionError(f"{group_column} leakage across train/calibration/test")

