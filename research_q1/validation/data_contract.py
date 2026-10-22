"""Strict input contract and provenance fingerprinting for PI-MCS experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "sample_id", "flight_id", "interval_start_utc", "interval_end_utc",
    "year", "season", "origin_icao", "destination_icao", "operator_id",
    "aircraft_type", "aircraft_family", "flight_phase", "reference_fuel_kg",
    "label_source", "ground_velocity_east_mps", "ground_velocity_north_mps",
    "wind_east_mps", "wind_north_mps", "altitude_m", "vertical_rate_mps",
    "wing_area_m2", "aspect_ratio", "tank_capacity_kg", "adsb_position_count",
    "source_release", "source_license",
)

NUMERIC_NONNEGATIVE = (
    "reference_fuel_kg", "altitude_m", "wing_area_m2", "aspect_ratio",
    "tank_capacity_kg", "adsb_position_count",
)


class FrozenDataAccessError(RuntimeError):
    """Raised when code attempts to open a protected confirmation table."""


def guard_confirmation_access(path: Path, *, allow_confirmation: bool = False,
                              access_log: Path | None = None) -> None:
    """Refuse protected-label access unless a one-time run is explicitly enabled."""
    protected = any(token in path.name.lower() for token in ("confirm", "external_test", "future"))
    if protected and not allow_confirmation:
        if access_log is not None:
            access_log.parent.mkdir(parents=True, exist_ok=True)
            with access_log.open("a", encoding="utf-8") as handle:
                handle.write(f"DENIED | {path.resolve()}\n")
        raise FrozenDataAccessError(f"Protected confirmation data cannot be opened: {path}")


def read_table(path: Path, *, allow_confirmation: bool = False,
               access_log: Path | None = None) -> pd.DataFrame:
    guard_confirmation_access(path, allow_confirmation=allow_confirmation, access_log=access_log)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Approved analytical tables must be Parquet or CSV")


def validate_frame(frame: pd.DataFrame, *, required: Iterable[str] = REQUIRED_COLUMNS) -> dict:
    required = tuple(required)
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Analytical table is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Analytical table is empty")
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("sample_id must be complete and unique")
    identifiers = ["flight_id", "origin_icao", "destination_icao", "aircraft_type",
                   "aircraft_family", "flight_phase", "label_source", "source_release",
                   "source_license"]
    if frame[identifiers].isna().any().any():
        raise ValueError("Required identifiers and provenance fields must be complete")
    start = pd.to_datetime(frame.interval_start_utc, utc=True, errors="coerce")
    end = pd.to_datetime(frame.interval_end_utc, utc=True, errors="coerce")
    if start.isna().any() or end.isna().any() or (end <= start).any():
        raise ValueError("Intervals require valid increasing UTC boundaries")
    numeric_names = list(NUMERIC_NONNEGATIVE) + [
        "ground_velocity_east_mps", "ground_velocity_north_mps",
        "wind_east_mps", "wind_north_mps", "vertical_rate_mps",
    ]
    numeric = frame[numeric_names].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Physical and label fields must be finite")
    if (numeric[list(NUMERIC_NONNEGATIVE)] < 0).any().any():
        raise ValueError("Fuel, geometry, altitude and observation counts must be nonnegative")
    if (frame.reference_fuel_kg > frame.tank_capacity_kg).any():
        raise ValueError("Reference fuel exceeds declared tank capacity")
    route = frame.origin_icao.astype(str) + "-" + frame.destination_icao.astype(str)
    return {
        "rows": int(len(frame)),
        "flights": int(frame.flight_id.nunique()),
        "years": sorted(map(int, pd.Series(frame.year).unique())),
        "operators": int(frame.operator_id.nunique(dropna=True)),
        "routes": int(route.nunique()),
        "aircraft_types": int(frame.aircraft_type.nunique()),
        "aircraft_families": int(frame.aircraft_family.nunique()),
        "first_start_utc": start.min().isoformat(),
        "last_end_utc": end.max().isoformat(),
        "label_sources": sorted(frame.label_source.astype(str).unique().tolist()),
        "source_releases": sorted(frame.source_release.astype(str).unique().tolist()),
        "licenses": sorted(frame.source_license.astype(str).unique().tolist()),
    }


def frame_fingerprint(frame: pd.DataFrame, columns: Iterable[str] = REQUIRED_COLUMNS) -> str:
    columns = tuple(columns)
    missing = [name for name in columns if name not in frame]
    if missing:
        raise ValueError(f"Cannot fingerprint missing columns: {missing}")
    canonical = frame[list(columns)].sort_values("sample_id").to_json(
        orient="records", date_format="iso", date_unit="us", double_precision=15,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_manifest(frame: pd.DataFrame, output: Path) -> dict:
    manifest = validate_frame(frame)
    manifest["sha256"] = frame_fingerprint(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest

