"""Extract retrospective interval features from ADS-B rows only.

Raw altitude, speed and vertical-rate values are statistical summaries, not
physical SI quantities; the published unit documentation needs reconciliation.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data_audit import SPLITS, checksum
from .download_archives import ARCHIVES

TRAJECTORY_COLS = [
    "timestamp", "flight_id", "source", "latitude", "longitude",
    "altitude", "groundspeed", "vertical_rate",
]
VALUE_COLS = ["altitude", "groundspeed", "vertical_rate"]
FEATURE_COLS = [
    "adsb_position_count", "adsb_speed_count", "adsb_altitude_median_raw",
    "adsb_altitude_p10_raw", "adsb_altitude_p90_raw", "adsb_speed_median_raw",
    "adsb_speed_p10_raw", "adsb_speed_p90_raw", "adsb_vertical_rate_abs_median_raw",
    "adsb_vertical_rate_abs_p90_raw", "adsb_max_gap_s", "adsb_first_gap_s",
    "adsb_last_gap_s", "adsb_max_gap_fraction",
]


def _stat(values: pd.Series, q: float) -> float:
    clean = values.dropna()
    return float(clean.quantile(q)) if len(clean) else float("nan")


def features_for_flight(trajectory: pd.DataFrame, intervals: pd.DataFrame, flight_id: str) -> pd.DataFrame:
    if set(trajectory.flight_id.dropna()) != {flight_id}:
        raise ValueError(f"Trajectory ID mismatch: {flight_id}")
    # Select the source *before* computing any aggregate, and only retain
    # allowed fields. ACARS-only Mach/TAS/CAS are never loaded.
    adsb = trajectory.loc[trajectory.source.eq("adsb"), TRAJECTORY_COLS].dropna(
        subset=["timestamp", "latitude", "longitude"]
    ).sort_values("timestamp")
    times = adsb.timestamp.to_numpy(dtype="datetime64[ns]")
    starts = intervals.start.to_numpy(dtype="datetime64[ns]")
    ends = intervals.end.to_numpy(dtype="datetime64[ns]")
    left = np.searchsorted(times, starts, side="left")
    right = np.searchsorted(times, ends, side="left")
    rows = []
    for idx, start, end, lo, hi in zip(intervals.idx, starts, ends, left, right):
        part = adsb.iloc[lo:hi]
        boundaries = np.concatenate(([start], times[lo:hi], [end]))
        gaps = np.diff(boundaries).astype("timedelta64[ns]").astype(np.int64) / 1e9
        duration_s = float((end - start) / np.timedelta64(1, "s"))
        if duration_s <= 0:
            raise ValueError(f"Invalid duration for interval {idx}")
        alt = part.altitude
        speed = part.groundspeed
        rate = part.vertical_rate.abs()
        rows.append({
            "idx": int(idx), "flight_id": flight_id,
            "adsb_position_count": len(part),
            "adsb_speed_count": int(speed.notna().sum()),
            "adsb_altitude_median_raw": _stat(alt, .5),
            "adsb_altitude_p10_raw": _stat(alt, .1),
            "adsb_altitude_p90_raw": _stat(alt, .9),
            "adsb_speed_median_raw": _stat(speed, .5),
            "adsb_speed_p10_raw": _stat(speed, .1),
            "adsb_speed_p90_raw": _stat(speed, .9),
            "adsb_vertical_rate_abs_median_raw": _stat(rate, .5),
            "adsb_vertical_rate_abs_p90_raw": _stat(rate, .9),
            "adsb_max_gap_s": float(gaps.max()),
            "adsb_first_gap_s": float(gaps[0]),
            "adsb_last_gap_s": float(gaps[-1]),
            "adsb_max_gap_fraction": float(gaps.max() / duration_s),
        })
    return pd.DataFrame(rows, columns=["idx", "flight_id", *FEATURE_COLS])


def build_split(raw_dir: Path, split: str, output_dir: Path, *, max_flights: int | None = None) -> dict:
    filename = f"flights_{split}.zip"
    size, md5 = ARCHIVES[filename]
    archive_path = raw_dir / filename
    if not archive_path.exists() or archive_path.stat().st_size != size or checksum(archive_path) != md5:
        raise ValueError(f"Missing or MD5-invalid archive: {archive_path}")
    flightlist = pd.read_parquet(raw_dir / f"flightlist_{split}.parquet", columns=["flight_id"])
    intervals = pd.read_parquet(raw_dir / f"fuel_{split}.parquet", columns=["idx", "flight_id", "start", "end"])
    selected = sorted(flightlist.flight_id)
    if max_flights is not None:
        selected = selected[:max_flights]
    by_flight = dict(tuple(intervals.groupby("flight_id", sort=False)))
    if set(selected) - set(by_flight):
        raise ValueError("Selected flight has no fuel intervals")
    parts = []
    with zipfile.ZipFile(archive_path) as archive:
        members = {name.rsplit("/", 1)[-1].removesuffix(".parquet"): name
                   for name in archive.namelist() if name.startswith(f"flights_{split}/") and name.endswith(".parquet")}
        if max_flights is None and set(members) != set(flightlist.flight_id):
            raise ValueError("Archive members do not match flight list")
        for n, flight_id in enumerate(selected, 1):
            if flight_id not in members:
                raise ValueError(f"Missing trajectory for {flight_id}")
            with archive.open(members[flight_id]) as member:
                trajectory = pd.read_parquet(io.BytesIO(member.read()), columns=TRAJECTORY_COLS)
            parts.append(features_for_flight(trajectory, by_flight[flight_id], flight_id))
            if n % 1000 == 0:
                print(f"{split}: {n}/{len(selected)} flights", flush=True)
    result = pd.concat(parts, ignore_index=True)
    if result.idx.duplicated().any():
        raise ValueError("Duplicate interval index in extracted features")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if max_flights is None else f"_first_{max_flights}"
    result.to_parquet(output_dir / f"features_{split}{suffix}.parquet", index=False)
    return {"split": split, "archive_md5": md5, "flights": len(selected),
            "intervals": len(result), "zero_adsb_intervals": int(result.adsb_position_count.eq(0).sum()),
            "feature_file": f"features_{split}{suffix}.parquet", "subset": max_flights is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["train", "rank"])
    parser.add_argument("--max-flights", type=int, default=None,
                        help="Smoke test only; choose lexicographically first N flights (not an evaluation sample)")
    args = parser.parse_args()
    if args.max_flights is not None and args.max_flights < 1:
        parser.error("--max-flights must be positive")
    path = args.output_dir / "feature_manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest.update({s: build_split(args.raw_dir, s, args.output_dir, max_flights=args.max_flights) for s in args.splits})
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
