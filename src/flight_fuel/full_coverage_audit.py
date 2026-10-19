"""Population-wide, streaming ADS-B availability audit; no model training.

Run after download_archives and data_audit. ZIP and Parquet files remain ignored;
only aggregate counts are written to reports/.
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

FEATURE_COLS = ("timestamp", "flight_id", "latitude", "longitude", "altitude", "groundspeed", "track", "vertical_rate", "source")


def summarize(raw_dir: Path) -> dict:
    result = {"method": "ADS-B rows with timestamp and position in each labeled [start,end) interval; gap includes start/end edges", "splits": {}}
    for split in SPLITS:
        filename = f"flights_{split}.zip"
        size, md5 = ARCHIVES[filename]
        archive_path = raw_dir / filename
        if not archive_path.exists() or archive_path.stat().st_size != size or checksum(archive_path) != md5:
            raise ValueError(f"ZIP absent or checksum mismatch: {archive_path}")
        flights = pd.read_parquet(raw_dir / f"flightlist_{split}.parquet")
        fuel = pd.read_parquet(raw_dir / f"fuel_{split}.parquet")
        by_flight = dict(tuple(fuel.groupby("flight_id", sort=False)))
        ids = set(flights.flight_id)
        totals = {
            "flights": len(ids), "intervals": len(fuel), "trajectory_rows": 0, "adsb_rows": 0,
            "acars_rows": 0, "other_source_rows": 0, "adsb_position_rows": 0,
            "adsb_position_rows_missing_speed": 0, "zero_adsb_intervals": 0,
            "one_to_five_adsb_intervals": 0, "six_plus_adsb_intervals": 0,
            "intervals_with_max_observation_gap_over_60s": 0,
            "intervals_with_max_observation_gap_over_300s": 0,
            "flights_with_any_zero_adsb_interval": 0,
            "flights_with_all_zero_adsb_intervals": 0,
            "flight_id_mismatch_rows": 0,
        }
        per_type: dict[str, dict] = {}
        sampled_raw_values = {"flight_median_adsb_altitude": [], "flight_median_adsb_groundspeed": [], "flight_p95_abs_adsb_vertical_rate": []}
        with zipfile.ZipFile(archive_path) as archive:
            members = {
                name.removeprefix(f"flights_{split}/").removesuffix(".parquet"): name
                for name in archive.namelist() if name.startswith(f"flights_{split}/") and name.endswith(".parquet")
            }
            if set(members) != ids:
                raise ValueError(f"ZIP member IDs differ from flightlist in {split}: missing={len(ids-set(members))}, extra={len(set(members)-ids)}")
            for i, row in enumerate(flights.itertuples(index=False), 1):
                flight_id = row.flight_id
                with archive.open(members[flight_id]) as member:
                    trajectory = pd.read_parquet(io.BytesIO(member.read()), columns=list(FEATURE_COLS))
                if not set(trajectory.flight_id.dropna()).issubset({flight_id}):
                    totals["flight_id_mismatch_rows"] += int((trajectory.flight_id != flight_id).sum())
                totals["trajectory_rows"] += len(trajectory)
                totals["acars_rows"] += int(trajectory.source.eq("acars").sum())
                totals["other_source_rows"] += int((~trajectory.source.isin(("adsb", "acars"))).sum())
                adsb = trajectory.loc[trajectory.source.eq("adsb")]
                totals["adsb_rows"] += len(adsb)
                position = adsb.dropna(subset=["timestamp", "latitude", "longitude"])
                totals["adsb_position_rows"] += len(position)
                totals["adsb_position_rows_missing_speed"] += int(position.groundspeed.isna().sum())
                if len(position):
                    for key, values in (
                        ("flight_median_adsb_altitude", position.altitude),
                        ("flight_median_adsb_groundspeed", position.groundspeed),
                        ("flight_p95_abs_adsb_vertical_rate", position.vertical_rate.abs()),
                    ):
                        sampled_raw_values[key].append(float(values.median() if "median" in key else values.quantile(.95)))

                timestamps = np.sort(position.timestamp.to_numpy(dtype="datetime64[ns]"))
                interval = by_flight[flight_id]
                starts = interval.start.to_numpy(dtype="datetime64[ns]")
                ends = interval.end.to_numpy(dtype="datetime64[ns]")
                left = np.searchsorted(timestamps, starts, side="left")
                right = np.searchsorted(timestamps, ends, side="left")
                counts = right - left
                zero = int((counts == 0).sum())
                totals["zero_adsb_intervals"] += zero
                totals["one_to_five_adsb_intervals"] += int(((counts >= 1) & (counts <= 5)).sum())
                totals["six_plus_adsb_intervals"] += int((counts >= 6).sum())
                totals["flights_with_any_zero_adsb_interval"] += int(zero > 0)
                totals["flights_with_all_zero_adsb_intervals"] += int(zero == len(interval))
                kind = per_type.setdefault(row.aircraft_type, {"flights": 0, "intervals": 0, "zero_adsb_intervals": 0})
                kind["flights"] += 1
                kind["intervals"] += len(interval)
                kind["zero_adsb_intervals"] += zero
                for start, end, lo, hi in zip(starts, ends, left, right):
                    gaps = np.diff(np.concatenate(([start], timestamps[lo:hi], [end]))).astype("timedelta64[ms]").astype(np.int64)
                    biggest = int(gaps.max())
                    totals["intervals_with_max_observation_gap_over_60s"] += int(biggest > 60000)
                    totals["intervals_with_max_observation_gap_over_300s"] += int(biggest > 300000)
                if i % 2000 == 0:
                    print(f"{split}: {i}/{len(ids)} flights", flush=True)
        result["splits"][split] = {
            "archive_md5": md5, "totals": totals, "by_aircraft_type": per_type,
            "raw_unit_diagnostics": {
                name: {
                    "n": len(finite := [v for v in values if np.isfinite(v)]),
                    "median": float(np.median(finite)) if finite else None,
                    "p95": float(np.percentile(finite, 95)) if finite else None,
                }
                for name, values in sampled_raw_values.items()
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("reports/phase-2-full-coverage-audit.json"))
    args = parser.parse_args()
    results = summarize(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
