"""Small, deterministic range-request probe of the large remote trajectory archives.

This is a schema and feasibility check, NOT a population estimate. No trajectory
or fuel observations are written to the report; only aggregate diagnostics.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip

from .data_audit import RECORD, SPLITS

ALLOWED_ADSB = (
    "timestamp", "flight_id", "typecode", "latitude", "longitude",
    "altitude", "groundspeed", "track", "vertical_rate", "source",
)
FORBIDDEN_PREDICTORS = ("mach", "TAS", "CAS", "fuel_kg")


def probe(raw_dir: Path, splits: tuple[str, ...], limit: int) -> dict:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    result = {"sampling": "lexicographically first N flight IDs per split; diagnostic only", "splits": {}}
    for split in splits:
        flights = pd.read_parquet(raw_dir / f"flightlist_{split}.parquet")
        intervals = pd.read_parquet(raw_dir / f"fuel_{split}.parquet")
        ids = sorted(flights.flight_id)[:limit]
        url = f"{RECORD}/flights_{split}.zip/content"
        split_results = []
        with RemoteZip(url) as archive:
            names = set(archive.namelist())
            for flight_id in ids:
                member = f"flights_{split}/{flight_id}.parquet"
                if member not in names:
                    raise ValueError(f"Missing archive member {member}")
                trajectory = pd.read_parquet(io.BytesIO(archive.read(member)))
                if not set(ALLOWED_ADSB).issubset(trajectory.columns):
                    raise ValueError(f"Missing ADS-B fields in {member}")
                if not set(FORBIDDEN_PREDICTORS[:-1]).issubset(trajectory.columns):
                    raise ValueError(f"Missing ACARS fields needed for leakage audit: {member}")
                if set(trajectory.flight_id.dropna()) != {flight_id}:
                    raise ValueError(f"Mismatched flight ID in {member}")
                adsb = trajectory.loc[trajectory.source.eq("adsb"), list(ALLOWED_ADSB)].sort_values("timestamp")
                acars = trajectory.source.eq("acars")
                own_intervals = intervals.loc[intervals.flight_id.eq(flight_id)]
                times = adsb.timestamp.to_numpy(dtype="datetime64[ns]")
                start = own_intervals.start.to_numpy(dtype="datetime64[ns]")
                end = own_intervals.end.to_numpy(dtype="datetime64[ns]")
                counts = pd.Series(times.searchsorted(end, side="left") - times.searchsorted(start, side="left"))
                gaps = adsb.timestamp.diff().dt.total_seconds().dropna()
                split_results.append({
                    "flight_id": flight_id,
                    "rows": len(trajectory), "adsb_rows": len(adsb), "acars_rows": int(acars.sum()),
                    "intervals": len(own_intervals), "intervals_without_adsb": int((counts == 0).sum()),
                    "max_adsb_gap_seconds": float(gaps.max()) if len(gaps) else None,
                    "adsb_altitude_p50_raw": float(adsb.altitude.median()),
                    "adsb_groundspeed_p50_raw": float(adsb.groundspeed.median()),
                    "adsb_vertical_rate_abs_p95_raw": float(adsb.vertical_rate.abs().quantile(.95)),
                    "adsb_null_counts": adsb.isna().sum().to_dict(),
                })
        result["splits"][split] = split_results
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("reports/phase-2-trajectory-probe.json"))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    result = probe(args.raw_dir, tuple(args.splits), args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
