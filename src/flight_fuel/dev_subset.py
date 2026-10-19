"""Freeze a small flight-ID-only development subset from train and ranking periods.

The final period is intentionally absent. Selection is deterministic within each
aircraft type, independent of labels, and is not an unbiased test sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def choose(raw_dir: Path, per_type: int) -> dict:
    if per_type < 1:
        raise ValueError("per_type must be positive")
    result = {"rule": "lowest SHA-256 of prc2025-dev-v1|flight_id within aircraft_type", "per_type": per_type, "splits": {}}
    for split in ("train", "rank"):
        flights = pd.read_parquet(raw_dir / f"flightlist_{split}.parquet", columns=["flight_id", "aircraft_type"])
        if flights.flight_id.duplicated().any():
            raise ValueError(f"Repeated flight IDs in {split}")
        records = []
        for aircraft_type, group in flights.groupby("aircraft_type"):
            ordered = sorted(group.flight_id, key=lambda flight_id: hashlib.sha256(
                f"prc2025-dev-v1|{flight_id}".encode()
            ).hexdigest())
            records.extend({"flight_id": flight_id, "aircraft_type": aircraft_type} for flight_id in ordered[:per_type])
        result["splits"][split] = sorted(records, key=lambda record: record["flight_id"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--per-type", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("configs/dev-flights.json"))
    args = parser.parse_args()
    result = choose(args.raw_dir, args.per_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}: " + ", ".join(f"{s}={len(rows)}" for s, rows in result["splits"].items()))


if __name__ == "__main__":
    main()
