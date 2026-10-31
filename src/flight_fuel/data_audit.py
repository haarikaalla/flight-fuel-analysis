"""Reproducible, metadata-only dataset acceptance audit.

No model is fitted. All counts are computed independently per published split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

RECORD = "https://zenodo.org/api/records/19184662/files"
MD5 = {
    "readme.txt": "bb62f84f63c10b08d4381e82f83abe33",
    "flightlist_train.parquet": "1c7b9c86aa9e366236828e2439184d37",
    "flightlist_rank.parquet": "a5285c9f50171d376dabb478f9d286c3",
    "flightlist_final.parquet": "c70836916aa5d27fb792a4391f5ac339",
    "fuel_train.parquet": "4ff4df3aca1a2d0850f48ef08395f1f9",
    "fuel_rank.parquet": "97f44f189f41df4957338b20e48f647c",
    "fuel_final.parquet": "90aff13775122bacbe0054040d7e83b6",
    "airports.parquet": "c0f2d599ee9989385e27d28d18c9ac75",
}
SPLITS = ("train", "rank", "final")
POUND_KG = 0.453592


def checksum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_metadata(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in MD5.items():
        path = directory / name
        if not path.exists() or checksum(path) != expected:
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                with urlopen(f"{RECORD}/{name}/content", timeout=120) as source, temporary.open("wb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
                if checksum(temporary) != expected:
                    raise ValueError(f"MD5 mismatch for {name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Verified {name}")


def audit(directory: Path) -> dict:
    verified = {}
    for name, expected in MD5.items():
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run with --download")
        actual = checksum(path)
        if actual != expected:
            raise ValueError(f"MD5 mismatch for {path}: {actual}")
        verified[name] = {"bytes": path.stat().st_size, "md5": actual}

    results: dict = {"dataset_doi": "10.5281/zenodo.19184662", "verified_files": verified, "splits": {}}
    flight_sets = {}
    type_sets = {}
    for split in SPLITS:
        flights = pd.read_parquet(directory / f"flightlist_{split}.parquet")
        fuel = pd.read_parquet(directory / f"fuel_{split}.parquet")
        for field in ("flight_id", "aircraft_type", "flight_date"):
            if field not in flights:
                raise ValueError(f"Missing flight-list field: {field}")
        for field in ("idx", "flight_id", "start", "end", "fuel_kg"):
            if field not in fuel:
                raise ValueError(f"Missing fuel field: {field}")
        if flights.flight_id.isna().any() or flights.flight_id.duplicated().any():
            raise ValueError(f"Null or repeated flight ID in {split}")
        if fuel.idx.isna().any() or fuel.idx.duplicated().any():
            raise ValueError(f"Null or repeated interval index in {split}")
        flight_sets[split] = set(flights.flight_id)
        type_sets[split] = set(flights.aircraft_type)
        duration = (fuel.end - fuel.start).dt.total_seconds() / 60
        pounds = fuel.fuel_kg / POUND_KG
        # Diagnostic only: integer-lattice matches do not prove the original unit.
        pound_lattice = (pounds - pounds.round()).abs() <= 1e-6
        results["splits"][split] = {
            "flights": len(flights), "intervals": len(fuel),
            "labeled_flights": int(fuel.flight_id.nunique()),
            "orphan_intervals": int((~fuel.flight_id.isin(flights.flight_id)).sum()),
            "duplicate_indices": int(fuel.idx.duplicated().sum()),
            "null_targets": int(fuel.fuel_kg.isna().sum()),
            "nonpositive_targets": int((fuel.fuel_kg <= 0).sum()),
            "nonpositive_durations": int((duration <= 0).sum()),
            "under_1_min": int((duration < 1).sum()),
            "under_5_min": int((duration < 5).sum()),
            "over_60_min": int((duration > 60).sum()),
            "duration_min_quantiles": {str(q): float(duration.quantile(q)) for q in (0, .5, .99, 1)},
            "fuel_kg_quantiles": {str(q): float(fuel.fuel_kg.quantile(q)) for q in (0, .5, .99, 1)},
            "integer_pound_lattice_count": int(pound_lattice.sum()),
            "exact_200_kg_count": int((fuel.fuel_kg == 200).sum()),
            "first_date": str(flights.flight_date.min()),
            "last_date": str(flights.flight_date.max()),
            "aircraft_types": flights.aircraft_type.value_counts().sort_index().to_dict(),
            "flight_columns": {c: str(t) for c, t in flights.dtypes.items()},
            "fuel_columns": {c: str(t) for c, t in fuel.dtypes.items()},
        }
    results["cross_split"] = {
        "overlapping_flight_ids": {
            f"{a}_{b}": len(flight_sets[a] & flight_sets[b])
            for a, b in (("train", "rank"), ("train", "final"), ("rank", "final"))
        },
        "test_only_aircraft_types": sorted(type_sets["final"] - type_sets["train"]),
        "all_aircraft_type_count": len(set.union(*type_sets.values())),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("reports/phase-2-metadata-audit.json"))
    parser.add_argument("--download", action="store_true", help="Fetch the seven small metadata/label files")
    args = parser.parse_args()
    if args.download:
        fetch_metadata(args.raw_dir)
    result = audit(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
