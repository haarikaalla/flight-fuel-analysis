"""Smoke-test the journal study on a tiny self-consistent subset (not evidence)."""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

from research_q1.journal import study

ROOT = Path(__file__).resolve().parents[2]


def make_subset(raw: Path, smoke_features: Path, out: Path, flights_per_split: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    (out / "processed").mkdir(exist_ok=True)
    manifest = {}
    tables = {}
    for split in ("rank", "final"):
        n = flights_per_split * 2 if split == "rank" else flights_per_split
        tables[split] = (pd.read_parquet(smoke_features / f"kinematics_{split}_first_{n}.parquet"),
                         pd.read_parquet(smoke_features / f"features_{split}_first_{n}.parquet"), split)
    # No training archive in the smoke set: use the second half of the September subset as "train".
    kin, feat, _ = tables["rank"]
    first = set(sorted(kin.flight_id.unique())[:flights_per_split])
    tables["train"] = (kin[~kin.flight_id.isin(first)], feat[~feat.flight_id.isin(first)], "rank")
    tables["rank"] = (kin[kin.flight_id.isin(first)], feat[feat.flight_id.isin(first)], "rank")
    for split, (kin, feat, source) in tables.items():
        flights = pd.read_parquet(raw / f"flightlist_{source}.parquet")
        fuel = pd.read_parquet(raw / f"fuel_{source}.parquet")
        keep = set(kin.flight_id)
        flights[flights.flight_id.isin(keep)].to_parquet(out / "raw" / f"flightlist_{split}.parquet", index=False)
        fuel[fuel.flight_id.isin(keep)].to_parquet(out / "raw" / f"fuel_{split}.parquet", index=False)
        kin.to_parquet(out / "processed" / f"kinematics_{split}.parquet", index=False)
        feat.to_parquet(out / "processed" / f"features_{split}.parquet", index=False)
        manifest[split] = {"subset": False, "intervals": len(kin)}
    pd.read_parquet(raw / "airports.parquet").to_parquet(out / "raw" / "airports.parquet", index=False)
    (out / "processed" / "feature_manifest.json").write_text(json.dumps(manifest))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    tmp = Path(tempfile.mkdtemp())
    make_subset(ROOT / "data/raw", ROOT / "data/smoke", tmp, n)
    cfg = json.loads((ROOT / "research_q1/configs/journal_study.json").read_text())
    cfg["bootstrap_repetitions"] = 50
    cfg["hist_gradient_boosting"]["max_iter"] = 20
    cfg["risk_hist_gradient_boosting"]["max_iter"] = 20
    cfg["loto_min_intervals"] = 50
    cfg_path = tmp / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    result = study.run(tmp / "raw", tmp / "processed", cfg_path)
    (tmp / "out.json").write_text(json.dumps(result, indent=1, allow_nan=False))
    print("smoke OK ->", tmp / "out.json")
