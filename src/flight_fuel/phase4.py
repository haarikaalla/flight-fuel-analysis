"""Frozen September policy development and one-time October interval evaluation.

Run --rank-only first. The default command additionally opens the held-out
October labels after the protocol and software have been checked on September.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from .baselines import design, load_split, paired_flight_bootstrap, score, type_rate_predictions
from .features import FEATURE_COLS, TRAJECTORY_COLS, features_for_flight


def load_final(raw_dir: Path, feature_dir: Path) -> pd.DataFrame:
    """Phase 4's explicit, manifest-gated access to the locked October split."""
    flights = pd.read_parquet(raw_dir / "flightlist_final.parquet", columns=["flight_id", "aircraft_type"])
    labels = pd.read_parquet(raw_dir / "fuel_final.parquet", columns=["idx", "flight_id", "start", "end", "fuel_kg"])
    features = pd.read_parquet(feature_dir / "features_final.parquet")
    if set(features.columns) != {"idx", "flight_id", *FEATURE_COLS}:
        raise ValueError("Unexpected October feature columns")
    frame = labels.merge(flights, on="flight_id", how="left", validate="many_to_one", indicator=True)
    if len(frame) != len(labels) or not frame._merge.eq("both").all():
        raise ValueError("Incomplete October flight metadata")
    frame = frame.drop(columns="_merge").merge(features, on=["idx", "flight_id"], how="left", validate="one_to_one", indicator=True)
    if len(frame) != len(labels) or not frame._merge.eq("both").all():
        raise ValueError("Incomplete October ADS-B feature coverage")
    frame = frame.drop(columns="_merge")
    frame["duration_min"] = (frame.end - frame.start).dt.total_seconds() / 60
    if frame.duration_min.le(0).any() or frame.fuel_kg.isna().any():
        raise ValueError("Invalid October labels or durations")
    return frame


def blocks(rank: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    a = pd.Timestamp(config["risk_train_before"])
    b = pd.Timestamp(config["calibration_before"])
    if not a < b or not (rank.start < a).any() or not (rank.start >= b).any():
        raise ValueError("Invalid September boundaries")
    times = rank.groupby("flight_id").agg(first=("start", "min"), last=("end", "max"))
    crosses = ((times["first"] < a) & (times["last"] >= a)) | ((times["first"] < b) & (times["last"] >= b))
    # Exclude the few flights spanning block boundaries; never divide a flight.
    flight_date = times.loc[~crosses, "first"]
    flight_block = pd.cut(flight_date, [pd.Timestamp("2025-09-01"), a, b, pd.Timestamp("2025-10-01")],
                          labels=["risk_train", "calibration", "validation"], right=False)
    if flight_block.isna().any():
        raise ValueError("Ranking flight outside September")
    result = {key: rank[rank.flight_id.isin(flight_block.index[flight_block == key])].copy()
              for key in ("risk_train", "calibration", "validation")}
    if any(len(x) == 0 for x in result.values()):
        raise ValueError("Empty September block")
    return result


def stable_tie(frame: pd.DataFrame, name: str) -> np.ndarray:
    return np.array([int.from_bytes(hashlib.blake2b(
        f"{name}:{fid}:{idx}".encode(), digest_size=8).digest(), "big") / 2**64
        for fid, idx in zip(frame.flight_id, frame.idx)])


def threshold(values: np.ndarray, tie: np.ndarray, coverage: float) -> tuple[float, float]:
    if not 0 < coverage <= 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid selection scores or coverage")
    if coverage == 1:
        return float(np.finfo(float).max), 1.0  # All later finite scores are accepted.
    order = np.lexsort((tie, values))
    k = max(1, math.ceil(coverage * len(values))) - 1
    return float(values[order[k]]), float(tie[order[k]])


def accepted(values: np.ndarray, tie: np.ndarray, cutoff: tuple[float, float]) -> np.ndarray:
    return (values < cutoff[0]) | ((values == cutoff[0]) & (tie <= cutoff[1]))


def risk_design(reference: pd.DataFrame, frame: pd.DataFrame,
                reference_pred: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    def make(x: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
        numbers = pd.DataFrame({
            "log_duration": np.log1p(x.duration_min.to_numpy()),
            "log_positions": np.log1p(x.adsb_position_count.to_numpy()),
            "log_speeds": np.log1p(x.adsb_speed_count.to_numpy()),
            "max_gap_fraction": x.adsb_max_gap_fraction.to_numpy(),
            "log_predicted_kg": np.log1p(np.maximum(0, p)),
        })
        types = pd.get_dummies(x.aircraft_type.reset_index(drop=True), prefix="type", dtype=float)
        return pd.concat([numbers, types], axis=1)
    first, second = make(reference, reference_pred), make(frame, predicted)
    second = second.reindex(columns=first.columns, fill_value=0)
    return first.to_numpy(dtype=np.float32), second.to_numpy(dtype=np.float32), first.columns.tolist()


def assess(frame: pd.DataFrame, prediction: np.ndarray, mask: np.ndarray) -> dict:
    selected = frame.loc[mask]
    result = {"accepted_fraction": float(mask.mean()), "accepted_intervals": int(mask.sum()),
              "accepted_flights": int(selected.flight_id.nunique())}
    if len(selected):
        result["accepted_score"] = score(selected, prediction[mask])
    group = frame[["aircraft_type", "duration_min", "adsb_position_count"]].copy()
    group["accepted"] = mask
    group["duration_bin"] = pd.cut(group.duration_min, [0, 5, 15, 60, float("inf")],
                                    labels=["<5 min", "5–15 min", "15–60 min", "60+ min"], right=False)
    group["adsb_input"] = np.where(group.adsb_position_count.eq(0), "zero ADS-B", "observed ADS-B")
    result["acceptance_by_group"] = {
        field: {str(key): {"n": len(part), "accepted_fraction": float(part.accepted.mean())}
                for key, part in group.groupby(field, observed=True)}
        for field in ("aircraft_type", "duration_bin", "adsb_input")}
    return result


def label_audit(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    # Diagnostic only: the quantization flag is never an estimator input.
    pounds = frame.fuel_kg.to_numpy() / 0.453592
    quantized = np.isclose(pounds, np.rint(pounds), rtol=0, atol=1e-5)
    label = pd.Series(np.where(quantized, "integer-pound lattice", "other"), index=frame.index)
    result = {}
    for key in ("all", "duration_bin", "aircraft_type"):
        bins = (pd.Series("all", index=frame.index) if key == "all" else
                pd.cut(frame.duration_min, [0, 5, 15, 60, float("inf")],
                       labels=["<5 min", "5–15 min", "15–60 min", "60+ min"], right=False).astype(str)
                if key == "duration_bin" else frame.aircraft_type)
        result[key] = {}
        for value in sorted(bins.unique()):
            group = bins.eq(value).to_numpy()
            result[key][str(value)] = {"n": int(group.sum()), "lattice_fraction": float(quantized[group].mean()),
                                       "lattice_mae_kg": float(np.abs(prediction[group & quantized] - frame.fuel_kg.to_numpy()[group & quantized]).mean())
                                       if (group & quantized).any() else None,
                                       "other_mae_kg": float(np.abs(prediction[group & ~quantized] - frame.fuel_kg.to_numpy()[group & ~quantized]).mean())
                                       if (group & ~quantized).any() else None}
    return result


def outage_experiment(raw_dir: Path, validation: pd.DataFrame, model: HistGradientBoostingRegressor,
                      train: pd.DataFrame, config: dict) -> dict:
    counts = validation.groupby("flight_id").adsb_position_count.sum()
    eligible = sorted(counts.index[counts >= config["min_natural_adsb_positions"]])
    rng = np.random.default_rng(42)
    chosen = sorted(rng.choice(eligible, size=min(config["validation_flights"], len(eligible)), replace=False))
    subset = validation[validation.flight_id.isin(chosen) & validation.adsb_position_count.gt(0)].copy()
    if not len(subset):
        raise ValueError("No observable outage intervals")
    baseline = np.maximum(0, model.predict(design(train, subset, True)[1]))
    natural = score(subset, baseline)
    raw_parts = {(fraction, seed, method): [] for fraction in config["removal_fractions"]
                 for seed in config["seeds"] for method in ("random points", "contiguous point block")}
    by_flight = dict(tuple(subset.groupby("flight_id", sort=False)))
    with zipfile.ZipFile(raw_dir / "flights_rank.zip") as archive:
        member = {name.rsplit("/", 1)[-1].removesuffix(".parquet"): name for name in archive.namelist()
                  if name.startswith("flights_rank/") and name.endswith(".parquet")}
        for fid in chosen:
            with archive.open(member[fid]) as handle:
                trajectory = pd.read_parquet(io.BytesIO(handle.read()), columns=TRAJECTORY_COLS)
            original = features_for_flight(trajectory, by_flight[fid], fid).set_index("idx")
            cached = by_flight[fid].set_index("idx")[FEATURE_COLS]
            if not np.allclose(original.loc[cached.index, FEATURE_COLS].to_numpy(dtype=float),
                               cached.to_numpy(dtype=float), equal_nan=True):
                raise ValueError(f"Original features disagree with verified cache for {fid}")
            valid = trajectory.source.eq("adsb") & trajectory.timestamp.notna() & trajectory.latitude.notna() & trajectory.longitude.notna()
            positions = np.flatnonzero(valid.to_numpy())
            for fraction, seed, method in raw_parts:
                n_remove = int(round(fraction * len(positions)))
                local_rng = np.random.default_rng(int.from_bytes(hashlib.blake2b(
                    f"{fid}:{seed}:{fraction}:{method}".encode(), digest_size=8).digest(), "big"))
                if method == "random points":
                    removed = local_rng.choice(positions, size=n_remove, replace=False)
                else:
                    start = local_rng.integers(0, len(positions) - n_remove + 1)
                    removed = positions[start:start + n_remove]
                perturbed = trajectory.drop(index=trajectory.index[removed])
                raw_parts[(fraction, seed, method)].append(features_for_flight(perturbed, by_flight[fid], fid))
    scenarios = {}
    for (fraction, seed, method), parts in raw_parts.items():
        changed = pd.concat(parts, ignore_index=True).set_index("idx").loc[subset.idx].reset_index(drop=True)
        replacement = subset.drop(columns=FEATURE_COLS).reset_index(drop=True).join(changed[FEATURE_COLS])
        predicted = np.maximum(0, model.predict(design(train, replacement, True)[1]))
        type_rate = type_rate_predictions(train, replacement)
        fallback = np.where(replacement.adsb_position_count.eq(0), type_rate, predicted)
        key = f"{method} / {round(100 * fraction)}% / seed {seed}"
        scenarios[key] = {"adsb": score(subset, predicted), "fallback": score(subset, fallback),
                          "zero_adsb_fraction": float(replacement.adsb_position_count.eq(0).mean()),
                          "adsb_vs_natural": paired_flight_bootstrap(subset, baseline, predicted, 500)}
    return {"sample_rule": "seeded flight sample from September validation with >=20 original ADS-B positions; score only originally observed intervals",
            "flights": len(chosen), "intervals": len(subset), "natural_adsb": natural, "scenarios": scenarios}


def run(raw_dir: Path, feature_dir: Path, config: dict, *, rank_only: bool = False,
        outages: bool = True) -> dict:
    manifest = json.loads((feature_dir / "feature_manifest.json").read_text())
    needed = ("train", "rank") if rank_only else ("train", "rank", "final")
    if any(s not in manifest or manifest[s]["subset"] for s in needed):
        raise ValueError("Full verified feature manifests required for all evaluated periods")
    train = load_split(raw_dir, "train", feature_dir)
    rank = load_split(raw_dir, "rank", feature_dir)
    if len(train) != manifest["train"]["intervals"] or len(rank) != manifest["rank"]["intervals"]:
        raise ValueError("Train/rank manifest interval count mismatch")
    if set(train.flight_id) & set(rank.flight_id):
        raise ValueError("Train/rank flight overlap")
    pieces = blocks(rank, config)
    excluded = rank[~rank.flight_id.isin(pd.concat(pieces.values()).flight_id)]
    periods = {"risk_train": pieces["risk_train"], "calibration": pieces["calibration"],
               "validation": pieces["validation"]}
    if not rank_only:
        periods["final"] = load_final(raw_dir, feature_dir)
    if not rank_only and len(periods["final"]) != manifest["final"]["intervals"]:
        raise ValueError("October manifest interval count mismatch")
    if not rank_only and set(rank.flight_id) & set(periods["final"].flight_id):
        raise ValueError("Ranking and final flights overlap")
    model = HistGradientBoostingRegressor(**json.loads(Path(config["base_config"]).read_text())["hist_gradient_boosting"])
    x_train, _, columns = design(train, pieces["risk_train"], True)
    model.fit(x_train, train.fuel_kg.to_numpy())
    predictions = {}
    for name, frame in periods.items():
        adsb = np.maximum(0, model.predict(design(train, frame, True)[1]))
        type_rate = type_rate_predictions(train, frame)
        predictions[name] = {"adsb": adsb, "type_rate": type_rate,
                             "fallback": np.where(frame.adsb_position_count.eq(0), type_rate, adsb)}
    ref = pieces["risk_train"]
    ref_prediction = predictions["risk_train"]["fallback"]
    risk_x, _, risk_columns = risk_design(ref, ref, ref_prediction, ref_prediction)
    risk_model = HistGradientBoostingRegressor(**config["risk_hist_gradient_boosting"])
    risk_model.fit(risk_x, np.abs(ref_prediction - ref.fuel_kg.to_numpy()))
    risk_scores = {}
    for name, frame in periods.items():
        _, x, _ = risk_design(ref, frame, ref_prediction, predictions[name]["fallback"])
        risk_scores[name] = np.maximum(0, risk_model.predict(x))
    selectors = {
        "learned_risk": lambda name: risk_scores[name],
        "max_gap_fraction": lambda name: periods[name].adsb_max_gap_fraction.to_numpy(),
        "duration": lambda name: periods[name].duration_min.to_numpy(),
        "random_hash": lambda name: stable_tie(periods[name], "random_score"),
    }
    cutoffs = {selector: {str(c): threshold(fn("calibration"), stable_tie(pieces["calibration"], selector), c)
                          for c in config["target_coverages"]}
               for selector, fn in selectors.items()}
    output = {"protocol": config, "versions": {"numpy": np.__version__, "pandas": pd.__version__,
               "scikit_learn": sklearn.__version__}, "feature_manifest": {s: manifest[s] for s in needed},
              "base_columns": columns, "risk_columns": risk_columns,
              "boundary_exclusions": {"flights": int(excluded.flight_id.nunique()), "intervals": len(excluded)},
              "block_counts": {name: {"intervals": len(frame), "flights": int(frame.flight_id.nunique()),
                                      "first_start": str(frame.start.min()), "last_start": str(frame.start.max())}
                               for name, frame in periods.items()},
              "selection_thresholds": cutoffs, "periods": {}}
    for name in ("calibration", "validation", "final"):
        if name not in periods:
            continue
        frame = periods[name]
        p = predictions[name]
        selected = {}
        for selector, fn in selectors.items():
            values, ties = fn(name), stable_tie(frame, selector)
            selected[selector] = {str(c): assess(frame, p["fallback"],
                accepted(values, ties, cutoffs[selector][str(c)])) for c in config["target_coverages"]}
        output["periods"][name] = {
            "models": {key: score(frame, pred) for key, pred in p.items()},
            "fallback_vs_adsb": paired_flight_bootstrap(frame, p["adsb"], p["fallback"]),
            "selectors": selected,
            "label_granularity": label_audit(frame, p["fallback"]),
        }
    if outages:
        output["outage_experiment"] = outage_experiment(raw_dir, pieces["validation"], model, train, config["outage"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("configs/phase-4-protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/phase-4-results.json"))
    parser.add_argument("--rank-only", action="store_true")
    parser.add_argument("--skip-outages", action="store_true", help="Fast smoke run; omit the controlled outage experiment")
    args = parser.parse_args()
    result = run(args.raw_dir, args.feature_dir, json.loads(args.config.read_text()),
                 rank_only=args.rank_only, outages=not args.skip_outages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for period, data in result["periods"].items():
        print(period, {key: round(value["mae_kg"], 2) for key, value in data["models"].items()},
              "75% risk acceptance", round(data["selectors"]["learned_risk"]["0.75"]["accepted_fraction"], 3), flush=True)


if __name__ == "__main__":
    main()
