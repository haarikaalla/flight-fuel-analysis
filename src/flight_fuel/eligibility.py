"""Exploratory batch eligibility audit for retrospective fuel estimates.

The April--August fuel model and early-September residual model are fixed to
the published Phase 4 architecture. This module was designed after October
results were examined: October comparisons here are exploratory, not a new
untouched test. Selection uses only predictions, scores and group metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from .baselines import design, load_split, score, type_rate_predictions
from .phase4 import blocks, load_final, risk_design, stable_tie


DURATION_LABELS = ("<5 min", "5–15 min", "15–60 min", "60+ min")


def duration_groups(duration: np.ndarray) -> np.ndarray:
    values = np.asarray(duration, dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("All interval durations must be finite and positive")
    return np.asarray(DURATION_LABELS, dtype=object)[np.searchsorted([5, 15, 60], values, side="right")]


def select(scores: np.ndarray, types: np.ndarray, duration: np.ndarray, ties: np.ndarray,
           target: float, floor: float, min_type_n: int = 50) -> tuple[np.ndarray, dict]:
    """Select low-risk rows, serving group floors before filling the budget.

    Duration groups are served first, then eligible aircraft types sorted by
    increasing group size. If floors exceed the target, keep the floors and
    disclose the excess. There is no claim of globally optimal risk.
    """
    scores, types, duration, ties = (np.asarray(x) for x in (scores, types, duration, ties))
    n = len(scores)
    if not n or any(len(x) != n for x in (types, duration, ties)):
        raise ValueError("Nonempty, equally sized arrays required")
    if not np.isfinite(scores).all() or not np.isfinite(ties).all():
        raise ValueError("Scores and ties must be finite")
    if not (0 < target <= 1 and 0 <= floor <= 1 and min_type_n >= 1):
        raise ValueError("Invalid target, floor or minimum aircraft group size")
    groups = duration_groups(duration)
    order = np.lexsort((ties, scores))
    selected = np.zeros(n, dtype=bool)
    type_names, type_counts = np.unique(types.astype(str), return_counts=True)
    protected = sorted(((name, int(count)) for name, count in zip(type_names, type_counts)
                        if count >= min_type_n), key=lambda item: (item[1], item[0]))
    def serve(members: np.ndarray) -> None:
        quota = math.ceil(floor * int(members.sum()))
        deficit = max(0, quota - int((selected & members).sum()))
        if deficit:
            candidates = order[members[order] & ~selected[order]]
            selected[candidates[:deficit]] = True

    for name in DURATION_LABELS:
        serve(groups == name)
    for name, _ in protected:
        serve(types.astype(str) == name)
    budget = math.ceil(target * n)
    if selected.sum() < budget:
        selected[order[~selected[order]][:budget - int(selected.sum())]] = True
    return selected, {"target_count": budget, "floor_forced_over_budget": bool(selected.sum() > budget),
                      "protected_aircraft_types": [name for name, _ in protected],
                      "unprotected_aircraft_types": [str(name) for name, count in zip(type_names, type_counts)
                                                     if count < min_type_n]}


def describe(frame: pd.DataFrame, predicted: np.ndarray, risk: np.ndarray,
             mask: np.ndarray, policy: dict) -> dict:
    if not (len(frame) == len(predicted) == len(risk) == len(mask)):
        raise ValueError("Prediction, risk, selection and frame lengths differ")
    truth = frame.fuel_kg.to_numpy()
    result = {"policy": policy, "n": len(frame), "accepted": int(mask.sum()),
              "coverage": float(mask.mean()), "all_mae_kg": float(np.abs(predicted - truth).mean()),
              "accepted_mae_kg": float(np.abs(predicted[mask] - truth[mask]).mean()) if mask.any() else None,
              "rejected_mae_kg": float(np.abs(predicted[~mask] - truth[~mask]).mean()) if (~mask).any() else None,
              "accepted_reference_fuel_fraction": float(truth[mask].sum() / truth.sum()),
              "accepted_flights": int(frame.loc[mask, "flight_id"].nunique())}
    groups = {"duration": duration_groups(frame.duration_min.to_numpy()),
              "aircraft_type": frame.aircraft_type.to_numpy(dtype=str),
              "adsb_input": np.where(frame.adsb_position_count.eq(0), "zero", "observed")}
    result["groups"] = {}
    for label, values in groups.items():
        result["groups"][label] = {}
        for name in sorted(set(values)):
            subset = values == name
            accept = subset & mask
            result["groups"][label][name] = {
                "n": int(subset.sum()), "accepted": int(accept.sum()),
                "coverage": float(accept.sum() / subset.sum()),
                "all_mae_kg": float(np.abs(predicted[subset] - truth[subset]).mean()),
                "accepted_mae_kg": float(np.abs(predicted[accept] - truth[accept]).mean()) if accept.any() else None,
                "reference_fuel_fraction": float(truth[subset].sum() / truth.sum()),
                "accepted_reference_fuel_fraction": float(truth[accept].sum() / truth[subset].sum()),
            }
    return result


def fit_and_predict(train: pd.DataFrame, rank: pd.DataFrame, final: pd.DataFrame | None,
                    config: dict) -> dict:
    pieces = blocks(rank, config)
    if len(set(train.flight_id) & set(rank.flight_id)):
        raise ValueError("Training and September flights overlap")
    if final is not None and len(set(final.flight_id) & set(rank.flight_id)):
        raise ValueError("September and October flights overlap")
    base_config = json.loads(Path(config["base_config"]).read_text())
    base = HistGradientBoostingRegressor(**base_config["hist_gradient_boosting"])
    base.fit(design(train, pieces["risk_train"], True)[0], train.fuel_kg.to_numpy())
    frames = {"risk_train": pieces["risk_train"], "calibration": pieces["calibration"],
              "validation": pieces["validation"]}
    if final is not None:
        frames["october_exploratory"] = final
    predicted = {}
    for key, frame in frames.items():
        adsb = np.maximum(0, base.predict(design(train, frame, True)[1]))
        predicted[key] = np.where(frame.adsb_position_count.eq(0), type_rate_predictions(train, frame), adsb)
    reference = frames["risk_train"]
    reference_x, _, _ = risk_design(reference, reference, predicted["risk_train"], predicted["risk_train"])
    risk_model = HistGradientBoostingRegressor(**config["risk_hist_gradient_boosting"])
    risk_model.fit(reference_x, np.abs(predicted["risk_train"] - reference.fuel_kg.to_numpy()))
    risks = {}
    for key, frame in frames.items():
        _, x, _ = risk_design(reference, frame, predicted["risk_train"], predicted[key])
        risks[key] = np.maximum(0, risk_model.predict(x))
    return {"frames": frames, "predicted": predicted, "risks": risks,
            "excluded_boundary_intervals": int(len(rank) - sum(map(len, pieces.values())))}


def evaluate(fitted: dict, *, target: float = .75, floor: float = .5,
             min_type_n: int = 50) -> dict:
    results = {}
    for key in ("calibration", "validation", "october_exploratory"):
        if key not in fitted["frames"]:
            continue
        frame, pred, risk = fitted["frames"][key], fitted["predicted"][key], fitted["risks"][key]
        ties = stable_tie(frame, "eligibility")
        policies = {}
        for name, group_floor in (("risk_only", 0), ("representation_floor", floor)):
            mask, details = select(risk, frame.aircraft_type.to_numpy(), frame.duration_min.to_numpy(),
                                   ties, target, group_floor, min_type_n)
            policies[name] = describe(frame, pred, risk, mask, {"floor": group_floor, **details})
        results[key] = {"intervals": len(frame), "flights": int(frame.flight_id.nunique()),
                        "fallback": score(frame, pred), "policies": policies}
    return results


def demo_rows(fitted: dict, key: str = "validation") -> list[list]:
    frame, pred, risk = (fitted[name][key] for name in ("frames", "predicted", "risks"))
    ties = stable_tie(frame, "eligibility")
    return [[str(t), float(d), float(s), float(p), float(y), int(a), float(h)]
            for t, d, s, p, y, a, h in zip(frame.aircraft_type, frame.duration_min, risk,
                                           pred, frame.fuel_kg, frame.adsb_position_count, ties)]


def render(template: str, payload: dict) -> str:
    marker = "/*__AUDIT_DATA__*/"
    if template.count(marker) != 1:
        raise ValueError("Expected one dashboard data marker")
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return template.replace(marker, "const DATA=" + serialized.replace("<", "\\u003c") + ";")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("configs/phase-4-protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/eligibility-results.json"))
    parser.add_argument("--template", type=Path, default=Path("demo/template.html"))
    parser.add_argument("--demo-output", type=Path, default=Path("demo/index.html"))
    parser.add_argument("--skip-october", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.feature_dir / "feature_manifest.json").read_text())
    required = ("train", "rank") if args.skip_october else ("train", "rank", "final")
    if any(s not in manifest or manifest[s]["subset"] for s in required):
        raise ValueError("Complete verified feature archives required")
    train, rank = (load_split(args.raw_dir, s, args.feature_dir) for s in ("train", "rank"))
    final = None if args.skip_october else load_final(args.raw_dir, args.feature_dir)
    for s, frame in zip(required, (train, rank, final)):
        if len(frame) != manifest[s]["intervals"]:
            raise ValueError(f"Manifest count mismatch: {s}")
    fitted = fit_and_predict(train, rank, final, json.loads(args.config.read_text()))
    results = {"status": "Exploratory: policy chosen after viewing October Phase 4 results",
               "decision": "Batch eligibility for retrospective interval estimates, with group access floors",
               "selection_inputs": "Predicted absolute-error rank, aircraft type, duration and stable ID tie; no fuel label",
               "floor_definition": "At least 50% of each nonempty duration bin and each aircraft type with >=50 intervals per batch; small aircraft groups are reported but not protected",
               "coverage_target": .75, "min_aircraft_group_n": 50,
               "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
               "feature_manifest": {s: manifest[s] for s in required},
               "excluded_boundary_intervals": fitted["excluded_boundary_intervals"],
               "periods": evaluate(fitted)}
    historical = Path("reports/phase-4-results.json")
    if historical.exists():
        previous = json.loads(historical.read_text())["periods"]
        for new, old in (("validation", "validation"), ("october_exploratory", "final")):
            if new in results["periods"] and abs(results["periods"][new]["fallback"]["mae_kg"] -
                                                  previous[old]["models"]["fallback"]["mae_kg"]) > 1e-6:
                raise ValueError(f"Fuel model reconstruction disagrees with historical Phase 4: {new}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    payload = {"rows": demo_rows(fitted), "period": "September 19–27, 2025 validation",
               "reference": {key: results["periods"]["validation"]["policies"][key]
                             for key in ("risk_only", "representation_floor")}}
    args.demo_output.parent.mkdir(parents=True, exist_ok=True)
    args.demo_output.write_text(render(args.template.read_text(), payload))
    for key, period in results["periods"].items():
        print(key, {policy: {"coverage": round(value["coverage"], 4),
                             "accepted_mae_kg": round(value["accepted_mae_kg"], 2),
                             "fuel_share": round(value["accepted_reference_fuel_fraction"], 4)}
                    for policy, value in period["policies"].items()}, flush=True)


if __name__ == "__main__":
    main()
