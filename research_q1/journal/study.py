"""Journal study runner: physics-informed hybrid fuel model, uncertainty and representation audit.

Every model is fit on April--August 2025 only. September 2025 provides risk
fitting (flight half A) and conformal calibration (flight half B). October 2025
is a forward temporal test. See ``research_q1/configs/journal_study.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from research_q1.journal.extract import KINEMATIC_COLS
from research_q1.novel.conformal import HierarchicalMondrian, finite_quantile
from research_q1.stats.inference import adjust_pvalues
from research_q1.validation.metrics import interval_metrics
from src.flight_fuel.eligibility import DURATION_LABELS, duration_groups, select
from src.flight_fuel.features import FEATURE_COLS
from src.flight_fuel.phase4 import stable_tie

ROOT = Path(__file__).resolve().parents[2]
EARTH_RADIUS_KM = 6371.0088

META_NUMERIC = ["duration_min", "flight_duration_min", "elapsed_from_takeoff_min",
                "remaining_to_landing_min", "start_hour_sin", "start_hour_cos", "route_gc_km"]
PHYS_COLS = ["phys_openap_kg", "phys_frac_ground", "phys_max_interp_gap_s"]
KIN_COLS = [c for c in KINEMATIC_COLS if c not in PHYS_COLS]
GROUPS = {"meta": META_NUMERIC, "adsb": list(FEATURE_COLS), "kin": KIN_COLS, "phys": PHYS_COLS}
MODEL_FEATURES = {
    "hgb_metadata": ["meta"],
    "hgb_metadata_adsb": ["meta", "adsb"],
    "hgb_full": ["meta", "adsb", "kin", "phys"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------------- data

def load_period(raw_dir: Path, feature_dir: Path, split: str) -> pd.DataFrame:
    flights = pd.read_parquet(raw_dir / f"flightlist_{split}.parquet")
    labels = pd.read_parquet(raw_dir / f"fuel_{split}.parquet")
    frame = labels.merge(flights, on="flight_id", how="left", validate="many_to_one", indicator=True)
    if len(frame) != len(labels) or not frame._merge.eq("both").all():
        raise ValueError(f"Incomplete flight metadata in {split}")
    frame = frame.drop(columns="_merge")
    for name in ("features", "kinematics"):
        table = pd.read_parquet(feature_dir / f"{name}_{split}.parquet")
        frame = frame.merge(table, on=["idx", "flight_id"], how="left", validate="one_to_one", indicator=True)
        if len(frame) != len(labels) or not frame._merge.eq("both").all():
            raise ValueError(f"Incomplete {name} for {split}")
        frame = frame.drop(columns="_merge")
    airports = pd.read_parquet(raw_dir / "airports.parquet").set_index("icao")
    o = airports.loc[frame.origin_icao, ["latitude", "longitude"]].to_numpy(float)
    d = airports.loc[frame.destination_icao, ["latitude", "longitude"]].to_numpy(float)
    p1, p2 = np.radians(o[:, 0]), np.radians(d[:, 0])
    dl = np.radians(d[:, 1] - o[:, 1])
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    frame["route_gc_km"] = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    frame["duration_min"] = (frame.end - frame.start).dt.total_seconds() / 60
    frame["flight_duration_min"] = (frame.landed - frame.takeoff).dt.total_seconds() / 60
    frame["elapsed_from_takeoff_min"] = (frame.start - frame.takeoff).dt.total_seconds() / 60
    frame["remaining_to_landing_min"] = (frame.landed - frame.end).dt.total_seconds() / 60
    frame["start_hour_sin"] = np.sin(2 * np.pi * frame.start.dt.hour / 24)
    frame["start_hour_cos"] = np.cos(2 * np.pi * frame.start.dt.hour / 24)
    frame["route_id"] = frame.origin_icao.astype(str) + "-" + frame.destination_icao.astype(str)
    if frame.duration_min.le(0).any() or frame.fuel_kg.isna().any() or frame.fuel_kg.le(0).any():
        raise ValueError(f"Invalid labels or durations in {split}")
    return frame.reset_index(drop=True)


def design(train: pd.DataFrame, frame: pd.DataFrame, groups: list[str]) -> tuple[np.ndarray, list[str]]:
    numeric = [c for g in groups for c in GROUPS[g]]
    types = pd.get_dummies(train.aircraft_type, prefix="type", dtype=float).columns
    dummies = pd.get_dummies(frame.aircraft_type, prefix="type", dtype=float).reindex(columns=types, fill_value=0)
    table = pd.concat([frame[numeric].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return table.to_numpy(dtype=np.float32), table.columns.tolist()


# ----------------------------------------------------------------------------- models

def type_rate(train: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    rates = train.fuel_kg / train.duration_min
    mapping = rates.groupby(train.aircraft_type).median()
    rate = frame.aircraft_type.map(mapping).fillna(float(rates.median())).to_numpy()
    return np.maximum(0.0, rate * frame.duration_min.to_numpy())


def physics_scaling(train: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Per-type least-squares scale of OpenAP fuel to labels (through the origin)."""
    ok = train.phys_openap_kg.notna() & (train.phys_openap_kg > 0)
    sub = train.loc[ok]
    num = (sub.fuel_kg * sub.phys_openap_kg).groupby(sub.aircraft_type).sum()
    den = (sub.phys_openap_kg ** 2).groupby(sub.aircraft_type).sum()
    scales = (num / den).to_dict()
    global_scale = float((sub.fuel_kg * sub.phys_openap_kg).sum() / (sub.phys_openap_kg ** 2).sum())
    return scales, global_scale


def physics_predictions(train: pd.DataFrame, frame: pd.DataFrame, *, calibrated: bool,
                        scales: dict | None = None, global_scale: float | None = None) -> np.ndarray:
    phys = frame.phys_openap_kg.to_numpy(float)
    if calibrated:
        scale = frame.aircraft_type.map(scales).fillna(global_scale).to_numpy(float)
        phys = phys * scale
    return np.where(np.isfinite(phys), np.maximum(phys, 0.0), type_rate(train, frame))


def fit_hgb(x: np.ndarray, y: np.ndarray, params: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**params).fit(x, y)


def fit_all_models(train: pd.DataFrame, frames: dict[str, pd.DataFrame], cfg: dict,
                   *, exclude_type: str | None = None) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Return predictions[model][period] and fit audit. Optionally drop one aircraft type from training."""
    fit_frame = train if exclude_type is None else train[train.aircraft_type != exclude_type].reset_index(drop=True)
    params = cfg["hist_gradient_boosting"]
    scales, global_scale = physics_scaling(fit_frame)
    predictions: dict[str, dict[str, np.ndarray]] = {m: {} for m in (
        "type_median_rate", "openap_physics", "openap_calibrated", *MODEL_FEATURES, "hgb_residual_physics")}
    audit = {"fit_seconds": {}, "train_rows": len(fit_frame)}
    for period, frame in frames.items():
        predictions["type_median_rate"][period] = type_rate(fit_frame, frame)
        predictions["openap_physics"][period] = physics_predictions(fit_frame, frame, calibrated=False)
        predictions["openap_calibrated"][period] = physics_predictions(
            fit_frame, frame, calibrated=True, scales=scales, global_scale=global_scale)
    y = fit_frame.fuel_kg.to_numpy(float)
    for name, groups in MODEL_FEATURES.items():
        tick = time.perf_counter()
        x_train, columns = design(fit_frame, fit_frame, groups)
        model = fit_hgb(x_train, y, params)
        audit["fit_seconds"][name] = time.perf_counter() - tick
        audit.setdefault("columns", {})[name] = columns
        for period, frame in frames.items():
            predictions[name][period] = np.maximum(0.0, model.predict(design(fit_frame, frame, groups)[0]))
    tick = time.perf_counter()
    prior_train = physics_predictions(fit_frame, fit_frame, calibrated=True, scales=scales, global_scale=global_scale)
    x_train, _ = design(fit_frame, fit_frame, MODEL_FEATURES["hgb_full"])
    residual_model = fit_hgb(x_train, y - prior_train, params)
    audit["fit_seconds"]["hgb_residual_physics"] = time.perf_counter() - tick
    for period, frame in frames.items():
        prior = physics_predictions(fit_frame, frame, calibrated=True, scales=scales, global_scale=global_scale)
        predictions["hgb_residual_physics"][period] = np.maximum(
            0.0, prior + residual_model.predict(design(fit_frame, frame, MODEL_FEATURES["hgb_full"])[0]))
    audit["physics_scales"] = {k: float(v) for k, v in sorted(scales.items())}
    audit["physics_global_scale"] = global_scale
    return predictions, audit


# ----------------------------------------------------------------------------- statistics

def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    error = p - y
    ae = np.abs(error)
    var = float(np.square(y - y.mean()).sum())
    return {"n": int(len(y)), "mae_kg": float(ae.mean()), "rmse_kg": float(np.sqrt(np.mean(error ** 2))),
            "bias_kg": float(error.mean()), "median_ae_kg": float(np.median(ae)),
            "p95_ae_kg": float(np.quantile(ae, .95)), "wape": float(ae.sum() / y.sum()),
            "r2": float(1 - np.square(error).sum() / var) if var > 0 else None}


def cluster_draws(values: np.ndarray, clusters: np.ndarray, repetitions: int, seed: int) -> np.ndarray:
    """Bootstrap means of `values` resampling whole clusters (vectorised)."""
    table = pd.DataFrame({"c": clusters, "v": values, "n": 1.0}).groupby("c", sort=True).sum()
    sums, counts = table.v.to_numpy(), table.n.to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(repetitions, len(sums)))
    return sums[draws].sum(axis=1) / counts[draws].sum(axis=1)


def cluster_ci(values: np.ndarray, clusters: np.ndarray, repetitions: int, seed: int) -> list[float]:
    return [float(v) for v in np.quantile(cluster_draws(values, clusters, repetitions, seed), [.025, .975])]


def paired_test(a_ae: np.ndarray, b_ae: np.ndarray, clusters: np.ndarray, repetitions: int, seed: int) -> dict:
    """b minus a MAE; cluster-bootstrap CI and two-sided bootstrap p-value."""
    draws = cluster_draws(b_ae - a_ae, clusters, repetitions, seed)
    observed = float(np.mean(b_ae - a_ae))
    p = 2 * min(np.mean(draws >= 0), np.mean(draws <= 0))
    return {"mae_difference_kg": observed, "ci95": [float(v) for v in np.quantile(draws, [.025, .975])],
            "bootstrap_p": float(min(1.0, max(p, 1 / repetitions))),
            "clusters": int(len(np.unique(clusters)))}


def subgroup_table(frame: pd.DataFrame, predictions: dict[str, np.ndarray]) -> dict:
    y = frame.fuel_kg.to_numpy(float)
    positions = frame.adsb_position_count.to_numpy()
    groups = {"duration": duration_groups(frame.duration_min.to_numpy()),
              "aircraft_type": frame.aircraft_type.to_numpy(dtype=str),
              "adsb_availability": np.select([positions == 0, positions <= 50], ["zero", "1-50"], "51+")}
    out = {}
    for label, values in groups.items():
        out[label] = {}
        for name in sorted(set(values)):
            member = values == name
            out[label][name] = {"n": int(member.sum()), "reference_fuel_kg": float(y[member].sum()),
                                "mae_kg": {m: float(np.abs(p[member] - y[member]).mean()) for m, p in predictions.items()}}
    return out


def flight_totals(frame: pd.DataFrame, predictions: dict[str, np.ndarray], repetitions: int, seed: int) -> dict:
    y = frame.fuel_kg.to_numpy(float)
    flights = frame.flight_id.to_numpy()
    truth = pd.Series(y).groupby(flights).sum()
    out = {"flights": int(len(truth)), "labelled_fuel_kg": float(y.sum()), "models": {}}
    rng_seed = seed
    for name, p in predictions.items():
        pred = pd.Series(p).groupby(flights).sum()
        rel = (pred - truth) / truth
        table = np.c_[truth.to_numpy(), pred.to_numpy()]
        rng = np.random.default_rng(rng_seed)
        draws = rng.integers(0, len(table), size=(repetitions, len(table)))
        sampled = table[draws].sum(axis=1)
        bias = (sampled[:, 1] - sampled[:, 0]) / sampled[:, 0]
        out["models"][name] = {
            "flight_total_mape": float(np.abs(rel).mean()), "flight_total_median_ape": float(np.abs(rel).median()),
            "flight_total_p90_ape": float(np.abs(rel).quantile(.9)),
            "inventory_relative_bias": float((p.sum() - y.sum()) / y.sum()),
            "inventory_bias_ci95": [float(v) for v in np.quantile(bias, [.025, .975])]}
    return out


# ----------------------------------------------------------------------------- uncertainty & selection

def duration_bin(frame: pd.DataFrame) -> np.ndarray:
    return duration_groups(frame.duration_min.to_numpy()).astype(str)


def risk_features(train: pd.DataFrame, frame: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
    x, _ = design(train, frame, MODEL_FEATURES["hgb_full"])
    return np.c_[x, np.log1p(np.maximum(prediction, 0)).astype(np.float32)]


def uncertainty_and_selection(train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame,
                              predictions: dict[str, dict[str, np.ndarray]], cfg: dict) -> dict:
    alpha, reps, seed = cfg["alpha"], cfg["bootstrap_repetitions"], cfg["seed"]
    half_a = stable_tie(cal, "journal-half") < 0.5
    # Keep whole flights together in either half.
    flight_half = pd.Series(half_a).groupby(cal.flight_id.to_numpy()).first()
    half_a = cal.flight_id.map(flight_half).to_numpy(bool)
    half_b = ~half_a
    y_cal, y_test = cal.fuel_kg.to_numpy(float), test.fuel_kg.to_numpy(float)
    cells_test = test.aircraft_type.astype(str).to_numpy() + "/" + duration_bin(test)
    cell_names, cell_counts = np.unique(cells_test, return_counts=True)
    eligible_cells = set(cell_names[cell_counts >= cfg["min_cell_n"]])
    out = {"calibration_split": {"half_a_intervals": int(half_a.sum()), "half_b_intervals": int(half_b.sum()),
                                 "half_a_flights": int(cal.flight_id[half_a].nunique()),
                                 "half_b_flights": int(cal.flight_id[half_b].nunique())},
           "eligible_test_cells": len(eligible_cells), "models": {}}
    for model in (cfg["reference_model"], "hgb_full", "hgb_residual_physics"):
        p_cal, p_test = predictions[model]["september"], predictions[model]["october"]
        ae_cal = np.abs(y_cal - p_cal)
        risk_model = HistGradientBoostingRegressor(**cfg["risk_hist_gradient_boosting"]).fit(
            risk_features(train, cal[half_a].reset_index(drop=True), p_cal[half_a]), ae_cal[half_a])
        sigma_cal = np.maximum(risk_model.predict(risk_features(train, cal, p_cal)), 1.0)
        sigma_test = np.maximum(risk_model.predict(risk_features(train, test, p_test)), 1.0)
        b = cal[half_b].reset_index(drop=True)
        scores_b = ae_cal[half_b] / sigma_cal[half_b]
        methods = {}
        q_const = finite_quantile(ae_cal[half_b], alpha)
        methods["split_constant"] = np.full(len(test), q_const)
        q_scaled = finite_quantile(scores_b, alpha)
        methods["split_scaled"] = q_scaled * sigma_test
        mondrian = HierarchicalMondrian(alpha=alpha, min_group_n=cfg["min_type_n"]).fit(
            scores_b, b.aircraft_type.to_numpy(str), duration_bin(b))
        mult, levels = mondrian.multipliers(test.aircraft_type.to_numpy(str), duration_bin(test))
        methods["mondrian_type_duration"] = mult * sigma_test
        intervals = {}
        for name, half in methods.items():
            lo, hi = np.maximum(0.0, p_test - half), p_test + half
            covered = (y_test >= lo) & (y_test <= hi)
            cells = {}
            for cell in sorted(eligible_cells):
                member = cells_test == cell
                cells[cell] = {"n": int(member.sum()), "coverage": float(covered[member].mean()),
                               "coverage_ci95": cluster_ci(covered[member].astype(float), test.flight_id.to_numpy()[member], reps, seed),
                               "mean_width_kg": float((hi - lo)[member].mean())}
            worst = min(cells.values(), key=lambda c: c["coverage"])
            intervals[name] = {**interval_metrics(y_test, lo, hi, alpha),
                               "coverage_ci95": cluster_ci(covered.astype(float), test.flight_id.to_numpy(), reps, seed),
                               "worst_cell_coverage": worst["coverage"],
                               "cells_below_085": int(sum(c["coverage"] < .85 for c in cells.values())),
                               "cells": cells}
            if name == "mondrian_type_duration":
                unique, counts = np.unique(levels, return_counts=True)
                intervals[name]["fallback_levels"] = dict(zip(unique.tolist(), counts.astype(int).tolist()))
        # Risk ranking quality on test: Spearman between sigma and realised error, decile MAE.
        ae_test = np.abs(y_test - p_test)
        order = np.argsort(sigma_test, kind="stable")
        deciles = [{"mean_sigma_kg": float(sigma_test[i].mean()), "observed_mae_kg": float(ae_test[i].mean()), "n": len(i)}
                   for i in np.array_split(order, 10)]
        spearman = float(pd.Series(sigma_test).corr(pd.Series(ae_test), method="spearman"))
        # Representation-aware selection on the October batch.
        ties = stable_tie(test, "journal-select")
        types, durations = test.aircraft_type.to_numpy(str), test.duration_min.to_numpy()
        frontier = []
        for target in cfg["frontier_targets"]:
            row = {"target": target}
            for policy, floor in (("risk_only", 0.0), ("representation_floor", cfg["selection_floor"])):
                mask, details = select(sigma_test, types, durations, ties, target, floor, cfg["min_type_n"])
                dur = duration_groups(durations)
                dur_cov = [mask[dur == g].mean() for g in DURATION_LABELS if (dur == g).any()]
                prot = details["protected_aircraft_types"]
                type_cov = [mask[types == t].mean() for t in prot]
                row[policy] = {"accepted": int(mask.sum()), "coverage": float(mask.mean()),
                               "accepted_mae_kg": float(ae_test[mask].mean()),
                               "rejected_mae_kg": float(ae_test[~mask].mean()) if (~mask).any() else None,
                               "fuel_share": float(y_test[mask].sum() / y_test.sum()),
                               "duration_gap": float(max(dur_cov) - min(dur_cov)),
                               "aircraft_gap": float(max(type_cov) - min(type_cov)) if type_cov else None,
                               "min_duration_coverage": float(min(dur_cov)),
                               "b744_coverage": float(mask[types == "B744"].mean()) if (types == "B744").any() else None,
                               "long_interval_coverage": float(mask[dur == "15–60 min"].mean()) if (dur == "15–60 min").any() else None}
            frontier.append(row)
        primary = next(r for r in frontier if r["target"] == cfg["selection_target"])
        out["models"][model] = {
            "sigma_spearman_with_error": spearman, "sigma_deciles": deciles,
            "intervals": intervals, "selection_frontier": frontier,
            "selection_primary": {**primary, "floor_cost_kg": primary["representation_floor"]["accepted_mae_kg"]
                                  - primary["risk_only"]["accepted_mae_kg"]}}
    return out


# ----------------------------------------------------------------------------- experiments

def leave_one_type_out(train: pd.DataFrame, frames: dict[str, pd.DataFrame], cfg: dict) -> dict:
    combined = pd.concat(frames.values(), ignore_index=True)
    counts = combined.aircraft_type.value_counts()
    eligible = sorted(t for t, n in counts.items() if n >= cfg["loto_min_intervals"] and t in set(train.aircraft_type))
    results = {"eligible_types": eligible, "min_intervals": cfg["loto_min_intervals"], "types": {}}
    for aircraft_type in eligible:
        held = {p: f[f.aircraft_type == aircraft_type].reset_index(drop=True) for p, f in frames.items()}
        preds, audit = fit_all_models(train, held, cfg, exclude_type=aircraft_type)
        y = np.concatenate([held[p].fuel_kg.to_numpy(float) for p in frames])
        flights = np.concatenate([held[p].flight_id.to_numpy() for p in frames])
        row = {"n": int(len(y)), "flights": int(len(np.unique(flights))),
               "openap_scale_fallback": audit["physics_global_scale"], "mae_kg": {}}
        ref = np.abs(np.concatenate([preds[cfg["reference_model"]][p] for p in frames]) - y)
        for model in preds:
            p = np.concatenate([preds[model][q] for q in frames])
            row["mae_kg"][model] = float(np.abs(p - y).mean())
            if model != cfg["reference_model"]:
                row.setdefault("vs_reference", {})[model] = paired_test(
                    ref, np.abs(p - y), flights, cfg["bootstrap_repetitions"], cfg["seed"])
        results["types"][aircraft_type] = row
        print(f"LOTO {aircraft_type}: " + ", ".join(f"{m}={v:.1f}" for m, v in row["mae_kg"].items()), flush=True)
    wins = {m: int(sum(r["mae_kg"][m] < r["mae_kg"][cfg["reference_model"]] for r in results["types"].values()))
            for m in next(iter(results["types"].values()))["mae_kg"]} if results["types"] else {}
    results["types_where_model_beats_reference"] = wins
    results["pooled_mae_kg"] = {m: float(np.average([r["mae_kg"][m] for r in results["types"].values()],
                                                    weights=[r["n"] for r in results["types"].values()]))
                                for m in wins}
    return results


def ablation(train: pd.DataFrame, test: pd.DataFrame, cfg: dict, full_pred: np.ndarray) -> dict:
    y = test.fuel_kg.to_numpy(float)
    out = {"full_mae_kg": float(np.abs(full_pred - y).mean()), "drop": {}}
    for dropped in ("adsb", "kin", "phys", "meta"):
        groups = [g for g in MODEL_FEATURES["hgb_full"] if g != dropped]
        x, _ = design(train, train, groups)
        model = fit_hgb(x, train.fuel_kg.to_numpy(float), cfg["hist_gradient_boosting"])
        p = np.maximum(0.0, model.predict(design(train, test, groups)[0]))
        out["drop"][dropped] = {"mae_kg": float(np.abs(p - y).mean()),
                                "delta_vs_full_kg": float(np.abs(p - y).mean() - out["full_mae_kg"])}
    return out


def route_transfer(train: pd.DataFrame, test: pd.DataFrame, predictions: dict[str, np.ndarray]) -> dict:
    seen = test.route_id.isin(set(train.route_id)).to_numpy()
    y = test.fuel_kg.to_numpy(float)
    return {"seen_route_intervals": int(seen.sum()), "unseen_route_intervals": int((~seen).sum()),
            "unseen_routes": int(test.route_id[~seen].nunique()),
            "mae_kg": {m: {"seen": float(np.abs(p[seen] - y[seen]).mean()),
                           "unseen": float(np.abs(p[~seen] - y[~seen]).mean())} for m, p in predictions.items()}}


def run(raw_dir: Path, feature_dir: Path, cfg_path: Path, *, skip_loto: bool = False) -> dict:
    started = time.perf_counter()
    cfg = json.loads(cfg_path.read_text())
    manifest = json.loads((feature_dir / "feature_manifest.json").read_text())
    if any(s not in manifest or manifest[s]["subset"] for s in ("train", "rank", "final")):
        raise ValueError("Complete verified feature manifests required for train, rank and final")
    train = load_period(raw_dir, feature_dir, "train")
    cal = load_period(raw_dir, feature_dir, "rank")
    test = load_period(raw_dir, feature_dir, "final")
    for s, f in (("train", train), ("rank", cal), ("final", test)):
        if len(f) != manifest[s]["intervals"]:
            raise ValueError(f"Manifest count mismatch for {s}")
    if set(train.flight_id) & set(cal.flight_id) or set(train.flight_id) & set(test.flight_id) or set(cal.flight_id) & set(test.flight_id):
        raise ValueError("Flight identifiers overlap across periods")
    frames = {"september": cal, "october": test}
    predictions, audit = fit_all_models(train, frames, cfg)
    reps, seed, ref = cfg["bootstrap_repetitions"], cfg["seed"], cfg["reference_model"]
    result = {
        "name": cfg["name"], "status": cfg["status"], "evidence_boundary": cfg["evidence_boundary"],
        "config": cfg, "config_sha256": sha256(cfg_path), "code_sha256": sha256(Path(__file__)),
        "extract_sha256": sha256(Path(__file__).with_name("extract.py")),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "platform": platform.platform(), "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "feature_manifest": manifest, "source_sha256": {p.name: sha256(p) for p in sorted(raw_dir.glob("*.parquet"))},
        "counts": {name: {"intervals": len(f), "flights": int(f.flight_id.nunique()),
                          "aircraft_types": int(f.aircraft_type.nunique()),
                          "zero_adsb_intervals": int(f.adsb_position_count.eq(0).sum()),
                          "physics_available": int(f.phys_openap_kg.notna().sum())}
                   for name, f in (("train", train), ("september", cal), ("october", test))},
        "fit_audit": audit, "periods": {},
    }
    for period, frame in frames.items():
        y = frame.fuel_kg.to_numpy(float)
        flights = frame.flight_id.to_numpy()
        ref_ae = np.abs(predictions[ref][period] - y)
        models = {}
        paired = {}
        for name in predictions:
            p = predictions[name][period]
            models[name] = {**metrics(y, p), "mae_flight_ci95": cluster_ci(np.abs(p - y), flights, reps, seed)}
            if name != ref:
                paired[name] = paired_test(ref_ae, np.abs(p - y), flights, reps, seed)
        names = list(paired)
        adjusted = adjust_pvalues(np.array([paired[n]["bootstrap_p"] for n in names]), "holm")
        for n, p_adj in zip(names, adjusted):
            paired[n]["holm_p"] = float(p_adj)
        result["periods"][period] = {
            "models": models, "paired_vs_reference": paired,
            "subgroups": subgroup_table(frame, {m: predictions[m][period] for m in predictions}),
            "flight_totals": flight_totals(frame, {m: predictions[m][period] for m in predictions}, reps, seed),
            "route_transfer": route_transfer(train, frame, {m: predictions[m][period] for m in predictions}),
        }
        print(period, {m: round(v["mae_kg"], 2) for m, v in models.items()}, flush=True)
    result["uncertainty_and_selection"] = uncertainty_and_selection(train, cal, test, predictions, cfg)
    result["ablation_october"] = ablation(train, test, cfg, predictions["hgb_full"]["october"])
    if not skip_loto:
        result["leave_one_type_out"] = leave_one_type_out(train, frames, cfg)
    result["runtime_seconds"] = round(time.perf_counter() - started, 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=ROOT / "research_q1/configs/journal_study.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research_q1/outputs/journal-study.json")
    parser.add_argument("--skip-loto", action="store_true")
    args = parser.parse_args()
    result = run(args.raw_dir, args.feature_dir, args.config, skip_loto=args.skip_loto)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} in {result['runtime_seconds']} s")


if __name__ == "__main__":
    main()
