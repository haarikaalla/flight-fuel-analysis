"""Single-pass interval feature extraction for the journal study.

For every published split this writes

* ``features_{split}.parquet`` - the original 14 ADS-B summaries produced by
  :func:`src.flight_fuel.features.features_for_flight` (same function, same
  manifest format), so the inherited eligibility pipeline remains reproducible;
* ``kinematics_{split}.parquet`` - SI kinematic, energy and OpenAP physics
  features defined below.

Only ``source == "adsb"`` rows enter any feature. ACARS rows, which carry the
fuel-on-board reports, are removed before any statistic is computed.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.flight_fuel.data_audit import checksum
from src.flight_fuel.download_archives import ARCHIVES
from src.flight_fuel.features import FEATURE_COLS, TRAJECTORY_COLS, features_for_flight

G = 9.80665
EARTH_RADIUS_KM = 6371.0088
# The archive README states metres and m/s, but the released values are
# barometric feet, knots and ft/min (cruise ~35,000 and ~450); see unit_check.
FT_TO_M = 0.3048
KT_TO_MPS = 0.514444
FPM_TO_MPS = 0.00508
MPS_TO_KT = 1 / KT_TO_MPS
M_TO_FT = 1 / FT_TO_M
MPS_TO_FPM = 1 / FPM_TO_MPS
GROUND_SPEED_MPS = 40.0       # no jet transport is airborne below ~78 kt
CLIMB_MPS = 2.5               # ~500 ft/min phase threshold
PHYSICS_STEP_S = 30.0
PHYSICS_MAX_SAMPLES = 240
# Plausible ranges in the released raw units; values outside become NaN.
VALID_RANGES = {"altitude": (-2000.0, 50000.0), "groundspeed": (0.0, 700.0),
                "vertical_rate": (-12000.0, 12000.0)}
TO_SI = {"altitude": FT_TO_M, "groundspeed": KT_TO_MPS, "vertical_rate": FPM_TO_MPS}

KINEMATIC_COLS = [
    "kin_alt_mean_m", "kin_alt_delta_m", "kin_gs_mean_mps", "kin_gs_delta_mps",
    "kin_vr_mean_mps", "kin_frac_climb", "kin_frac_descent", "kin_frac_ground",
    "kin_track_km", "kin_obs_span_fraction", "kin_specific_energy_rate_w_per_kg",
    "ctx_alt_start_m", "ctx_alt_end_m", "ctx_gs_mid_mps",
    "phys_openap_kg", "phys_max_interp_gap_s", "phys_frac_ground",
]

_PHYSICS_CACHE: dict[str, tuple] = {}


def physics_model(aircraft_type: str) -> tuple:
    """OpenAP fuel-flow model, nominal mass and ground fuel flow (kg/s) for one type.

    Nominal mass is OEW + 0.5 (MTOW - OEW); it is a documented assumption, not
    a measured mass. Ground fuel flow uses 7% of total static take-off thrust.
    """
    if aircraft_type not in _PHYSICS_CACHE:
        from openap import FuelFlow, Thrust, prop
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            props = prop.aircraft(aircraft_type, use_synonym=True)
            fuel = FuelFlow(aircraft_type, use_synonym=True)
            static = float(np.ravel(Thrust(aircraft_type, use_synonym=True).takeoff(tas=0, alt=0))[0])
            ground = float(np.ravel(fuel.at_thrust(0.07 * static))[0])
        mass = props["oew"] + 0.5 * (props["mtow"] - props["oew"])
        _PHYSICS_CACHE[aircraft_type] = (fuel, float(mass), ground, str(props.get("aircraft", "")))
    return _PHYSICS_CACHE[aircraft_type]


def openap_fuel_flow(aircraft_type: str, gs_mps: np.ndarray, alt_m: np.ndarray,
                     vr_mps: np.ndarray) -> np.ndarray:
    """Fuel flow (kg/s); groundspeed stands in for TAS because winds are absent."""
    fuel, mass, ground, _ = physics_model(aircraft_type)
    gs, alt, vr = (np.asarray(v, float) for v in (gs_mps, alt_m, vr_mps))
    flow = np.full(len(gs), ground)
    airborne = gs >= GROUND_SPEED_MPS
    if airborne.any():
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            value = fuel.enroute(mass=np.full(airborne.sum(), mass),
                                 tas=np.clip(gs[airborne] * MPS_TO_KT, 100, 650),
                                 alt=np.clip(alt[airborne] * M_TO_FT, 0, 45000),
                                 vs=np.clip(vr[airborne] * MPS_TO_FPM, -6000, 6000))
        flow[airborne] = np.nan_to_num(np.ravel(value), nan=ground, posinf=ground)
    return np.maximum(flow, 0.0)


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if len(lat) < 2:
        return np.zeros(0)
    p1, p2 = np.radians(lat[:-1]), np.radians(lat[1:])
    dl = np.radians(lon[1:] - lon[:-1])
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def track_km(t: np.ndarray, lat: np.ndarray, lon: np.ndarray, max_kmps: float = 0.35) -> float:
    """Along-track distance after removing position spikes implying > ~680 kt."""
    keep = np.ones(len(t), dtype=bool)
    for _ in range(5):
        tt, la, lo = t[keep], lat[keep], lon[keep]
        if len(tt) < 2:
            break
        speed = haversine_km(la, lo) / np.maximum(np.diff(tt), 1e-3)
        fast = speed > max_kmps
        if not fast.any():
            break
        # A spike is a point whose incoming and outgoing legs are both impossible.
        spike = np.r_[False, fast[:-1] & fast[1:], fast[-1]]
        if not spike.any():
            spike = np.r_[False, fast]
        keep[np.flatnonzero(keep)[spike]] = False
    tt, la, lo = t[keep], lat[keep], lon[keep]
    if len(tt) < 2:
        return 0.0
    step = haversine_km(la, lo)
    return float(step[step / np.maximum(np.diff(tt), 1e-3) <= max_kmps].sum())


def clean_adsb(trajectory: pd.DataFrame) -> pd.DataFrame:
    adsb = trajectory.loc[trajectory.source.eq("adsb"), TRAJECTORY_COLS].dropna(
        subset=["timestamp", "latitude", "longitude"]).sort_values("timestamp", kind="stable")
    adsb = adsb.drop_duplicates("timestamp", keep="first").copy()
    for column, (lo, hi) in VALID_RANGES.items():
        values = adsb[column].astype(float)
        adsb[column] = values.where(values.between(lo, hi)) * TO_SI[column]
    return adsb


def _interp(times: np.ndarray, obs_t: np.ndarray, obs_v: np.ndarray) -> np.ndarray:
    keep = np.isfinite(obs_v)
    if keep.sum() == 0:
        return np.full(len(times), np.nan)
    return np.interp(times, obs_t[keep], obs_v[keep])


def kinematics_for_flight(trajectory: pd.DataFrame, intervals: pd.DataFrame, flight_id: str,
                          aircraft_type: str) -> pd.DataFrame:
    adsb = clean_adsb(trajectory)
    origin = (adsb.timestamp.iloc[0] if len(adsb) else pd.Timestamp(intervals.start.min())).to_datetime64()
    to_s = lambda values: (pd.to_datetime(values).to_numpy(dtype="datetime64[ns]")
                           - origin.astype("datetime64[ns]")).astype(np.int64) / 1e9
    t = to_s(adsb.timestamp) if len(adsb) else np.zeros(0)
    alt, gs, vr = (adsb[c].to_numpy(float) for c in ("altitude", "groundspeed", "vertical_rate"))
    lat, lon = adsb.latitude.to_numpy(float), adsb.longitude.to_numpy(float)
    starts, ends = to_s(intervals.start), to_s(intervals.end)
    left = np.searchsorted(t, starts, side="left")
    right = np.searchsorted(t, ends, side="left")
    rows = []
    for idx, s, e, lo, hi in zip(intervals.idx.to_numpy(), starts, ends, left, right):
        duration = e - s
        if duration <= 0:
            raise ValueError(f"Invalid duration for interval {idx}")
        row = {"idx": int(idx), "flight_id": flight_id, **{c: np.nan for c in KINEMATIC_COLS}}
        if hi > lo:
            a, g, v = alt[lo:hi], gs[lo:hi], vr[lo:hi]
            span = t[hi - 1] - t[lo]
            fa, fg = a[np.isfinite(a)], g[np.isfinite(g)]
            fv = v[np.isfinite(v)]
            row.update({
                "kin_alt_mean_m": fa.mean() if len(fa) else np.nan,
                "kin_alt_delta_m": fa[-1] - fa[0] if len(fa) else np.nan,
                "kin_gs_mean_mps": fg.mean() if len(fg) else np.nan,
                "kin_gs_delta_mps": fg[-1] - fg[0] if len(fg) else np.nan,
                "kin_vr_mean_mps": fv.mean() if len(fv) else np.nan,
                "kin_frac_climb": float((fv > CLIMB_MPS).mean()) if len(fv) else np.nan,
                "kin_frac_descent": float((fv < -CLIMB_MPS).mean()) if len(fv) else np.nan,
                "kin_frac_ground": float((fg < GROUND_SPEED_MPS).mean()) if len(fg) else np.nan,
                "kin_track_km": track_km(t[lo:hi], lat[lo:hi], lon[lo:hi]),
                "kin_obs_span_fraction": span / duration,
            })
            if span > 0 and len(fa) and len(fg):
                row["kin_specific_energy_rate_w_per_kg"] = (
                    G * (fa[-1] - fa[0]) + 0.5 * (fg[-1] ** 2 - fg[0] ** 2)) / span
        if len(t) >= 2:
            k = int(np.clip(np.ceil(duration / PHYSICS_STEP_S), 2, PHYSICS_MAX_SAMPLES))
            samples = s + (np.arange(k) + 0.5) * duration / k
            sa, sg, sv = _interp(samples, t, alt), _interp(samples, t, gs), _interp(samples, t, vr)
            nearest = np.searchsorted(t, samples).clip(1, len(t) - 1)
            gap = np.minimum(np.abs(samples - t[nearest - 1]), np.abs(t[nearest] - samples))
            row.update({"ctx_alt_start_m": _interp(np.array([s]), t, alt)[0],
                        "ctx_alt_end_m": _interp(np.array([e]), t, alt)[0],
                        "ctx_gs_mid_mps": _interp(np.array([(s + e) / 2]), t, gs)[0],
                        "phys_max_interp_gap_s": float(gap.max())})
            if np.isfinite(sa).all() and np.isfinite(sg).all():
                sv = np.nan_to_num(sv, nan=0.0)
                flow = openap_fuel_flow(aircraft_type, sg, sa, sv)
                row["phys_openap_kg"] = float(flow.mean() * duration)
                row["phys_frac_ground"] = float((sg < GROUND_SPEED_MPS).mean())
        rows.append(row)
    return pd.DataFrame(rows, columns=["idx", "flight_id", *KINEMATIC_COLS])


def _process_chunk(args: tuple) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict]:
    archive_path, split, chunk = args
    original, kinematic = [], []
    unit = {"cruise_alt_raw": [], "cruise_gs_raw": []}
    with zipfile.ZipFile(archive_path) as archive:
        for flight_id, aircraft_type, intervals in chunk:
            with archive.open(f"flights_{split}/{flight_id}.parquet") as member:
                trajectory = pd.read_parquet(io.BytesIO(member.read()), columns=TRAJECTORY_COLS)
            original.append(features_for_flight(trajectory, intervals, flight_id))
            kinematic.append(kinematics_for_flight(trajectory, intervals, flight_id, aircraft_type))
            adsb = trajectory[trajectory.source.eq("adsb")]
            fast = adsb[adsb.groundspeed > 300]
            if len(fast):
                unit["cruise_alt_raw"].append(float(fast.altitude.quantile(.9)))
                unit["cruise_gs_raw"].append(float(fast.groundspeed.median()))
    return original, kinematic, unit


def build(raw_dir: Path, output_dir: Path, split: str, workers: int, max_flights: int | None) -> dict:
    filename = f"flights_{split}.zip"
    size, md5 = ARCHIVES[filename]
    archive_path = raw_dir / filename
    if not archive_path.exists() or archive_path.stat().st_size != size or checksum(archive_path) != md5:
        raise ValueError(f"Missing or MD5-invalid archive: {archive_path}")
    flights = pd.read_parquet(raw_dir / f"flightlist_{split}.parquet", columns=["flight_id", "aircraft_type"])
    intervals = pd.read_parquet(raw_dir / f"fuel_{split}.parquet", columns=["idx", "flight_id", "start", "end"])
    by_flight = dict(tuple(intervals.groupby("flight_id", sort=False)))
    types = flights.set_index("flight_id").aircraft_type.to_dict()
    selected = sorted(flights.flight_id)[:max_flights] if max_flights else sorted(flights.flight_id)
    if set(selected) - set(by_flight):
        raise ValueError("Selected flight has no fuel intervals")
    with zipfile.ZipFile(archive_path) as archive:
        members = {n.rsplit("/", 1)[-1].removesuffix(".parquet") for n in archive.namelist()
                   if n.startswith(f"flights_{split}/") and n.endswith(".parquet")}
    if max_flights is None and members != set(flights.flight_id):
        raise ValueError("Archive members do not match flight list")
    jobs = [(flight_id, types[flight_id], by_flight[flight_id]) for flight_id in selected]
    chunks = [(archive_path, split, jobs[i:i + 100]) for i in range(0, len(jobs), 100)]
    original, kinematic, cruise_alt, cruise_gs = [], [], [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, (a, b, unit) in enumerate(pool.map(_process_chunk, chunks), 1):
            original.extend(a)
            kinematic.extend(b)
            cruise_alt.extend(unit["cruise_alt_raw"])
            cruise_gs.extend(unit["cruise_gs_raw"])
            if n % 10 == 0 or n == len(chunks):
                print(f"{split}: {min(n * 100, len(jobs))}/{len(jobs)} flights "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
    base = pd.concat(original, ignore_index=True)
    kin = pd.concat(kinematic, ignore_index=True)
    if base.idx.duplicated().any() or kin.idx.duplicated().any() or len(base) != len(kin):
        raise ValueError("Duplicate or misaligned interval features")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if max_flights is None else f"_first_{max_flights}"
    base[["idx", "flight_id", *FEATURE_COLS]].to_parquet(output_dir / f"features_{split}{suffix}.parquet", index=False)
    kin.to_parquet(output_dir / f"kinematics_{split}{suffix}.parquet", index=False)
    return {
        "split": split, "archive_md5": md5, "flights": len(selected), "intervals": len(base),
        "zero_adsb_intervals": int(base.adsb_position_count.eq(0).sum()),
        "feature_file": f"features_{split}{suffix}.parquet",
        "kinematics_file": f"kinematics_{split}{suffix}.parquet",
        "physics_missing_intervals": int(kin.phys_openap_kg.isna().sum()),
        "unit_check": {"flights": len(cruise_alt),
                       "median_flight_p90_raw_altitude_when_groundspeed_gt_300": float(np.median(cruise_alt)) if cruise_alt else None,
                       "median_raw_groundspeed_when_gt_300": float(np.median(cruise_gs)) if cruise_gs else None,
                       "interpretation": "feet and knots (README states metres and m/s)"},
        "subset": max_flights is not None,
        "seconds": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits", nargs="+", choices=("train", "rank", "final"), default=["train", "rank", "final"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-flights", type=int, default=None, help="Smoke test only")
    args = parser.parse_args()
    path = args.output_dir / "feature_manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    for split in args.splits:
        manifest[split] = build(args.raw_dir, args.output_dir, split, args.workers, args.max_flights)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest[split], indent=2), flush=True)


if __name__ == "__main__":
    main()
