"""Training-fold-only aircraft physics prior in documented SI units.

The model is deliberately auditable. It is not BADA and is not a certified
performance model. Unknown type-level parameters are fit by bounded least
squares on the training fold only; the fitted values and bounds are returned so
that the approximation cannot be mistaken for measured mass or engine data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

G = 9.80665
R_AIR = 287.05287
GAMMA = 1.4
T0 = 288.15
P0 = 101325.0
LAPSE = 0.0065


@dataclass(frozen=True)
class TypeParameters:
    mass_kg: float
    tsfc_kg_per_newton_s: float
    cd0: float
    oswald_efficiency: float
    wave_drag_coefficient: float


DEFAULT_BOUNDS = {
    "mass_fraction_mtow": (0.45, 0.98),
    "tsfc_kg_per_newton_s": (0.8e-5, 3.5e-5),
    "cd0": (0.012, 0.055),
    "oswald_efficiency": (0.55, 0.92),
    "wave_drag_coefficient": (0.0, 0.25),
}


def isa_atmosphere(altitude_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tropospheric ISA temperature, density and speed of sound."""
    h = np.clip(np.asarray(altitude_m, float), 0.0, 11000.0)
    temperature = T0 - LAPSE * h
    pressure = P0 * (temperature / T0) ** (G / (R_AIR * LAPSE))
    density = pressure / (R_AIR * temperature)
    speed_of_sound = np.sqrt(GAMMA * R_AIR * temperature)
    return temperature, density, speed_of_sound


def true_airspeed(ground_east_mps: np.ndarray, ground_north_mps: np.ndarray,
                  wind_east_mps: np.ndarray, wind_north_mps: np.ndarray) -> np.ndarray:
    east = np.asarray(ground_east_mps, float) - np.asarray(wind_east_mps, float)
    north = np.asarray(ground_north_mps, float) - np.asarray(wind_north_mps, float)
    return np.sqrt(east ** 2 + north ** 2)


def aerodynamic_state(altitude_m: np.ndarray, tas_mps: np.ndarray, mass_kg: float,
                      wing_area_m2: np.ndarray, aspect_ratio: np.ndarray,
                      cd0: float, oswald_efficiency: float,
                      wave_drag_coefficient: float = 0.0,
                      critical_mach: float = 0.78) -> dict[str, np.ndarray]:
    _, density, sound = isa_atmosphere(altitude_m)
    speed = np.maximum(np.asarray(tas_mps, float), 30.0)
    area = np.asarray(wing_area_m2, float)
    ar = np.asarray(aspect_ratio, float)
    mach = speed / sound
    cl = 2 * mass_kg * G / (density * speed ** 2 * area)
    wave = wave_drag_coefficient * np.maximum(0.0, mach - critical_mach) ** 2
    cd = cd0 + cl ** 2 / (np.pi * oswald_efficiency * ar) + wave
    lift_to_drag = cl / np.maximum(cd, 1e-8)
    drag_newton = mass_kg * G / np.maximum(lift_to_drag, 1e-8)
    return {"density_kg_m3": density, "mach": mach, "cl": cl, "cd": cd,
            "lift_to_drag": lift_to_drag, "drag_newton": drag_newton}


def estimate_segment_fuel(frame: pd.DataFrame, parameters: TypeParameters) -> np.ndarray:
    """Estimate segment fuel for taxi, cruise, climb and descent rows."""
    required = {
        "flight_phase", "altitude_m", "ground_velocity_east_mps",
        "ground_velocity_north_mps", "wind_east_mps", "wind_north_mps",
        "wing_area_m2", "aspect_ratio", "segment_duration_s", "range_m",
        "vertical_rate_mps", "tas_acceleration_mps2", "tank_capacity_kg",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Physics table is missing columns: {sorted(missing)}")
    tas = true_airspeed(frame.ground_velocity_east_mps, frame.ground_velocity_north_mps,
                        frame.wind_east_mps, frame.wind_north_mps)
    state = aerodynamic_state(frame.altitude_m, tas, parameters.mass_kg,
                              frame.wing_area_m2, frame.aspect_ratio, parameters.cd0,
                              parameters.oswald_efficiency,
                              parameters.wave_drag_coefficient)
    duration = np.maximum(frame.segment_duration_s.to_numpy(float), 0.0)
    phase = frame.flight_phase.astype(str).str.lower().to_numpy()
    thrust = state["drag_newton"].copy()
    energy_thrust = state["drag_newton"] + (
        parameters.mass_kg * G * frame.vertical_rate_mps.to_numpy(float)
        + parameters.mass_kg * tas * frame.tas_acceleration_mps2.to_numpy(float)
    ) / np.maximum(tas, 30.0)
    dynamic = np.isin(phase, ["climb", "descent"])
    thrust[dynamic] = np.maximum(0.08 * state["drag_newton"][dynamic], energy_thrust[dynamic])
    fuel = parameters.tsfc_kg_per_newton_s * np.maximum(thrust, 0.0) * duration
    cruise = phase == "cruise"
    exponent = (frame.range_m.to_numpy(float) * parameters.tsfc_kg_per_newton_s * G
                / (np.maximum(tas, 30.0) * np.maximum(state["lift_to_drag"], 1.0)))
    fuel[cruise] = parameters.mass_kg * (1 - np.exp(-np.maximum(exponent[cruise], 0.0)))
    taxi = phase == "taxi"
    fuel[taxi] = parameters.tsfc_kg_per_newton_s * (0.07 * parameters.mass_kg * G) * duration[taxi]
    return np.clip(fuel, 0.0, frame.tank_capacity_kg.to_numpy(float))


class PhysicsPrior:
    """Fit bounded aircraft-type parameters using training labels only."""

    def __init__(self, bounds: dict | None = None):
        self.bounds = DEFAULT_BOUNDS if bounds is None else bounds
        self.parameters_: dict[str, TypeParameters] = {}
        self.fit_audit_: dict[str, dict] = {}

    def fit(self, frame: pd.DataFrame) -> "PhysicsPrior":
        needed = {"aircraft_type", "reference_fuel_kg", "max_takeoff_mass_kg"}
        if needed - set(frame):
            raise ValueError(f"Physics fit is missing columns: {sorted(needed - set(frame))}")
        for aircraft_type, group in frame.groupby("aircraft_type", sort=True):
            mtow = float(np.median(group.max_takeoff_mass_kg))
            lo = np.array([self.bounds["mass_fraction_mtow"][0] * mtow,
                           self.bounds["tsfc_kg_per_newton_s"][0], self.bounds["cd0"][0],
                           self.bounds["oswald_efficiency"][0], self.bounds["wave_drag_coefficient"][0]])
            hi = np.array([self.bounds["mass_fraction_mtow"][1] * mtow,
                           self.bounds["tsfc_kg_per_newton_s"][1], self.bounds["cd0"][1],
                           self.bounds["oswald_efficiency"][1], self.bounds["wave_drag_coefficient"][1]])
            initial = (lo + hi) / 2
            truth = group.reference_fuel_kg.to_numpy(float)

            def residual(values: np.ndarray) -> np.ndarray:
                params = TypeParameters(*map(float, values))
                scale = np.maximum(np.sqrt(np.maximum(truth, 1.0)), 1.0)
                return (estimate_segment_fuel(group, params) - truth) / scale

            fitted = least_squares(residual, initial, bounds=(lo, hi), method="trf")
            params = TypeParameters(*map(float, fitted.x))
            self.parameters_[str(aircraft_type)] = params
            self.fit_audit_[str(aircraft_type)] = {
                "n_training_rows": int(len(group)), "success": bool(fitted.success),
                "cost": float(fitted.cost), "parameters": asdict(params),
                "lower_bounds": lo.tolist(), "upper_bounds": hi.tolist(),
            }
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.parameters_:
            raise RuntimeError("PhysicsPrior must be fit before prediction")
        output = np.empty(len(frame), dtype=float)
        global_params = TypeParameters(*np.median(np.array([
            list(asdict(value).values()) for value in self.parameters_.values()
        ]), axis=0))
        for name, positions in frame.groupby("aircraft_type", sort=False).indices.items():
            params = self.parameters_.get(str(name), global_params)
            output[np.asarray(positions)] = estimate_segment_fuel(frame.iloc[positions], params)
        return output

