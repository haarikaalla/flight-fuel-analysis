import unittest

import numpy as np
import pandas as pd

from research_q1.novel.physics import (PhysicsPrior, TypeParameters,
                                       estimate_segment_fuel, isa_atmosphere,
                                       true_airspeed)


class PhysicsTests(unittest.TestCase):
    def test_isa_density_decreases_with_altitude(self):
        _, density, sound = isa_atmosphere(np.array([0.0, 5000.0, 10000.0]))
        self.assertTrue(np.all(np.diff(density) < 0))
        self.assertTrue(np.all(sound > 0))

    def test_wind_corrected_airspeed(self):
        value = true_airspeed(np.array([100.0]), np.array([0.0]),
                              np.array([20.0]), np.array([0.0]))
        self.assertAlmostEqual(value[0], 80.0)

    def test_physics_fuel_is_finite_and_bounded(self):
        frame = pd.DataFrame({
            "flight_phase": ["cruise", "climb"], "altitude_m": [10000.0, 3000.0],
            "ground_velocity_east_mps": [230.0, 160.0], "ground_velocity_north_mps": [0.0, 0.0],
            "wind_east_mps": [10.0, 5.0], "wind_north_mps": [0.0, 0.0],
            "wing_area_m2": [122.6, 122.6], "aspect_ratio": [9.4, 9.4],
            "segment_duration_s": [1800.0, 600.0], "range_m": [400000.0, 90000.0],
            "vertical_rate_mps": [0.0, 8.0], "tas_acceleration_mps2": [0.0, 0.02],
            "tank_capacity_kg": [24000.0, 24000.0],
        })
        params = TypeParameters(65000.0, 1.6e-5, 0.025, 0.8, 0.02)
        fuel = estimate_segment_fuel(frame, params)
        self.assertTrue(np.isfinite(fuel).all())
        self.assertTrue(np.all((fuel >= 0) & (fuel <= 24000)))


if __name__ == "__main__":
    unittest.main()

