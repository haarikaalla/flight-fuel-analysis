"""Checks the leakage boundary and train/rank-only evaluation contract."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.flight_fuel.baselines import design, load_split, type_rate_predictions
from src.flight_fuel.features import features_for_flight


class Phase3Tests(unittest.TestCase):
    def test_trajectory_uses_adsb_only_and_excludes_interval_end(self):
        t = pd.Timestamp("2025-04-12 00:00:00")
        trajectory = pd.DataFrame({
            "timestamp": [t, t + pd.Timedelta(seconds=30), t + pd.Timedelta(minutes=1)],
            "flight_id": ["f1"] * 3, "source": ["adsb", "acars", "adsb"],
            "latitude": [10., 10., 10.], "longitude": [20., 20., 20.],
            "altitude": [30000., 999999., 31000.],
            "groundspeed": [450., 999999., 460.],
            "vertical_rate": [0., 999999., 0.],
            "mach": [np.nan, .8, np.nan], "TAS": [np.nan, 250., np.nan],
        })
        intervals = pd.DataFrame({"idx": [0, 1], "start": [t, t + pd.Timedelta(minutes=1)],
                                  "end": [t + pd.Timedelta(minutes=1), t + pd.Timedelta(minutes=2)]})
        result = features_for_flight(trajectory, intervals, "f1")
        self.assertEqual(result.adsb_position_count.tolist(), [1, 1])
        self.assertEqual(result.adsb_altitude_median_raw.tolist(), [30000., 31000.])
        self.assertFalse(any(name in result.columns for name in ("mach", "TAS", "CAS", "fuel_kg")))

    def test_zero_adsb_retained_with_missing_stats(self):
        t = pd.Timestamp("2025-04-12")
        trajectory = pd.DataFrame({"timestamp": [t], "flight_id": ["f1"], "source": ["acars"],
            "latitude": [1.], "longitude": [2.], "altitude": [30000.],
            "groundspeed": [400.], "vertical_rate": [0.]})
        intervals = pd.DataFrame({"idx": [3], "start": [t], "end": [t + pd.Timedelta(minutes=5)]})
        row = features_for_flight(trajectory, intervals, "f1").iloc[0]
        self.assertEqual(row.adsb_position_count, 0)
        self.assertTrue(np.isnan(row.adsb_speed_median_raw))
        self.assertEqual(row.adsb_max_gap_s, 300.)

    def test_unknown_type_fallback_and_final_period_lock(self):
        train = pd.DataFrame({"fuel_kg": [100., 200.], "duration_min": [5., 10.], "aircraft_type": ["A320", "A320"]})
        rank = pd.DataFrame({"duration_min": [5.], "aircraft_type": ["B734"]})
        self.assertAlmostEqual(type_rate_predictions(train, rank)[0], 100.)
        a, b, columns = design(train, rank, False)
        self.assertEqual(a.shape[1], b.shape[1])
        self.assertFalse(any("B734" in name for name in columns))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                load_split(Path(temp), "final")


if __name__ == "__main__":
    unittest.main()
