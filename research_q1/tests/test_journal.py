import unittest

import numpy as np
import pandas as pd

from research_q1.journal import extract, study


class ExtractTests(unittest.TestCase):
    def test_clean_adsb_converts_units_and_drops_acars(self):
        t = pd.to_datetime(["2025-09-01 00:00:00", "2025-09-01 00:00:01", "2025-09-01 00:00:02"])
        frame = pd.DataFrame({"timestamp": t, "flight_id": ["f"] * 3, "source": ["adsb", "acars", "adsb"],
                              "latitude": [50.0, 50.0, 50.001], "longitude": [8.0, 8.0, 8.001],
                              "altitude": [35000.0, 35000.0, 99999.0], "groundspeed": [450.0, np.nan, 450.0],
                              "vertical_rate": [0.0, 0.0, 0.0]})
        clean = extract.clean_adsb(frame)
        self.assertEqual(len(clean), 2)
        self.assertAlmostEqual(clean.altitude.iloc[0], 35000 * 0.3048)
        self.assertAlmostEqual(clean.groundspeed.iloc[0], 450 * 0.514444)
        self.assertTrue(np.isnan(clean.altitude.iloc[1]))  # out-of-range altitude removed

    def test_track_km_ignores_position_spikes(self):
        t = np.arange(0, 60, 1.0)
        lat = np.full(60, 50.0)
        lon = 8.0 + np.arange(60) * 0.002  # ~0.14 km/s ≈ 280 kt
        lat[30], lon[30] = 20.0, 8.06  # single impossible spike
        clean = extract.track_km(t, lat, lon)
        expected = extract.haversine_km(np.full(60, 50.0), 8.0 + np.arange(60) * 0.002).sum()
        self.assertLess(abs(clean - expected) / expected, 0.05)

    def test_openap_fuel_flow_is_positive_and_phase_sensitive(self):
        gs = np.array([0.0, 120.0, 230.0])
        alt = np.array([0.0, 2000.0, 11000.0])
        vr = np.array([0.0, 10.0, 0.0])
        flow = extract.openap_fuel_flow("A320", gs, alt, vr)
        self.assertTrue(np.all(flow > 0))
        self.assertGreater(flow[1], flow[2])  # climb burns more than cruise

    def test_kinematics_for_flight_handles_no_adsb(self):
        frame = pd.DataFrame({"timestamp": pd.to_datetime(["2025-09-01 00:00:00"]), "flight_id": ["f"],
                              "source": ["acars"], "latitude": [50.0], "longitude": [8.0], "altitude": [1.0],
                              "groundspeed": [1.0], "vertical_rate": [0.0]})
        intervals = pd.DataFrame({"idx": [1], "start": pd.to_datetime(["2025-09-01 00:00:00"]),
                                  "end": pd.to_datetime(["2025-09-01 00:05:00"])})
        rows = extract.kinematics_for_flight(frame, intervals, "f", "A320")
        self.assertEqual(len(rows), 1)
        self.assertTrue(np.isnan(rows.phys_openap_kg.iloc[0]))


class StudyTests(unittest.TestCase):
    def test_paired_test_detects_sign(self):
        rng = np.random.default_rng(0)
        a = rng.gamma(2, 10, 400)
        b = a + 5
        clusters = np.repeat(np.arange(40), 10)
        out = study.paired_test(a, b, clusters, 200, 1)
        self.assertGreater(out["ci95"][0], 0)
        self.assertAlmostEqual(out["mae_difference_kg"], 5.0)

    def test_physics_scaling_recovers_known_scale(self):
        train = pd.DataFrame({"aircraft_type": ["A"] * 50 + ["B"] * 50,
                              "phys_openap_kg": np.r_[np.linspace(10, 100, 50), np.linspace(10, 100, 50)]})
        train["fuel_kg"] = train.phys_openap_kg * np.where(train.aircraft_type == "A", 0.8, 1.2)
        scales, global_scale = study.physics_scaling(train)
        self.assertAlmostEqual(scales["A"], 0.8)
        self.assertAlmostEqual(scales["B"], 1.2)
        self.assertAlmostEqual(global_scale, 1.0)

    def test_design_aligns_unseen_type_to_zero_dummies(self):
        cols = study.META_NUMERIC
        train = pd.DataFrame({c: [1.0, 2.0] for c in cols} | {"aircraft_type": ["A", "B"]})
        test = pd.DataFrame({c: [3.0] for c in cols} | {"aircraft_type": ["Z"]})
        x, names = study.design(train, test, ["meta"])
        self.assertEqual(x.shape, (1, len(cols) + 2))
        self.assertEqual(x[0, len(cols):].sum(), 0)


if __name__ == "__main__":
    unittest.main()
