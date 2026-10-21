"""Check the decision rule, its accounting, and browser source contract."""
import unittest

import numpy as np
import pandas as pd

from src.flight_fuel.eligibility import (describe, flight_bootstrap_tradeoff,
                                         render, residual_profile, select)


class EligibilityTests(unittest.TestCase):
    def test_duration_and_aircraft_floors_hold_even_when_budget_is_insufficient(self):
        scores = np.arange(10, dtype=float)
        types = np.array(["A"] * 8 + ["B"] * 2)
        duration = np.array([2] * 7 + [20] * 3)
        tie = np.arange(10) / 10
        mask, detail = select(scores, types, duration, tie, target=.5, floor=.5, min_type_n=2)
        self.assertGreaterEqual(mask[:8].sum(), 4)
        self.assertGreaterEqual(mask[8:].sum(), 1)
        self.assertGreaterEqual(mask[7:].sum(), 2)
        self.assertEqual(mask.sum(), 6)
        self.assertTrue(detail["floor_forced_over_budget"])
        self.assertEqual(detail["protected_aircraft_types"], ["B", "A"])
        np.testing.assert_array_equal(mask, select(scores, types, duration, tie, .5, .5, 2)[0])
        self.assertEqual(select(scores, types, duration, tie, 1, .5, 2)[0].sum(), 10)

    def test_selection_never_receives_labels_and_accounting_reconciles(self):
        frame = pd.DataFrame({"flight_id": ["f1", "f1", "f2"],
                              "aircraft_type": ["A", "A", "B"],
                              "duration_min": [2., 6., 20.],
                              "adsb_position_count": [3, 0, 2],
                              "fuel_kg": [10., 100., 500.]})
        scores = np.array([1., 2., 3.])
        mask, policy = select(scores, frame.aircraft_type, frame.duration_min,
                              np.array([.1, .2, .3]), .5, 0, 2)
        result = describe(frame, np.array([12., 110., 490.]), scores, mask, policy)
        self.assertEqual(result["accepted"], 2)
        self.assertAlmostEqual(result["accepted_reference_fuel_fraction"], 110 / 610)
        self.assertAlmostEqual(result["accepted_mae_kg"], 6)
        self.assertEqual(result["groups"]["duration"]["15–60 min"]["accepted_mae_kg"], None)
        changed_labels = frame.assign(fuel_kg=[999., 999., 999.])
        np.testing.assert_array_equal(mask, select(scores, changed_labels.aircraft_type,
                                                    changed_labels.duration_min,
                                                    np.array([.1, .2, .3]), .5, 0, 2)[0])

    def test_dashboard_embedding_rejects_extra_markers_and_escapes_script(self):
        self.assertIn("\\u003c/script", render("<script>/*__AUDIT_DATA__*/</script>",
                                               {"text": "</script>"}))
        with self.assertRaises(ValueError):
            render("no marker", {})

    def test_residual_diagnostics_and_flight_cluster_bootstrap(self):
        frame = pd.DataFrame({"flight_id": ["f1", "f1", "f2", "f2"],
                              "duration_min": [1., 2., 6., 20.],
                              "fuel_kg": [1., 2., 10., 20.]})
        predicted = np.array([2., 4., 8., 16.])
        profile = residual_profile(frame, predicted, np.array([1., 2., 3., 4.]))
        self.assertEqual(profile["full"]["n_intervals_under_5min"], 2)
        self.assertAlmostEqual(profile["full"]["mae_kg"], 2.25)
        self.assertAlmostEqual(profile["full"]["wape"], 9 / 33)
        self.assertEqual(sum(d["n"] for d in profile["risk_deciles"]), 4)
        with self.assertRaises(ValueError):
            residual_profile(frame.assign(fuel_kg=[0., 2., 10., 20.]),
                             predicted, np.ones(4))
        risk = np.array([True, False, True, False])
        floor = np.array([True, True, True, False])
        bootstrap = flight_bootstrap_tradeoff(frame, predicted, risk, floor, 100)
        self.assertEqual(bootstrap["resampling_unit"], "flight")
        self.assertEqual(bootstrap["valid_resamples"], 100)
        self.assertGreater(bootstrap["floor_minus_risk_labeled_fuel_share_95pct_ci"][1], 0)


if __name__ == "__main__":
    unittest.main()
