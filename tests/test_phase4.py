import unittest

import numpy as np
import pandas as pd

from src.flight_fuel.phase4 import accepted, blocks, risk_design, threshold


class Phase4Tests(unittest.TestCase):
    def test_tied_quality_threshold_exact_calibration_coverage(self):
        scores = np.array([0.1, 1, 1, 1, 1], dtype=float)
        ties = np.array([0.1, 0.4, 0.2, 0.3, 0.5])
        self.assertEqual(accepted(scores, ties, threshold(scores, ties, 0.6)).sum(), 3)
        self.assertEqual(accepted(scores, ties, threshold(scores, ties, 1.0)).sum(), 5)
        self.assertTrue(accepted(np.array([10.0]), np.array([0.9]), threshold(scores, ties, 1.0))[0])

    def test_temporal_blocks_keep_flights_whole_and_exclude_boundary_crossings(self):
        rows = [
            ("a", "2025-09-05", "2025-09-05T00:05"),
            ("b", "2025-09-09T23:50", "2025-09-10T00:10"),
            ("c", "2025-09-14", "2025-09-14T00:05"),
            ("d", "2025-09-22", "2025-09-22T00:05"),
        ]
        frame = pd.DataFrame(rows, columns=["flight_id", "start", "end"])
        frame[["start", "end"]] = frame[["start", "end"]].apply(lambda s: pd.to_datetime(s, format="mixed"))
        result = blocks(frame, {"risk_train_before": "2025-09-10", "calibration_before": "2025-09-19"})
        self.assertEqual({k: v.flight_id.tolist() for k, v in result.items()},
                         {"risk_train": ["a"], "calibration": ["c"], "validation": ["d"]})

    def test_risk_design_never_uses_target_or_unseen_type_column(self):
        reference = pd.DataFrame({"aircraft_type": ["A320"], "duration_min": [3],
                                  "adsb_position_count": [0], "adsb_speed_count": [0],
                                  "adsb_max_gap_fraction": [1], "fuel_kg": [1]})
        holdout = reference.copy()
        holdout["aircraft_type"] = "UNSEEN"
        holdout["fuel_kg"] = 999999
        x, y, columns = risk_design(reference, holdout, np.array([50]), np.array([50]))
        self.assertEqual(columns[-1], "type_A320")
        self.assertEqual(y[0, -1], 0)
        np.testing.assert_array_equal(y, risk_design(reference, holdout.assign(fuel_kg=2),
                                                        np.array([50]), np.array([50]))[1])


if __name__ == "__main__":
    unittest.main()
