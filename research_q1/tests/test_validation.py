from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from research_q1.validation.data_contract import FrozenDataAccessError, guard_confirmation_access
from research_q1.validation.metrics import interval_metrics
from research_q1.validation.labels import label_audit
from research_q1.validation.nested_cv import grouped_nested_cv
from research_q1.validation.splits import grouped_folds, grouped_holdout, temporal_split


class ValidationTests(unittest.TestCase):
    def setUp(self):
        starts = pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC")
        self.frame = pd.DataFrame({
            "flight_id": np.repeat([f"f{i}" for i in range(6)], 2),
            "interval_start_utc": starts,
            "interval_end_utc": starts + pd.Timedelta(hours=1),
            "route": np.repeat(["A-B", "C-D", "E-F"], 4),
        })

    def test_temporal_split_keeps_flights_whole(self):
        split = temporal_split(self.frame, "2024-01-05", "2024-01-09")
        partitions = [set(self.frame.loc[mask, "flight_id"]) for mask in
                      (split.train, split.calibration, split.test)]
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])

    def test_grouped_holdout_has_no_route_leakage(self):
        split = grouped_holdout(self.frame, "route", test_fraction=0.34,
                                calibration_fraction=0.5)
        partitions = [set(self.frame.loc[mask, "route"]) for mask in
                      (split.train, split.calibration, split.test)]
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])

    def test_grouped_folds_never_divide_a_group(self):
        folds = grouped_folds(self.frame, "flight_id", n_splits=3)
        self.assertTrue(all(len(set(folds[self.frame.flight_id.eq(name)])) == 1
                            for name in self.frame.flight_id.unique()))

    def test_confirmation_guard_logs_and_refuses(self):
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "access.log"
            with self.assertRaises(FrozenDataAccessError):
                guard_confirmation_access(Path(tmp) / "future_confirmation.parquet",
                                          access_log=log)
            self.assertIn("DENIED", log.read_text())

    def test_interval_score_penalizes_misses(self):
        covered = interval_metrics(np.array([1.0]), np.array([0.0]), np.array([2.0]))
        missed = interval_metrics(np.array([4.0]), np.array([0.0]), np.array([2.0]))
        self.assertGreater(missed["winkler_score"], covered["winkler_score"])

    def test_label_audit_documents_source_and_lattice(self):
        frame = pd.DataFrame({"reference_fuel_kg": [0.45359237, 2.0],
                              "label_source": ["ACARS", "ACARS"]})
        audit = label_audit(frame)
        self.assertEqual(audit["n"], 2)
        self.assertEqual(audit["label_sources"]["ACARS"], 2)

    def test_nested_cv_keeps_grouped_oof_predictions(self):
        x = np.arange(24, dtype=float).reshape(-1, 1)
        y = x[:, 0] * 2
        groups = np.repeat([f"g{i}" for i in range(12)], 2)
        result = grouped_nested_cv(x, y, groups,
                                   lambda parameters: Ridge(**parameters),
                                   {"alpha": [.1, 1.]}, outer_folds=3, inner_folds=2)
        self.assertEqual(len(result["oof_prediction"]), 24)
        self.assertFalse(result["group_overlap"])


if __name__ == "__main__":
    unittest.main()

