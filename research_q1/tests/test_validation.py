from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from research_q1.validation.data_contract import FrozenDataAccessError, guard_confirmation_access
from research_q1.validation.metrics import interval_metrics
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


if __name__ == "__main__":
    unittest.main()

