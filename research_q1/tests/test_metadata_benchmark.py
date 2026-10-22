import unittest

import numpy as np
import pandas as pd

from research_q1.baselines.metadata_benchmark import design, type_rate


class MetadataBenchmarkTests(unittest.TestCase):
    def test_design_uses_training_aircraft_dictionary(self):
        train = pd.DataFrame({name: [1., 2.] for name in [
            "duration_min", "flight_duration_min", "elapsed_from_takeoff_min",
            "remaining_to_landing_min", "start_hour_sin", "start_hour_cos"]})
        train["aircraft_type"] = ["A", "B"]
        test = train.iloc[[0]].copy()
        test["aircraft_type"] = "UNSEEN"
        a, b, columns = design(train, test)
        type_columns = [index for index, name in enumerate(columns) if name.startswith("type_")]
        self.assertEqual(a.shape[1], b.shape[1])
        self.assertTrue(np.all(b[:, type_columns] == 0))

    def test_type_rate_has_global_unseen_fallback(self):
        train = pd.DataFrame({"fuel_kg": [10., 20.], "duration_min": [1., 2.],
                              "aircraft_type": ["A", "A"]})
        test = pd.DataFrame({"duration_min": [3.], "aircraft_type": ["B"]})
        self.assertAlmostEqual(type_rate(train, test)[0], 30.)


if __name__ == "__main__":
    unittest.main()

