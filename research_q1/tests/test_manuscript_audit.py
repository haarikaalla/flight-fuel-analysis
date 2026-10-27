import unittest
import numpy as np
from research_q1.validation.manuscript_audit import block_crc_threshold, inventory_metrics


class ManuscriptAuditTests(unittest.TestCase):
    def test_flight_weighting_does_not_treat_long_flights_as_independent(self):
        scores = np.zeros(101)
        events = np.r_[np.ones(100), 0]
        flights = np.concatenate([np.repeat('long', 100), ['short']])
        threshold, losses = block_crc_threshold(scores, events, flights, .7, [-1, 1])
        self.assertAlmostEqual(losses[1], 2/3)
        self.assertEqual(threshold, 1)

    def test_small_risk_sample_can_have_no_feasible_threshold(self):
        threshold, _ = block_crc_threshold(np.zeros(2), np.zeros(2), ['a', 'b'], .05, [-1, 1])
        self.assertIsNone(threshold)

    def test_same_predictor_imputation_preserves_inventory(self):
        y, p, z = np.array([10.,20.,30.]), np.array([11.,19.,31.]), np.array([True,False,True])
        a = inventory_metrics(y, p, ['a','a','b'], 100, 42)
        b = inventory_metrics(y, np.where(z,p,p), ['a','a','b'], 100, 42)
        self.assertEqual(a, b)

if __name__ == '__main__':
    unittest.main()
