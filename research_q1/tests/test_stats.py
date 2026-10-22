import unittest

import numpy as np

from research_q1.stats.inference import adjust_pvalues, cluster_bootstrap_difference


class StatisticsTests(unittest.TestCase):
    def test_cluster_bootstrap_keeps_cluster_unit(self):
        a = np.array([1., 2., 3., 4.])
        b = a + 1
        result = cluster_bootstrap_difference(a, b, np.array(["A", "A", "B", "B"]),
                                              repetitions=100)
        self.assertAlmostEqual(result["mean_difference_b_minus_a"], 1.0)
        self.assertEqual(result["clusters"], 2)

    def test_multiple_testing_adjustments_are_valid(self):
        p = np.array([.01, .03, .2])
        for method in ("holm", "bh"):
            adjusted = adjust_pvalues(p, method)
            self.assertTrue(np.all((adjusted >= p) & (adjusted <= 1)))


if __name__ == "__main__":
    unittest.main()

