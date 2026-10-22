import unittest

import numpy as np

from research_q1.novel.conformal import (HierarchicalMondrian, benjamini_hochberg,
                                         bonferroni_risk_control_threshold,
                                         finite_quantile)
from research_q1.novel.selection import exact_risk_budget, robust_risk_budget


class NovelMethodTests(unittest.TestCase):
    def test_mondrian_uses_hierarchy_and_physical_bounds(self):
        scores = np.r_[np.linspace(.1, 1, 60), np.linspace(.2, 1.2, 60)]
        aircraft = np.array(["A"] * 60 + ["B"] * 60)
        phase = np.array(["cruise"] * 120)
        model = HierarchicalMondrian(alpha=.1, min_group_n=50).fit(scores, aircraft, phase)
        lo, hi, audit = model.interval(np.array([10., 20.]), np.ones(2),
                                       np.array(["A", "UNSEEN"]), np.array(["cruise", "new"]),
                                       np.array([10.5, 25.]))
        self.assertTrue(np.all(lo >= 0))
        self.assertTrue(np.all(hi <= np.array([10.5, 25.])))
        self.assertEqual(audit["fallback_counts"]["global"], 1)

    def test_exact_optimizer_dominates_feasible_greedy_example(self):
        utility = np.array([9., 8., 7., 6.])
        risk = np.array([5., 4., 3., 2.])
        selected, audit = exact_risk_budget(utility, risk, budget=7, minimum_coverage=.5)
        self.assertGreaterEqual(selected.sum(), 2)
        self.assertLessEqual(risk[selected].sum(), 7 + 1e-9)
        self.assertEqual(audit["method"], "exact_milp")

    def test_robust_budget_accounts_for_top_deviation(self):
        utility = np.array([5., 4., 3.])
        risk = np.array([1., 1., 1.])
        deviation = np.array([2., .5, .2])
        selected, audit = robust_risk_budget(utility, risk, deviation, 3.5, 1/3, gamma=1)
        self.assertLessEqual(audit["robust_risk_bound"], 3.5 + 1e-6)
        self.assertTrue(selected.any())

    def test_bh_and_risk_control_are_explicit(self):
        selected = benjamini_hochberg(np.array([.001, .02, .8]), .05)
        self.assertTrue(selected[0])
        result = bonferroni_risk_control_threshold(
            np.arange(20), np.zeros(20, dtype=int), np.array([5, 10, 19]), .5, .1)
        self.assertIsNotNone(result["chosen"])
        self.assertIn("iid calibration", result["guarantee_scope"])

    def test_finite_quantile_is_conservative(self):
        self.assertEqual(finite_quantile(np.arange(1, 11), .1), 10)


if __name__ == "__main__":
    unittest.main()

