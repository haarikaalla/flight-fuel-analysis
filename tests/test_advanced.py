"""Tests for the optimization-based research extension."""

import unittest

import numpy as np

from src.flight_fuel.advanced import constrained_select, decision_reasons, render
from src.flight_fuel.eligibility import duration_groups, select


class AdvancedSelectionTests(unittest.TestCase):
    def test_exact_selector_meets_budget_and_all_group_floors(self):
        scores = np.array([1, 2, 8, 9, 3, 4, 10, 11], dtype=float)
        types = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        duration = np.array([2, 2, 20, 20, 2, 2, 20, 20], dtype=float)
        ties = np.arange(8, dtype=float) / 8
        mask, details = constrained_select(
            scores, types, duration, ties, target=0.5, floor=0.5, min_type_n=2
        )
        self.assertEqual(mask.sum(), 4)
        self.assertEqual(details["selected_count"], 4)
        for group in ("A", "B"):
            self.assertGreaterEqual(mask[types == group].sum(), 2)
        groups = duration_groups(duration)
        for group in set(groups):
            self.assertGreaterEqual(mask[groups == group].sum(), 2)

    def test_optimizer_is_no_worse_than_greedy_on_predicted_risk(self):
        scores = np.array([1, 100, 2, 90, 3, 80, 4, 70], dtype=float)
        types = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        duration = np.array([2, 20, 2, 20, 2, 20, 2, 20], dtype=float)
        ties = np.linspace(0, .7, 8)
        greedy, _ = select(scores, types, duration, ties, .75, .5, 2)
        exact, _ = constrained_select(scores, types, duration, ties, .75, .5, 2)
        self.assertEqual(greedy.sum(), exact.sum())
        self.assertLessEqual(scores[exact].sum(), scores[greedy].sum() + 1e-9)

    def test_decision_explanations_distinguish_rank_and_constraint(self):
        selected = np.array([True, True, False])
        risk_only = np.array([True, False, False])
        reasons = decision_reasons(
            selected,
            risk_only,
            np.array(["A", "B", "B"]),
            np.array([2., 20., 20.]),
            min_type_n=2,
        )
        self.assertIn("low predicted-error", reasons[0])
        self.assertIn("coverage-constrained", reasons[1])
        self.assertIn("not selected", reasons[2])

    def test_dashboard_embedding_escapes_script_content(self):
        html = render("<script>/*__ADVANCED_DATA__*/</script>", {"x": "</script>"})
        self.assertIn("\\u003c/script", html)
        with self.assertRaises(ValueError):
            render("no marker", {})


if __name__ == "__main__":
    unittest.main()
