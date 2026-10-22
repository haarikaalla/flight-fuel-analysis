"""Tests for optimization, uncertainty calibration and dashboard generation."""

import unittest

import numpy as np

from src.flight_fuel.advanced import (
    constrained_select,
    cross_fitted_mondrian_uncertainty,
    cross_fitted_uncertainty,
    decision_reasons,
    leave_aircraft_out_risk_calibration,
    render,
)
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

    def test_cross_fitted_uncertainty_is_finite_and_out_of_fold(self):
        risk = np.tile(np.arange(1, 11, dtype=float), 5)
        truth = np.linspace(100, 400, 50)
        signed_error = np.where(np.arange(50) % 2, risk * 1.8, -risk * 1.8)
        predicted = truth + signed_error
        ties = (np.arange(50) + .5) / 50
        calibrated, half_width, diagnostics = cross_fitted_uncertainty(
            risk, predicted, truth, ties, folds=5, interval_level=.9
        )
        self.assertEqual(calibrated.shape, risk.shape)
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue((calibrated >= 0).all())
        self.assertTrue((half_width > 0).all())
        self.assertEqual(diagnostics["folds"], 5)
        self.assertGreaterEqual(diagnostics["observed_interval_coverage"], .8)

    def test_mondrian_uncertainty_reports_group_coverage_and_fallback(self):
        risk = np.tile(np.arange(1, 11, dtype=float), 6)
        truth = np.linspace(100, 500, 60)
        predicted = truth + np.where(np.arange(60) % 2, risk * 2, -risk * 2)
        ties = (np.arange(60) + .5) / 60
        groups = np.where(np.arange(60) < 50, "large", "small")
        half_width, diagnostics = cross_fitted_mondrian_uncertainty(
            risk, predicted, truth, ties, groups,
            group_label="synthetic", folds=5, min_group_n=12,
        )
        self.assertEqual(half_width.shape, risk.shape)
        self.assertTrue((half_width > 0).all())
        self.assertEqual(len(diagnostics["groups"]), 2)
        small = next(row for row in diagnostics["groups"] if row["group"] == "small")
        self.assertEqual(small["fallback_fraction"], 1.0)

    def test_aircraft_holdout_calibrates_without_held_group_labels(self):
        aircraft = np.repeat(["A", "B", "C"], 20)
        risk = np.tile(np.arange(1, 21, dtype=float), 3)
        truth = np.linspace(100, 700, 60)
        predicted = truth + np.where(np.arange(60) % 2, risk, -risk)
        result = leave_aircraft_out_risk_calibration(
            risk, predicted, truth, aircraft, min_type_n=20
        )
        self.assertEqual(result["eligible_aircraft_types"], 3)
        self.assertEqual(result["rows_evaluated"], 60)
        self.assertEqual(len(result["aircraft"]), 3)
        self.assertGreaterEqual(result["pooled_interval_coverage"], .8)


if __name__ == "__main__":
    unittest.main()
