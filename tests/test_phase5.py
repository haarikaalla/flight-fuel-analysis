"""Published-report integrity and offline-demo reproducibility checks."""

import copy
import json
import unittest
from pathlib import Path

from src.flight_fuel.phase5 import audit, dashboard_payload, render


ROOT = Path(__file__).resolve().parents[1]


class Phase5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / "reports/phase-4-results.json").read_text())

    def test_published_aggregates_reconcile_with_model_scores(self):
        result = audit(self.report)
        self.assertEqual(result, json.loads((ROOT / "reports/phase-5-robustness.json").read_text()))

    def test_inconsistent_report_is_rejected(self):
        damaged = copy.deepcopy(self.report)
        damaged["periods"]["final"]["selectors"]["learned_risk"]["0.75"]["accepted_intervals"] += 1
        with self.assertRaisesRegex(ValueError, "acceptance fraction"):
            audit(damaged)
        damaged = copy.deepcopy(self.report)
        damaged["periods"]["final"]["label_granularity"]["duration_bin"]["<5 min"]["lattice_mae_kg"] += 10
        with self.assertRaisesRegex(ValueError, "duration reconstruction"):
            audit(damaged)

    def test_offline_demo_matches_source_and_escapes_script_text(self):
        template = (ROOT / "demo/template.html").read_text()
        html = render(template, dashboard_payload(self.report, audit(self.report)))
        self.assertEqual(html, (ROOT / "demo/index.html").read_text())
        self.assertNotIn("/*__PHASE5_DATA__*/", html)
        self.assertNotIn("https://cdn", html)
        self.assertIn("\\u003c/script", render("<script>/*__PHASE5_DATA__*/</script>",
                                                {"unsafe": "</script>"}))


if __name__ == "__main__":
    unittest.main()
