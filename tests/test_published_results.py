"""Keep the portfolio summary tied to the committed evaluation artifact."""

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublishedResultsTests(unittest.TestCase):
    def test_readme_summary_matches_committed_validation_results(self):
        report = json.loads((ROOT / "reports/eligibility-results.json").read_text())
        readme = (ROOT / "README.md").read_text()
        validation = report["periods"]["validation"]
        risk = validation["policies"]["risk_only"]
        floor = validation["policies"]["representation_floor"]
        residual = validation["residual_diagnostics"]["full"]

        self.assertIn(f"{validation['intervals']:,} September validation intervals", readme)
        self.assertIn(f"**{risk['coverage'] * 100:.2f}%** overall acceptance", readme)
        claims = {
            "Mean absolute error on accepted intervals": (
                f"{risk['accepted_mae_kg']:.2f} kg", f"{floor['accepted_mae_kg']:.2f} kg"
            ),
            "15–60-minute intervals accepted": (
                f"{risk['groups']['duration']['15–60 min']['coverage'] * 100:.2f}%",
                f"{floor['groups']['duration']['15–60 min']['coverage'] * 100:.2f}%",
            ),
            "Boeing 747-400 (B744) intervals accepted": (
                f"{risk['groups']['aircraft_type']['B744']['accepted']} of "
                f"{risk['groups']['aircraft_type']['B744']['n']}",
                f"{floor['groups']['aircraft_type']['B744']['accepted']} of "
                f"{floor['groups']['aircraft_type']['B744']['n']}",
            ),
            "Share of this sample's labeled fuel represented": (
                f"{risk['accepted_reference_fuel_fraction'] * 100:.2f}%",
                f"{floor['accepted_reference_fuel_fraction'] * 100:.2f}%",
            ),
        }
        for label, values in claims.items():
            with self.subTest(label=label):
                pattern = rf"\| {re.escape(label)} \| {re.escape(values[0])} \| {re.escape(values[1])} \|"
                self.assertRegex(readme, pattern)
        for value in (
            f"**{residual['mae_kg']:.2f} kg**",
            f"**{residual['rmse_kg']:.2f} kg**",
            f"**{residual['r2']:.3f}**",
            f"**{residual['p95_absolute_error_kg']:.2f} kg**",
        ):
            self.assertIn(value, readme)


if __name__ == "__main__":
    unittest.main()
