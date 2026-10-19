"""Small fixture checks the ADS-B boundary and zero-input behavior."""

import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.flight_fuel import full_coverage_audit as coverage
from src.flight_fuel.data_audit import checksum


class FullCoverageAuditTests(unittest.TestCase):
    def test_acars_and_interval_end_do_not_count_as_adsb_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp)
            flight_id = "flight1"
            date = pd.Timestamp("2025-04-12 00:00:00")
            pd.DataFrame({"flight_id": [flight_id], "aircraft_type": ["A320"]}).to_parquet(raw / "flightlist_train.parquet")
            pd.DataFrame({
                "idx": [0, 1], "flight_id": [flight_id, flight_id],
                "start": [date, date + pd.Timedelta(minutes=2)],
                "end": [date + pd.Timedelta(minutes=2), date + pd.Timedelta(minutes=4)],
                "fuel_kg": [100.0, 150.0],
            }).to_parquet(raw / "fuel_train.parquet")
            trajectory = pd.DataFrame({
                "timestamp": [date + pd.Timedelta(minutes=1), date + pd.Timedelta(minutes=2), date + pd.Timedelta(minutes=3)],
                "flight_id": [flight_id] * 3,
                "latitude": [10.0] * 3, "longitude": [20.0] * 3,
                "altitude": [30000.0] * 3, "groundspeed": [400.0] * 3,
                "track": [80.0] * 3, "vertical_rate": [0.0] * 3,
                "source": ["acars", "adsb", "acars"],
            })
            data = io.BytesIO()
            trajectory.to_parquet(data)
            archive = raw / "flights_train.zip"
            with zipfile.ZipFile(archive, "w") as out:
                out.writestr("flights_train/flight1.parquet", data.getvalue())
            with patch.object(coverage, "SPLITS", ("train",)), patch.dict(
                coverage.ARCHIVES, {"flights_train.zip": (archive.stat().st_size, checksum(archive))}
            ):
                totals = coverage.summarize(raw)["splits"]["train"]["totals"]
            self.assertEqual(totals["acars_rows"], 2)
            self.assertEqual(totals["zero_adsb_intervals"], 1)
            self.assertEqual(totals["one_to_five_adsb_intervals"], 1)
            self.assertEqual(totals["flights_with_any_zero_adsb_interval"], 1)
            self.assertEqual(totals["intervals_with_max_observation_gap_over_60s"], 2)


if __name__ == "__main__":
    unittest.main()
