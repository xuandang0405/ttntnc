import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from plot_logging.plot_training import save_training_plot
from plot_logging.training_log import TrainingLog


class TrainingLogTests(unittest.TestCase):
    def test_save_writes_csv_rows_with_metadata_and_flat_metrics(self):
        log = TrainingLog(started=0, interval=5)
        log.records.append({
            "time_seconds": 12.5,
            "white_actor_loss": 0.25,
            "timing_profile": {"White": {"action_prepare": {"calls": 2}}},
        })
        experiment = SimpleNamespace(id="exp_test", resumed_from=None)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "training_log.csv")
            log.save(path, experiment, {"algorithms": {"White": "A2C"}})

            with open(path, newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            plot_path = os.path.join(temp_dir, "training_log.jpg")
            save_training_plot(path, output_path=plot_path)
            self.assertTrue(os.path.isfile(plot_path))

        self.assertEqual(row["experiment"], "exp_test")
        self.assertIn("A2C", row["title"])
        self.assertEqual(row["time_seconds"], "12.5")
        self.assertEqual(row["white_actor_loss"], "0.25")
        self.assertEqual(
            json.loads(row["timing_profile"]),
            {"White": {"action_prepare": {"calls": 2}}},
        )


if __name__ == "__main__":
    unittest.main()
