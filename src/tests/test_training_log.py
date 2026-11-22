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
            "white_elo": 1216.0,
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
        self.assertEqual(row["white_elo"], "1216.0")
        self.assertEqual(
            json.loads(row["timing_profile"]),
            {"White": {"action_prepare": {"calls": 2}}},
        )

    def test_elo_update_is_zero_sum_and_draw_preserves_equal_ratings(self):
        from plot_logging.elo import update_elo

        white, black = update_elo(1200.0, 1200.0, 1.0)
        self.assertEqual((white, black), (1216.0, 1184.0))
        self.assertEqual(sum((white, black)), 2400.0)
        self.assertEqual(update_elo(1200.0, 1200.0, 0.5), (1200.0, 1200.0))


if __name__ == "__main__":
    unittest.main()
