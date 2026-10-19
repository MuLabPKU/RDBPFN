"""Regression checks for cache-to-leaderboard row and class alignment."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src.predict_relbench import align_rows, inspect_task, key_frame, positive_class


class AlignmentTests(unittest.TestCase):
    def test_reorders_by_entity_and_time_and_excludes_obsolete_rows(self):
        local = pd.DataFrame({"id": [7, 7, 8], "time": pd.to_datetime(["2020-02-01", "2020-01-01", "2020-02-01"])})
        official = local.iloc[[2, 0]].reset_index(drop=True)
        positions, summary = align_rows(local, official)
        np.testing.assert_array_equal(positions, [2, 0])
        self.assertEqual(summary["excluded_cached_rows"], 1)
        self.assertEqual(summary["missing_official_rows"], 0)

    def test_duplicate_keys_rejected(self):
        keys = pd.DataFrame({"id": [1, 1], "time": [1, 1]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            align_rows(keys, keys.iloc[:1])

    def test_positive_class_can_be_local_zero(self):
        self.assertEqual(positive_class(np.array([1, 0, 1, 0]), np.array([0, 1, 0, 1])), 0)
        self.assertEqual(positive_class(np.array([0, 1]), np.array([0, 1])), 1)

    def test_inconsistent_labels_cannot_be_treated_as_an_encoding_flip(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            positive_class(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0]))

    def test_entity_alias_and_nanosecond_time(self):
        metadata = {"columns": [{"name": "user_id", "dtype": "foreign_key"}], "time_column": "timestamp"}
        official = {"entity_col": "user", "time_col": "timestamp"}
        frame = key_frame({"user_id": np.array([3]), "timestamp": np.array([1354147200000000000])}, metadata, official)
        self.assertEqual(list(frame.columns), ["user", "timestamp"])
        self.assertEqual(frame.iloc[0]["timestamp"], pd.Timestamp("2012-11-29"))

    def test_audit_uses_train_labels_and_rejects_missing_test_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "rel-event-dfs-2"
            (folder / "user-ignore").mkdir(parents=True)
            task = {
                "name": "user-ignore", "task_type": "classification", "format": "numpy",
                "source": "user-ignore/{split}.npz", "target_column": "old_target", "time_column": "timestamp",
                "columns": [{"name": "user_id", "dtype": "foreign_key"},
                            {"name": "feature", "dtype": "float"},
                            {"name": "old_target", "dtype": "category"}],
            }
            (folder / "metadata.yaml").write_text(yaml.safe_dump({"dataset_name": "rel-event-dfs-2", "tasks": [task]}))
            train_time = pd.Timestamp("2010-01-01").value
            test_time = pd.Timestamp("2011-01-01").value
            np.savez(folder / "user-ignore/train.npz", user_id=[2, 1, 3], timestamp=[train_time] * 3, old_target=[0, 1, 1], feature=[20, 10, 30])
            # There is deliberately NO test label array.
            np.savez(folder / "user-ignore/test.npz", user_id=[2, 1, 3], timestamp=[test_time] * 3, feature=[20, 10, 30])
            official = {"entity_col": "user", "time_col": "timestamp", "target_col": "target"}
            train = pd.DataFrame({"user": [1, 2], "timestamp": pd.to_datetime([train_time] * 2), "target": [0, 1]})
            test = pd.DataFrame({"user": [1, 2], "timestamp": pd.to_datetime([test_time] * 2)})
            with patch("src.predict_relbench.get_official", return_value=(official, train, test)):
                summary, prepared = inspect_task(root, "rel-event/user-ignore", "revision")
            self.assertEqual(summary["local_positive_class"], 0)
            np.testing.assert_array_equal(prepared[3], [0, 1])
            np.testing.assert_array_equal(prepared[4], [1, 0])
            missing = test.copy()
            missing.loc[0, "user"] = 99
            with patch("src.predict_relbench.get_official", return_value=(official, train, missing)):
                with self.assertRaisesRegex(ValueError, "Missing official test rows"):
                    inspect_task(root, "rel-event/user-ignore", "revision")
            (folder / "user-ignore/test.npz").rename(folder / "user-ignore/validation.npz")
            # Validation inspection must work even when no test file exists.
            with patch("src.predict_relbench.get_official", return_value=(official, train, test)) as fetch:
                summary, _ = inspect_task(root, "rel-event/user-ignore", "revision", split="val")
                fetch.assert_called_once_with("rel-event", "user-ignore", "revision", "val")
                self.assertIn("val", summary)
                self.assertNotIn("test", summary)


if __name__ == "__main__":
    unittest.main()
