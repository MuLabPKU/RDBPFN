"""Reject mixed-run and changed predictions before packaging."""

import json
import tempfile
import unittest
from pathlib import Path

from relbench.hf import RELBENCH_HF
from relbench.submit import LEADERBOARD_TASKS

from src.package_relbench_context import collect_sources
from src.predict_relbench import sha256


class PackagingTests(unittest.TestCase):
    def make_run(self, root, tasks, seed=0):
        root.mkdir()
        rows = []
        for task in tasks:
            path = root / task.replace("/", "__") / "4096x8.test.csv"
            path.parent.mkdir()
            path.write_text("id,time,target\n1,2020-01-01,0.5\n")
            rows.append({"task": task, "configuration": "4096x8",
                         "prediction_sha256": sha256(path)})
        report = {"status": "complete", "split": "test",
                  "official_repo": RELBENCH_HF, "official_revision": "pinned-revision",
                  "checkpoint_sha256": "weights", "model_config_sha256": "config",
                  "code_sha256": {"eval_relbench_context.py": "source"},
                  "dtype": "float32", "settings": {"seed": seed}, "results": rows}
        (root / "run.json").write_text(json.dumps(report))
        return root

    def test_disjoint_runs_combine_but_duplicates_and_missing_tasks_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = LEADERBOARD_TASKS["classification"]
            a = self.make_run(root / "a", tasks[:6])
            b = self.make_run(root / "b", tasks[6:])
            identity, sources = collect_sources([a, b], "4096x8")
            self.assertEqual(set(sources), set(tasks))
            self.assertEqual(identity["official_revision"], "pinned-revision")
            with self.assertRaisesRegex(ValueError, "Duplicate task"):
                collect_sources([a, a, b], "4096x8")
            with self.assertRaisesRegex(ValueError, "12 classification"):
                collect_sources([a], "4096x8")

    def test_mixed_seed_or_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = LEADERBOARD_TASKS["classification"]
            a = self.make_run(root / "a", tasks[:6])
            b = self.make_run(root / "b", tasks[6:], seed=1)
            with self.assertRaisesRegex(ValueError, "different data"):
                collect_sources([a, b], "4096x8")
            path = b / "run.json"
            report = json.loads(path.read_text())
            report["settings"]["seed"] = 0
            report["official_revision"] = "other-revision"
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "different data"):
                collect_sources([a, b], "4096x8")

    def test_changed_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks = LEADERBOARD_TASKS["classification"]
            root = self.make_run(Path(directory) / "run", tasks)
            path = root / tasks[0].replace("/", "__") / "4096x8.test.csv"
            path.write_text("id,time,target\n1,2020-01-01,0.9\n")
            with self.assertRaisesRegex(ValueError, "changed since inference"):
                collect_sources([root], "4096x8")


if __name__ == "__main__":
    unittest.main()
