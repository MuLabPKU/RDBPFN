import unittest

import numpy as np
import torch

from src.eval_relbench_context import context_pool, context_windows, predict_ensemble
from src.models import ModelConfig, build_classifier, build_model


class ContextTests(unittest.TestCase):
    def test_extended_pool_retains_original_four_member_experiment(self):
        for count in (1353, 19239, 50000):
            eligible = np.arange(count) * 3
            rng = np.random.default_rng(417)
            first = rng.choice(count, 1024, replace=False)
            remaining = np.ones(count, dtype=bool)
            remaining[first] = False
            extra = rng.choice(np.flatnonzero(remaining), min(15360, count - 1024), replace=False)
            original = eligible[np.concatenate([first, extra])]
            extended = context_pool(eligible, 32768, 417)
            np.testing.assert_array_equal(extended[:len(original)], original)
            np.testing.assert_array_equal(context_pool(eligible, 8192, 417), extended[:8192])
            self.assertEqual(len(extended), min(count, 32768))
            self.assertEqual(len(np.unique(extended)), len(extended))

    def test_pool_preserves_original_baseline_and_has_no_repeats(self):
        eligible = np.arange(20000) * 3
        seed = 417
        expected = eligible[np.random.default_rng(seed).choice(len(eligible), 1024, replace=False)]
        pool = context_pool(eligible, 16384, seed)
        np.testing.assert_array_equal(pool[:1024], expected)
        self.assertEqual(len(np.unique(pool)), len(pool))
        windows = context_windows(len(pool), 4096, 4)
        np.testing.assert_array_equal(np.concatenate(windows), np.arange(16384))
        np.testing.assert_array_equal(np.concatenate(context_windows(len(pool), 1024, 4)), windows[0])

    def test_small_task_full_coverage_without_repeated_windows(self):
        pool = context_pool(np.arange(1353), 16384, 0)
        self.assertEqual(len(pool), 1353)
        self.assertEqual([len(w) for w in context_windows(len(pool), 1024, 4)], [1024, 329])
        self.assertEqual([len(w) for w in context_windows(len(pool), 4096, 4)], [1353])

    def test_attention_optimization_and_query_chunking_preserve_predictions(self):
        torch.manual_seed(0)
        cfg = ModelConfig(embedding_size=16, num_attention_heads=4, mlp_hidden_size=32, num_layers=2)
        model = build_model(cfg).eval()
        rng = np.random.default_rng(0)
        context = rng.normal(size=(32, 7)).astype(np.float32)
        labels = np.arange(32) % 2
        queries = rng.normal(size=(11, 7)).astype(np.float32)
        classifier = build_classifier(model, torch.device("cpu"), cfg)
        classifier.fit(context, labels)
        reference = classifier.predict_proba(queries)
        old_fastpath = torch.backends.mha.get_fastpath_enabled()
        try:
            torch.backends.mha.set_fastpath_enabled(False)
            for layer in model.transformer_blocks:
                layer.memory_efficient_attention = True
            actual = classifier.predict_proba(queries)
            chunked = np.concatenate([classifier.predict_proba(q) for q in np.array_split(queries, 3)])
        finally:
            torch.backends.mha.set_fastpath_enabled(old_fastpath)
        np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(chunked, reference, atol=1e-6, rtol=1e-5)
        windows = [np.arange(16), np.arange(16, 32)]
        averaged, members, _ = predict_ensemble(model, cfg, torch.device("cpu"), context,
                                               labels, queries, windows, 0, 4, "test")
        np.testing.assert_allclose(averaged, members.mean(axis=0), atol=1e-7)
        for row, indices in zip(members, windows):
            classifier.fit(context[indices], labels[indices])
            np.testing.assert_allclose(row, classifier.predict_proba(queries)[:, 0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
