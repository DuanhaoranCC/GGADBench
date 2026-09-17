"""Random marked-node holdout; detector scores and all remaining queries stay intact."""

import unittest
from unittest.mock import patch

import numpy as np

from ggad import shot_query as module


class ShotQueryTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0] * 20 + [1] * 20 + [-1, 3])
        self.mark = np.arange(42) < 40
        self.scores = np.linspace(-3.0, 2.0, 42)

    def test_selection_is_seeded_prefix_nested_and_class_label_independent(self):
        small, _ = module.split_support(self.labels, self.mark, 3, 9)
        large, _ = module.split_support(self.labels, self.mark, 10, 9)
        repeated, _ = module.split_support(self.labels, self.mark, 10, 9)
        expected = np.flatnonzero(self.mark)
        np.random.RandomState(9).shuffle(expected)
        np.testing.assert_array_equal(large, expected[:10])
        np.testing.assert_array_equal(large, repeated)
        np.testing.assert_array_equal(small, large[:3])
        flipped = self.labels.copy()
        flipped[self.mark] = 1 - flipped[self.mark]
        changed, _ = module.split_support(flipped, self.mark, 10, 9)
        np.testing.assert_array_equal(changed, large)

    def test_support_is_unique_and_every_other_marked_node_is_query(self):
        support, query = module.split_support(self.labels, self.mark, 10, 4)
        self.assertEqual(len(np.unique(support)), 10)
        self.assertFalse(set(support) & set(query))
        self.assertEqual(set(support) | set(query), set(np.flatnonzero(self.mark)))
        self.assertTrue(self.mark[support].all())
        self.assertTrue(self.mark[query].all())

    def test_metric_receives_unmodified_scores_for_complete_query(self):
        _, query = module.split_support(self.labels, self.mark, 10, 4)
        scores_before = self.scores.copy()
        with patch.object(
            module, "evaluate", return_value={"AUROC": 0.6, "AUPRC": 0.7}
        ) as evaluate:
            result = module.QueryHoldout(10)("toy", 4, self.labels, self.scores, self.mark)
        labels_arg, scores_arg = evaluate.call_args[0]
        np.testing.assert_array_equal(labels_arg, self.labels[query])
        np.testing.assert_array_equal(scores_arg, self.scores[query])
        np.testing.assert_array_equal(self.scores, scores_before)
        self.assertEqual(result, {"AUROC": 0.6, "AUPRC": 0.7})

    def test_ten_shot_is_valid_with_disney_class_counts(self):
        # Uniform 10-node holdout is not 10 anomalies plus 10 normals.
        labels = np.array([0] * 118 + [1] * 6)
        support, query = module.split_support(labels, np.ones(124, dtype=bool), 10, 0)
        self.assertEqual(len(support), 10)
        self.assertEqual(len(query), 114)
        self.assertEqual(set(labels[query]), {0, 1})

    def test_missing_query_class_raises_without_label_guided_resampling(self):
        labels = np.array([0, 0, 0, 1])
        mark = np.ones(4, dtype=bool)
        # Choose a reproducible seed whose unstratified prefix removes the
        # sole anomaly; this must fail, not redraw a favorable support split.
        seed = next(
            seed for seed in range(100) if np.random.RandomState(seed).permutation(4)[0] == 3
        )
        with self.assertRaisesRegex(ValueError, "query lacks both classes"):
            module.split_support(labels, mark, 1, seed)

    def test_unknown_scores_are_ignored_and_query_nonfinite_scores_raise(self):
        scores = self.scores.copy()
        scores[~self.mark] = np.nan
        result = module.QueryHoldout(3)("toy", 2, self.labels, scores, self.mark)
        self.assertTrue(np.isfinite(list(result.values())).all())
        _, query = module.split_support(self.labels, self.mark, 3, 2)
        scores[query[0]] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite evaluation scores"):
            module.QueryHoldout(3)("toy", 2, self.labels, scores, self.mark)

    def test_budget_shape_and_binary_validation(self):
        for shot in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                module.QueryHoldout(shot)
        with self.assertRaisesRegex(ValueError, "shot < marked"):
            module.split_support(self.labels, self.mark, 40, 0)
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            module.QueryHoldout(1)("toy", 0, self.labels, self.scores[:-1], self.mark)
        labels = self.labels.copy()
        labels[0] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            module.split_support(labels, self.mark, 1, 0)


if __name__ == "__main__":
    unittest.main()
