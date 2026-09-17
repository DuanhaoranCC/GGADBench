"""TA-GGAD KDE underflow regression and ordinary-formula equivalence."""

import unittest
import warnings

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.special import logsumexp
from scipy.stats import gaussian_kde

from ggad.vendor.taggad.fusion import (
    average_train_kde,
    compute_js_between,
    compute_kde_distribution,
    testing_time_adaptive_fusion,
)


class TAGGADFusionTests(unittest.TestCase):
    def test_normal_case_matches_released_arithmetic(self):
        rng = np.random.RandomState(0)
        train = [compute_kde_distribution(rng.normal(size=100)) for _ in range(3)]
        target = rng.normal(size=100)
        grid = np.linspace(
            min(target.min(), *(x.min() for x, _ in train)),
            max(target.max(), *(x.max() for x, _ in train)),
            500,
        )
        density = gaussian_kde(target)(grid)
        density /= density.sum()
        expected = float(jensenshannon(average_train_kde(train, grid), density))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            self.assertEqual(compute_js_between(train, target), expected)

    def test_underflow_matches_gaussian_mixture_reference(self):
        rng = np.random.RandomState(0)
        source = rng.normal(0, 1e6, 100)
        target = rng.normal(0, 0.01, 100)
        train = [compute_kde_distribution(source)]
        grid = np.linspace(source.min(), source.max(), 500)
        kde = gaussian_kde(target)
        self.assertEqual(kde(grid).sum(), 0.0)
        log_weights = logsumexp(
            -0.5 * (grid[:, None] - target[None, :]) ** 2 / kde.covariance[0, 0],
            axis=1,
        )
        density = np.exp(log_weights - log_weights.max())
        density /= density.sum()
        expected = jensenshannon(average_train_kde(train, grid), density)
        with self.assertWarnsRegex(RuntimeWarning, "log-domain"):
            actual = compute_js_between(train, target)
        self.assertTrue(np.isfinite(actual))
        self.assertAlmostEqual(actual, expected, places=12)

    def test_full_fusion_with_underflow_remains_finite(self):
        rng = np.random.RandomState(0)
        train = [compute_kde_distribution(rng.normal(0, 1e6, 100))]
        scores = torch.from_numpy(rng.normal(0, 0.01, (40, 3)).astype(np.float32))
        labels = torch.zeros(40, dtype=torch.long)
        labels[:4] = 1
        with self.assertWarnsRegex(RuntimeWarning, "log-domain"):
            result = testing_time_adaptive_fusion(
                scores[:, 0],
                scores[:, 1],
                scores[:, 2],
                labels,
                train,
                train,
                train,
            )
        self.assertTrue(torch.isfinite(result["score_features"]).all())
        self.assertTrue(torch.isfinite(result["fused_score"]).all())
        self.assertTrue(np.isfinite(result["best_auc"]))


if __name__ == "__main__":
    unittest.main()
