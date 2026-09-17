"""Regression tests for the shared NumPy SVD cache."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from common import data as common


class SharedSVDCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.cache_patch = patch.object(common, "SVD_CACHE", Path(self.temporary_directory.name))
        self.cache_patch.start()

    def tearDown(self):
        self.cache_patch.stop()
        self.temporary_directory.cleanup()

    @staticmethod
    def _feature_matrix():
        return np.arange(60, dtype=np.float64).reshape(12, 5) / 17.0

    def _cache_path(self, features, dim):
        return common.SVD_CACHE / (f"svd{dim}_{common._feat_fingerprint(features, dim)}.npy")

    def test_corrupt_or_wrong_shape_cache_is_recomputed(self):
        features = self._feature_matrix()
        path = self._cache_path(features, 3)
        path.write_bytes(b"truncated-npy")

        repaired = common.x_svd(features, 3)
        self.assertEqual(repaired.shape, (12, 3))
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), repaired)

        np.save(path, np.zeros((12, 2), dtype=np.float64), allow_pickle=False)
        repaired_shape = common.x_svd(features, 3)
        self.assertEqual(repaired_shape.shape, (12, 3))
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), repaired_shape)

    def test_concurrent_writers_publish_one_complete_file(self):
        features = self._feature_matrix()
        real_svd = np.linalg.svd
        rendezvous = threading.Barrier(2)

        def synchronized_svd(*args, **kwargs):
            rendezvous.wait(timeout=10)
            return real_svd(*args, **kwargs)

        with patch.object(common.np.linalg, "svd", side_effect=synchronized_svd):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _index: common.x_svd(features, 3), range(2)))

        path = self._cache_path(features, 3)
        cached = np.load(path, allow_pickle=False)
        self.assertEqual(cached.shape, (12, 3))
        self.assertEqual(cached.dtype, np.float64)
        np.testing.assert_allclose(results[0], cached)
        np.testing.assert_allclose(results[1], cached)
        self.assertEqual(list(common.SVD_CACHE.glob("*.tmp.npy")), [])

    def test_unprompt_cache_uses_released_torch_float32_svd(self):
        features = self._feature_matrix()
        with TemporaryDirectory() as directory, patch.object(
            common, "SVD_TORCH_CACHE", Path(directory)
        ):
            actual = common.x_svd_torch(features, 3)
            values = torch.from_numpy(features.astype(np.float32))
            left, singular, _ = torch.linalg.svd(values, full_matrices=False)
            expected = (left[:, :3] * singular[:3]).numpy()
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.float32)
            cached = list(Path(directory).glob("*.npy"))
            self.assertEqual(len(cached), 1)
            np.testing.assert_array_equal(common.x_svd_torch(features, 3), expected)


if __name__ == "__main__":
    unittest.main()
