"""Regression tests for the classic-baseline feature cache."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from baselines import preprocess


class BaselinePreprocessCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.cache_patch = patch.object(preprocess, "CACHE", Path(self.temporary_directory.name))
        self.cache_patch.start()
        self.features = np.arange(60, dtype=np.float64).reshape(12, 5) / 13.0
        u, singular, _ = np.linalg.svd(self.features, full_matrices=False)
        self.projected = u[:, :3] @ np.diag(singular[:3])

    def tearDown(self):
        self.cache_patch.stop()
        self.temporary_directory.cleanup()

    def _path(self):
        return preprocess.CACHE / "toy_12_svd3.npy"

    def test_corrupt_shape_or_dtype_cache_is_recomputed(self):
        invalid_entries = (
            b"truncated-npy",
            np.zeros((12, 2), dtype=np.float64),
            np.zeros((12, 3), dtype=np.float32),
        )
        for invalid in invalid_entries:
            with self.subTest(kind=type(invalid).__name__):
                path = self._path()
                if isinstance(invalid, bytes):
                    path.write_bytes(invalid)
                else:
                    np.save(path, invalid, allow_pickle=False)
                with patch.object(preprocess, "x_svd", return_value=self.projected) as compute:
                    repaired = preprocess._svd_cached(self.features, "toy", 3)
                compute.assert_called_once()
                self.assertEqual(repaired.dtype, np.float32)
                np.testing.assert_array_equal(repaired, self.projected.astype(np.float32))
                np.testing.assert_array_equal(np.load(path, allow_pickle=False), self.projected)

    def test_concurrent_writers_publish_one_complete_file(self):
        rendezvous = threading.Barrier(2)

        def synchronized_projection(_features, _dim):
            rendezvous.wait(timeout=10)
            return self.projected.copy()

        with patch.object(preprocess, "x_svd", side_effect=synchronized_projection):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(
                    pool.map(
                        lambda _index: preprocess._svd_cached(self.features, "toy", 3),
                        range(2),
                    )
                )

        cached = np.load(self._path(), allow_pickle=False)
        self.assertEqual(cached.shape, (12, 3))
        self.assertEqual(cached.dtype, np.float64)
        np.testing.assert_array_equal(cached, self.projected)
        for result in results:
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_array_equal(result, cached.astype(np.float32))
        self.assertEqual(list(preprocess.CACHE.glob("*.tmp.npy")), [])


if __name__ == "__main__":
    unittest.main()
