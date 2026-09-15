# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

import unittest

import numpy as np

from reexpress_sdm import ExactL2Index, NestedCalibrator
from reexpress_sdm.calibration import _class_threshold
from reexpress_sdm.errors import DimensionMismatchError
from reexpress_sdm.math import ladder_alphas, region_accepts
from reexpress_sdm.types import Region


def _legacy_nested_calibration(
    probabilities: np.ndarray,
    q_prime: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    classes: int,
    resolution: float,
    ood_limit: int,
) -> tuple[Region, ...]:
    """The original quadratic candidate scan, retained only as a test oracle."""

    residual = np.ones(q_prime.size, dtype=bool)
    regions: list[Region] = []
    for alpha in ladder_alphas(resolution):
        candidate_mask = residual & (np.floor(q_prime) > ood_limit)
        candidates = sorted(set(float(value) for value in q_prime[candidate_mask]))
        if not candidates:
            break
        chosen = None
        for candidate in candidates:
            retained = candidate_mask & (q_prime >= np.float32(candidate))
            thresholds = []
            for true_class in range(classes):
                rows = np.flatnonzero(retained & (labels == true_class))
                thresholds.append(
                    _class_threshold(
                        [float(probabilities[row, true_class]) for row in rows], alpha
                    )
                )
            if all(value >= float(np.float32(alpha)) for value in thresholds):
                chosen = Region(alpha, candidate, tuple(thresholds))
                break
        if chosen is None:
            continue
        regions.append(chosen)
        for row in np.flatnonzero(residual):
            if region_accepts(
                float(q_prime[row]),
                probabilities[row],
                int(predictions[row]),
                chosen,
                ood_limit,
            ):
                residual[row] = False
    return tuple(regions)


class ExactL2TilingTests(unittest.TestCase):
    def test_tiled_partial_top_k_matches_full_lexicographic_reference(self):
        rng = np.random.default_rng(821)
        support = rng.normal(size=(97, 19)).astype(np.float32)
        queries = rng.normal(size=(23, 19)).astype(np.float32)
        # This oracle compares CPU arithmetic; accelerator rounding is covered
        # by separate exact-tie and runtime parity tests.
        index = ExactL2Index(support, device="cpu", query_tile_size=5, support_tile_size=11)
        actual_distances, actual_indices = index.search_many(queries, 13)

        centered_support = support - support[0]
        centered_queries = queries - support[0]
        support_norms = np.sum(
            centered_support * centered_support, axis=1, dtype=np.float32
        )
        for row, query in enumerate(centered_queries):
            distances = support_norms + np.sum(
                query * query, dtype=np.float32
            ) - np.float32(2.0) * (centered_support @ query)
            distances = np.maximum(distances, np.float32(0.0))
            expected = np.lexsort((np.arange(support.shape[0]), distances))[:13]
            np.testing.assert_array_equal(actual_indices[row], expected)
            np.testing.assert_allclose(
                actual_distances[row], distances[expected], rtol=2e-6, atol=2e-5
            )

    def test_exact_ties_cross_support_tiles_and_choose_lowest_indices(self):
        support = np.asarray(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]] * 3,
            dtype=np.float32,
        )
        queries = np.zeros((4, 2), dtype=np.float32)
        index = ExactL2Index(support, query_tile_size=2, support_tile_size=3)
        distances, indices = index.search_many(queries, 5)
        np.testing.assert_array_equal(indices, np.tile(np.arange(5), (4, 1)))
        np.testing.assert_array_equal(distances, np.ones((4, 5), dtype=np.float32))

    def test_batch_identity_exclusion_precedes_k_cap(self):
        support = np.zeros((9, 3), dtype=np.float32)
        index = ExactL2Index(support, query_tile_size=2, support_tile_size=2)
        distances, indices = index.search_many(support[:4], 20, [0, 3, 5, 8])
        self.assertEqual(distances.shape, (4, 8))
        for row, excluded in enumerate((0, 3, 5, 8)):
            self.assertNotIn(excluded, indices[row])
            self.assertEqual(
                indices[row].tolist(), [value for value in range(9) if value != excluded]
            )

    def test_batch_invariant_option_makes_search_one_and_batch_bitwise_identical(self):
        rng = np.random.default_rng(92)
        support = rng.normal(size=(41, 37)).astype(np.float32)
        queries = rng.normal(size=(7, 37)).astype(np.float32)
        index = ExactL2Index(
            support, query_tile_size=3, support_tile_size=7, batch_invariant=True
        )
        self.assertTrue(index.batch_invariant)
        distances, indices = index.search_many(queries, 8)
        untiled_distances, untiled_indices = ExactL2Index(
            support,
            query_tile_size=32,
            support_tile_size=support.shape[0],
            batch_invariant=True,
        ).search_many(queries, 8)
        np.testing.assert_array_equal(indices, untiled_indices)
        np.testing.assert_array_equal(distances, untiled_distances)
        for row, query in enumerate(queries):
            one_distances, one_indices = index.search_one(query, 8)
            np.testing.assert_array_equal(indices[row], one_indices)
            np.testing.assert_array_equal(distances[row], one_distances)

    def test_default_gemm_tiles_match_per_row_arithmetic_up_to_float32_rounding(self):
        # The default GEMM tiles follow the research FAISS/BLAS behavior: a
        # distance may differ from the per-row (GEMV) value in its last Float32
        # bits depending on the surrounding batch, while selection semantics and
        # neighbor order for non-tied data are unchanged.
        rng = np.random.default_rng(92)
        support = rng.normal(size=(41, 37)).astype(np.float32)
        queries = rng.normal(size=(7, 37)).astype(np.float32)
        gemm = ExactL2Index(support, query_tile_size=3, support_tile_size=7)
        gemv = ExactL2Index(support, query_tile_size=3, support_tile_size=7, batch_invariant=True)
        self.assertFalse(gemm.batch_invariant)
        distances, indices = gemm.search_many(queries, 8)
        reference_distances, reference_indices = gemv.search_many(queries, 8)
        np.testing.assert_array_equal(indices, reference_indices)
        np.testing.assert_allclose(distances, reference_distances, rtol=1e-5, atol=1e-4)
        with self.assertRaises(ValueError):
            ExactL2Index(support, batch_invariant="yes")

    def test_batch_validation_and_empty_query_contract(self):
        index = ExactL2Index(np.zeros((3, 2), dtype=np.float32))
        distances, indices = index.search_many(np.empty((0, 2), dtype=np.float32), 2)
        self.assertEqual(distances.shape, (0, 2))
        self.assertEqual(indices.shape, (0, 2))
        with self.assertRaises(DimensionMismatchError):
            index.search_many(np.zeros((2, 3), dtype=np.float32), 1)
        with self.assertRaises(DimensionMismatchError):
            index.search_many(np.zeros((2, 2), dtype=np.float32), 1, [0])
        with self.assertRaises(ValueError):
            index.search_many(np.zeros((2, 2), dtype=np.float32), 1, [[0], [1]])
        for exclusions in ([True], [0.0]):
            with self.subTest(exclusions=exclusions), self.assertRaises(ValueError):
                index.search_many(np.zeros((1, 2), dtype=np.float32), 1, exclusions)
        for bad in (True, 0, 1.5):
            with self.subTest(k=bad), self.assertRaises(ValueError):
                index.search_many(np.zeros((1, 2), dtype=np.float32), bad)
        with self.assertRaises(ValueError):
            ExactL2Index(np.zeros((2, 2), dtype=np.float32), query_tile_size=0)


class NestedCalibrationSweepTests(unittest.TestCase):
    def test_optimized_sweep_is_exactly_equivalent_to_legacy_scan(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            for classes in (2, 3, 5):
                row_count = 137
                probabilities = rng.dirichlet(
                    np.full(classes, 0.55), size=row_count
                ).astype(np.float32)
                # Quantization deliberately produces large exact-tie groups.
                q_prime = (
                    np.floor(rng.uniform(0.0, 18.0, size=row_count) * 4.0) / 4.0
                ).astype(np.float32)
                labels = rng.integers(0, classes, size=row_count, dtype=np.int64)
                labels[:classes] = np.arange(classes)
                predictions = np.argmax(probabilities, axis=1).astype(np.int64)
                predictions[::7] = rng.integers(
                    0, classes, size=predictions[::7].size, dtype=np.int64
                )
                for resolution in (0.05, 0.1, 0.2):
                    for ood_limit in (0, 1):
                        with self.subTest(
                            seed=seed,
                            classes=classes,
                            resolution=resolution,
                            ood_limit=ood_limit,
                        ):
                            expected = _legacy_nested_calibration(
                                probabilities,
                                q_prime,
                                labels,
                                predictions,
                                classes,
                                resolution,
                                ood_limit,
                            )
                            actual = NestedCalibrator(
                                classes, resolution, ood_limit
                            ).fit(probabilities, q_prime, labels, predictions)
                            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
