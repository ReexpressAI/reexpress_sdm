# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reexpress_sdm import (
    ExactL2Index,
    SDMModel,
    TrainingConfig,
    create_dense_index,
    create_training_backend,
    load_artifact,
    write_artifact,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class BackendSelectionTests(unittest.TestCase):
    def test_base_import_does_not_load_torch(self):
        environment = dict(os.environ)
        source_root = str(Path(__file__).resolve().parents[1] / "src")
        environment["PYTHONPATH"] = source_root
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, reexpress_sdm, reexpress_sdm.cli; "
                "assert 'torch' not in sys.modules",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_torch_is_the_only_backend_and_retired_backends_are_rejected(self):
        trainer = create_training_backend("torch", TrainingConfig(number_of_classes=2), device="cpu")
        self.assertEqual(type(trainer).__name__, "TorchTrainer")
        for name in ("numpy", "numpy-reference", "mlx", "mlx-python"):
            with self.assertRaises(ValueError):
                create_training_backend(name, TrainingConfig(number_of_classes=2))
            with self.assertRaises(ValueError):
                create_dense_index(name, np.asarray([[0.0]], dtype=np.float32))

    def test_factory_rejects_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, "torch"):
            create_training_backend("unknown", TrainingConfig(number_of_classes=2))

    def test_missing_optional_dependency_has_install_guidance(self):
        from reexpress_sdm.torch_backend import _load_torch

        with patch(
            "reexpress_sdm.torch_backend.importlib.import_module",
            side_effect=ImportError("missing"),
        ), self.assertRaisesRegex(ImportError, r"reexpress_sdm"):
            _load_torch()


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency is not installed")
class TorchBackendTests(unittest.TestCase):
    def test_q_reduction_is_bounded_and_preserves_global_identity_exclusions(self):
        from reexpress_sdm.torch_backend import TorchExactL2Index, _q_d0_many
        class RecordingIndex(TorchExactL2Index):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.calls = []
            def search_many(self, queries, k, exclude_indices=None):
                self.calls.append((len(queries), None if exclude_indices is None else exclude_indices.copy()))
                return super().search_many(queries, k, exclude_indices)
        support = np.column_stack((np.arange(23, dtype=np.float32), np.zeros(23, np.float32)))
        predictions = np.zeros(23, np.int64)
        labels = (np.arange(23) % 4 == 0).astype(np.int64)
        index = RecordingIndex(support, device="cpu", query_batch_size=4, support_tile_size=7)
        oracle = TorchExactL2Index(support, device="cpu", query_batch_size=4, support_tile_size=7)
        for identity in (False, True):
            index.calls.clear()
            exclusions = np.arange(23, dtype=np.int64) if identity else None
            distances, neighbors = oracle.search_many(support, 8 - int(identity), exclusions)
            good = (labels[neighbors] == predictions[neighbors]) & (predictions[neighbors] == predictions[:, None])
            expected_q = np.sum(np.cumprod(good, axis=1, dtype=np.int64), axis=1)
            q, d0 = _q_d0_many(index, support, predictions, labels, predictions, 8, identity)
            np.testing.assert_array_equal(q, expected_q)
            np.testing.assert_array_equal(d0, distances[:, 0])
            self.assertEqual([size for size, _ in index.calls], [4, 4, 4, 4, 4, 3])
            if identity:
                np.testing.assert_array_equal(np.concatenate([excluded for _, excluded in index.calls]), np.arange(23))

    def test_tiled_matching_preserves_distance_then_support_index_order(self):
        support = np.asarray(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]],
            dtype=np.float32,
        )
        queries = np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32)
        index = create_dense_index(
            "torch",
            support,
            device="cpu",
            query_batch_size=1,
            support_tile_size=2,
        )
        distances, neighbors = index.search_many(queries, 5)
        self.assertEqual(neighbors[0].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(neighbors[1].tolist(), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(
            distances,
            np.asarray(
                [[0.0, 0.0, 1.0, 1.0, 4.0], [0.25, 0.25, 0.25, 2.25, 2.25]],
                dtype=np.float32,
            ),
        )

        excluded_distances, excluded_neighbors = index.search_many(
            queries, 5, exclude_indices=np.asarray([0, 2], dtype=np.int64)
        )
        self.assertEqual(excluded_neighbors[0].tolist(), [1, 2, 3, 4])
        self.assertEqual(excluded_neighbors[1].tolist(), [0, 1, 3, 4])
        self.assertTrue(np.all(np.isfinite(excluded_distances)))

        all_ties = create_dense_index(
            "torch",
            np.zeros((100, 3), dtype=np.float32),
            device="cpu",
            query_batch_size=2,
            support_tile_size=11,
        )
        _, tied_neighbors = all_ties.search_many(
            np.zeros((3, 3), dtype=np.float32), 25
        )
        np.testing.assert_array_equal(
            tied_neighbors,
            np.tile(np.arange(25, dtype=np.int64), (3, 1)),
        )

        single = create_dense_index(
            "torch", np.asarray([[1.0]], dtype=np.float32), device="cpu"
        )
        empty_distances, empty_neighbors = single.search_one(
            np.asarray([1.0], dtype=np.float32), 1, exclude_index=0
        )
        self.assertEqual(empty_distances.shape, (0,))
        self.assertEqual(empty_neighbors.shape, (0,))

    def test_cpu_matching_agrees_with_direct_squared_distance_oracle(self):
        from reexpress_sdm.torch_backend import TorchExactL2Index

        rng = np.random.default_rng(7)
        support = rng.normal(size=(17, 5)).astype(np.float32)
        queries = rng.normal(size=(6, 5)).astype(np.float32)
        torch_index = TorchExactL2Index(
            support,
            device="cpu",
            query_batch_size=2,
            support_tile_size=4,
        )
        distances, neighbors = torch_index.search_many(queries, 9)
        for row, query in enumerate(queries):
            direct = np.sum((support - query) ** 2, axis=1, dtype=np.float32)
            expected_neighbors = np.lexsort((np.arange(len(support)), direct))[:9]
            expected_distances = direct[expected_neighbors]
            np.testing.assert_array_equal(neighbors[row], expected_neighbors)
            np.testing.assert_allclose(distances[row], expected_distances, rtol=2e-6, atol=2e-6)

    def test_trainer_returns_loadable_canonically_finalized_artifact(self):
        train_vectors = np.asarray(
            [
                [-2.0, -1.0],
                [-1.5, -0.5],
                [-1.0, -1.5],
                [2.0, 1.0],
                [1.5, 0.5],
                [1.0, 1.5],
            ],
            dtype=np.float32,
        )
        calibration_vectors = np.asarray(
            [[-1.8, -0.8], [-0.8, -1.2], [1.8, 0.8], [0.8, 1.2]],
            dtype=np.float32,
        )
        trainer = create_training_backend(
            "torch",
            TrainingConfig(
                number_of_classes=2,
                exemplar_dimension=2,
                epochs=2,
                batch_size=2,
                learning_rate=0.01,
                max_neighbors=6,
                alpha_resolution=0.1,
            ),
            device="cpu",
            matching_query_batch_size=2,
            matching_support_tile_size=3,
        )
        result = trainer.fit(
            train_vectors,
            [0, 0, 0, 1, 1, 1],
            calibration_vectors,
            [0, 0, 1, 1],
            representation_fingerprint="torch-training-test-v1",
        )
        metadata = result.artifact.manifest["metadata"]
        self.assertEqual(metadata["trainingBackend"], "pytorch")
        self.assertEqual(metadata["artifactFinalizationBackend"], "torch:cpu")
        self.assertEqual(result.artifact.support_vectors.dtype, np.dtype(np.float32))
        self.assertEqual(len(result.artifact.calibration_rows), 4)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torch-trained.sdmkitmodel"
            write_artifact(path, result.artifact)
            loaded = load_artifact(path)
            scores = SDMModel(loaded, device="cpu").score(calibration_vectors)
            self.assertEqual(len(scores), 4)

            eval_path = Path(directory) / "eval.jsonl"
            eval_path.write_text(
                "".join(
                    json.dumps({"id": f"eval-{row}", "embedding": vector.tolist()}) + "\n"
                    for row, vector in enumerate(calibration_vectors)
                ),
                encoding="utf-8",
            )
            score_path = Path(directory) / "scores.jsonl"
            from reexpress_sdm.cli import main

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(
                    [
                        "score",
                        "--model",
                        str(path),
                        "--input",
                        str(eval_path),
                        "--output",
                        str(score_path),
                        "--matching_backend",
                        "torch",
                        "--matching_device",
                        "cpu",
                        "--matching_query_batch_size",
                        "2",
                        "--matching_support_tile_size",
                        "3",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(len(score_path.read_text(encoding="utf-8").splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
