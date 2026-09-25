# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from reexpress_sdm import ExactL2Index, TorchTrainer, SDMModel, TrainingConfig, load_artifact, write_artifact
from reexpress_sdm.training import _validate_training_arrays
from reexpress_sdm.torch_backend import _q_d0_many


class TrainingTests(unittest.TestCase):
    def test_torch_trainer_emits_loadable_scoring_artifact(self):
        train_vectors = np.asarray(
            [[-2.0, -1.0], [-1.5, -0.5], [-1.0, -1.5], [2.0, 1.0], [1.5, 0.5], [1.0, 1.5]],
            dtype=np.float32,
        )
        train_labels = [0, 0, 0, 1, 1, 1]
        calibration_vectors = np.asarray(
            [[-1.8, -0.8], [-0.8, -1.2], [1.8, 0.8], [0.8, 1.2]], dtype=np.float32
        )
        calibration_labels = [0, 0, 1, 1]
        trainer = TorchTrainer(
            TrainingConfig(
                number_of_classes=2,
                exemplar_dimension=2,
                epochs=2,
                batch_size=2,
                learning_rate=0.01,
                max_neighbors=6,
                alpha_resolution=0.1,
            )
        )
        result = trainer.fit(
            train_vectors,
            train_labels,
            calibration_vectors,
            calibration_labels,
            representation_fingerprint="training-test-v1",
        )
        self.assertIn(result.best_epoch, (1, 2))
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.artifact.support_vectors.shape, (6, 2))
        expected_mean = float(np.mean(train_vectors, dtype=np.float32))
        expected_std = float(np.std(train_vectors, ddof=1, dtype=np.float32))
        self.assertEqual(result.artifact.manifest["normalization"]["mean"], expected_mean)
        self.assertEqual(
            result.artifact.manifest["normalization"]["standardDeviation"], expected_std
        )
        self.assertEqual(
            set(result.artifact.calibration_rows[0]),
            {"id", "label", "prediction", "sdm", "qPrime", "q", "d0", "d", "zPrime"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trained.sdmkitmodel"
            write_artifact(path, result.artifact)
            loaded = load_artifact(path)
            scores = SDMModel(loaded).score(calibration_vectors)
            self.assertEqual(len(scores), 4)
            self.assertTrue(all(score.model_id == loaded.model_id for score in scores))

    def test_training_neighbor_cap_includes_identity_slot(self):
        exemplars = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        predictions = np.zeros(4, dtype=np.int64)
        labels = np.zeros(4, dtype=np.int64)
        q, _ = _q_d0_many(
            ExactL2Index(exemplars),
            exemplars,
            predictions,
            labels,
            predictions,
            max_neighbors=3,
            identity=True,
        )
        self.assertEqual(q.tolist(), [2.0, 2.0, 2.0, 2.0])

    def test_batched_training_q_and_d0_match_row_reference(self):
        rng = np.random.default_rng(713)
        exemplars = rng.normal(size=(31, 11)).astype(np.float32)
        predictions = rng.integers(0, 3, size=31, dtype=np.int64)
        labels = rng.integers(0, 3, size=31, dtype=np.int64)
        index = ExactL2Index(exemplars, query_tile_size=4, support_tile_size=7)
        for identity in (False, True):
            requested = 9 - (1 if identity else 0)
            expected_q = []
            expected_d0 = []
            for row, exemplar in enumerate(exemplars):
                distances, neighbors = index.search_one(
                    exemplar, requested, row if identity else None
                )
                q = 0
                for neighbor in neighbors:
                    support_index = int(neighbor)
                    if (
                        labels[support_index] == predictions[support_index]
                        and predictions[support_index] == predictions[row]
                    ):
                        q += 1
                    else:
                        break
                expected_q.append(float(q))
                expected_d0.append(float(distances[0]))
            q, d0 = _q_d0_many(
                index,
                exemplars,
                predictions,
                labels,
                predictions,
                max_neighbors=9,
                identity=identity,
            )
            np.testing.assert_array_equal(q, np.asarray(expected_q, dtype=np.float32))
            # Tiled GEMM and single-row searches agree to Float32 rounding, as in
            # the research FAISS/BLAS path; the batch_invariant option is bitwise.
            np.testing.assert_allclose(
                d0, np.asarray(expected_d0, dtype=np.float32), rtol=1e-5, atol=1e-4
            )
        invariant = ExactL2Index(exemplars, query_tile_size=4, support_tile_size=7, batch_invariant=True)
        expected_d0 = [invariant.search_one(exemplar, 9)[0][0] for exemplar in exemplars]
        _, d0 = _q_d0_many(invariant, exemplars, predictions, labels, predictions, max_neighbors=9, identity=False)
        np.testing.assert_array_equal(d0, np.asarray(expected_d0, dtype=np.float32))

    def test_training_inputs_are_not_silently_coerced(self):
        good_vectors = np.asarray([[0.0], [1.0]], dtype=np.float32)
        for bad_vectors, bad_labels in (
            ([["0.0"], ["1.0"]], [0, 1]),
            (good_vectors, [0.0, 1.0]),
            (good_vectors, [False, True]),
        ):
            with self.subTest(bad_labels=bad_labels), self.assertRaises(ValueError):
                _validate_training_arrays(
                    bad_vectors, bad_labels, good_vectors, [0, 1], classes=2
                )

    def test_training_configuration_rejects_invalid_runtime_values(self):
        for kwargs in (
            {"max_neighbors": 1},
            {"seed": -1},
            {"ood_limit": -1},
            {"learning_rate": float("nan")},
            {"q_offset": float("inf")},
            {"alpha_resolution": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingConfig(number_of_classes=2, **kwargs)

    def test_training_rejects_calibration_id_overlap(self):
        trainer = TorchTrainer(
            TrainingConfig(
                number_of_classes=2,
                exemplar_dimension=1,
                epochs=1,
                max_neighbors=2,
            )
        )
        vectors = np.asarray([[0.0], [1.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            trainer.fit(
                vectors,
                [0, 1],
                vectors,
                [0, 1],
                representation_fingerprint="overlap-test",
                train_ids=["shared", "train"],
                calibration_ids=["shared", "calibration"],
            )


if __name__ == "__main__":
    unittest.main()
