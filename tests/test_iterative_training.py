# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reexpress_sdm import TorchTrainer, TrainingConfig, TrainingResult, load_artifact, train_iterations, write_artifact
from reexpress_sdm.cli import build_parser, main

from helpers import make_artifact


class IterativeTrainingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        self.train = rng.normal(size=(24, 3)).astype(np.float32)
        self.calibration = rng.normal(size=(17, 3)).astype(np.float32)
        self.train_labels = np.arange(24) % 2
        self.calibration_labels = np.arange(17) % 2
        self.config = TrainingConfig(2, exemplar_dimension=3, epochs=2, batch_size=8, max_neighbors=4)

    def fit(self, **kwargs):
        return train_iterations(
            self.config, self.train, self.train_labels, self.calibration, self.calibration_labels,
            representation_fingerprint="iteration-fixture", **kwargs,
        )

    def test_unshuffled_single_fit_uses_torch_and_keeps_memberships(self):
        expected = TorchTrainer(self.config).fit(
            self.train, self.train_labels, self.calibration, self.calibration_labels,
            representation_fingerprint="iteration-fixture",
        )
        result = self.fit(shuffle_training_and_calibration=False)
        self.assertEqual(result.best_epoch, expected.best_epoch)
        self.assertEqual(result.best_balanced_calibration_loss, expected.best_balanced_calibration_loss)
        np.testing.assert_array_equal(result.artifact.weights.projection_weight, expected.artifact.weights.projection_weight)
        self.assertEqual(result.artifact.calibration_rows, expected.artifact.calibration_rows)
        self.assertEqual(result.training_pool_indices, tuple(range(24)))
        self.assertEqual(result.calibration_pool_indices, tuple(range(24, 41)))

    def test_default_shuffles_are_reproducible_half_splits_with_exact_provenance(self):
        updates = []
        result = self.fit(number_of_random_shuffles=3, progress=updates.append)
        repeated = self.fit(number_of_random_shuffles=3)
        self.assertEqual(result.best_iteration, repeated.best_iteration)
        self.assertEqual(result.training_pool_indices, repeated.training_pool_indices)
        self.assertEqual(result.calibration_pool_indices, repeated.calibration_pool_indices)
        self.assertEqual(len(result.training_pool_indices), 20)
        self.assertEqual(len(result.calibration_pool_indices), 21)
        self.assertEqual(set(result.training_pool_indices) | set(result.calibration_pool_indices), set(range(41)))
        self.assertTrue(set(result.training_pool_indices).isdisjoint(result.calibration_pool_indices))
        self.assertEqual(len(result.history), 6)
        self.assertEqual([row["iteration"] for row in updates], [1, 1, 2, 2, 3, 3])
        metadata = result.artifact.manifest["metadata"]
        self.assertTrue(metadata["shuffledTrainingAndCalibration"])
        self.assertTrue(metadata["trainingRun"]["configuration"]["shuffleTrainingAndCalibration"])
        self.assertEqual(metadata["iterationSeeds"], [0, 1, 2])
        membership = metadata["bestIterationSplits"]
        self.assertEqual(membership["trainingIDs"], [row.id for row in result.artifact.support_records])
        self.assertEqual(membership["calibrationIDs"], [row["id"] for row in result.artifact.calibration_rows])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "iteration.sdmkitmodel"
            write_artifact(path, result.artifact)
            self.assertEqual(load_artifact(path).manifest["metadata"], metadata)

    def test_last_tie_wins_and_every_iteration_uses_a_fresh_backend(self):
        configurations = []

        class FakeTrainer:
            def fit(self, *args, **kwargs):
                rows = tuple({
                    "epoch": epoch, "trainingLoss": 0.2, "isBest": epoch == 1,
                    "balancedTrainingSDMLoss": 0.2, "balancedCalibrationSDMLoss": 0.125 if epoch == 1 else 0.25,
                    "balancedTrainingAccuracy": 0.5, "balancedCalibrationAccuracy": 0.5,
                    "balancedMeanTrainingQ": 1.0, "balancedMeanCalibrationQ": 1.0,
                    "durationSeconds": 1.0,
                } for epoch in (1, 2))
                return TrainingResult(make_artifact(), 1, 0.125, rows, duration_seconds=3.0)

        def make_backend(name, configuration, **kwargs):
            configurations.append(configuration)
            return FakeTrainer()

        with patch("reexpress_sdm.iterative_training.create_training_backend", side_effect=make_backend), \
                patch("reexpress_sdm.iterative_training.perf_counter", side_effect=[5., 15.]):
            result = self.fit(number_of_random_shuffles=3, shuffle_training_and_calibration=False)
        self.assertEqual([config.seed for config in configurations], [0, 1, 2])
        self.assertEqual(result.best_iteration, 3)
        self.assertEqual(result.training_pool_indices, tuple(range(24)))
        self.assertEqual(result.duration_seconds, 10.0)
        metadata = result.artifact.manifest["metadata"]
        self.assertEqual(metadata["trainingRun"]["durationSeconds"], 10.0,
                         "Aggregate timing spans the full attempt, not just the selected iteration")
        self.assertEqual([item["durationSeconds"] for item in metadata["iterationSummaries"]], [3.] * 3)
        self.assertEqual([row["durationSeconds"] for row in metadata["trainingRun"]["history"]], [1.] * 6)

    def test_invalid_counts_and_ids_fail_before_training(self):
        for count in [0, -1, True, 1.5]:
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "positive integer"):
                self.fit(number_of_random_shuffles=count)
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            self.fit(train_ids=[str(i) for i in range(24)], calibration_ids=[str(i) for i in range(17)])

    def test_shuffle_missing_a_class_fails_instead_of_retrying_or_stratifying(self):
        train = np.asarray([[1., 2.], [3., 4.]], dtype=np.float32)
        # Seed 5 produces two label-pure halves from [0, 1, 0, 1].
        seed = next(seed for seed in range(100) if len(set((np.arange(4) % 2)[np.random.default_rng(seed).permutation(4)[:2]])) == 1)
        with self.assertRaisesRegex(ValueError, "omitted a class"):
            train_iterations(
                replace(self.config, seed=seed), train, [0, 1], train, [0, 1],
                representation_fingerprint="fixture", shuffle_training_and_calibration=True,
            )

    def test_cli_class_count_and_shuffle_options(self):
        parser = build_parser()
        common = ["train", "--output", "out", "--representation_fingerprint", "fixture"]
        number_of_classes_args = parser.parse_args(common + [
            "--training", "train", "--calibration", "cal", "--number_of_classes", "2",
            "--exemplar_dimension", "16", "--epochs", "5", "--batch_size", "8",
            "--learning_rate", ".001", "--seed", "6", "--device", "mps",
            "--max_neighbors", "64", "--alpha_resolution", ".05", "--number_of_random_shuffles", "3",
            "--shuffle_training_and_calibration",
        ])
        self.assertEqual(number_of_classes_args.number_of_classes, 2)
        self.assertEqual(number_of_classes_args.number_of_random_shuffles, 3)
        inputs = common + ["--training", "train", "--calibration", "cal", "--number_of_classes", "2"]
        default = parser.parse_args(inputs)
        self.assertTrue(default.shuffle_training_and_calibration)
        self.assertEqual(default.number_of_random_shuffles, 1)
        self.assertFalse(parser.parse_args(inputs + ["--do_not_shuffle_data"]).shuffle_training_and_calibration)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(inputs + ["--do_not_shuffle_data", "--shuffle_training_and_calibration"])

    def test_cli_j_runs_export_selected_membership_and_all_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, vectors, labels in [("train", self.train, self.train_labels), ("cal", self.calibration, self.calibration_labels)]:
                (root / f"{name}.jsonl").write_text("".join(
                    json.dumps({"id": f"{name}-{index}", "label": int(label), "embedding": vector.tolist()}) + "\n"
                    for index, (vector, label) in enumerate(zip(vectors, labels))
                ))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main([
                    "train", "--training", str(root / "train.jsonl"),
                    "--calibration", str(root / "cal.jsonl"),
                    "--output", str(root / "model.sdmkitmodel"),
                    "--report_output", str(root / "report.json"),
                    "--number_of_classes", "2", "--exemplar_dimension", "3", "--epochs", "1",
                    "--max_neighbors", "4", "--representation_fingerprint", "fixture",
                    "--number_of_random_shuffles", "2",
                ])
            self.assertEqual(status, 0)
            report = json.loads((root / "report.json").read_text())
            artifact = load_artifact(root / "model.sdmkitmodel")
            self.assertEqual(report["trainingIterations"], 2)
            self.assertTrue(report["shuffledTrainingAndCalibration"])
            self.assertEqual(len(report["bestIterationSplits"]["trainingPoolIndices"]), 20)
            self.assertEqual(len(report["bestIterationSplits"]["calibrationPoolIndices"]), 21)
            self.assertEqual(len(report["history"]), 2)
            self.assertEqual(report["bestIterationSplits"], artifact.manifest["metadata"]["bestIterationSplits"])
            self.assertGreaterEqual(report["durationSeconds"], 0)
            self.assertEqual(report["durationSeconds"], artifact.manifest["metadata"]["trainingRun"]["durationSeconds"])
            self.assertEqual([row["durationSeconds"] for row in report["history"]],
                             [row["durationSeconds"] for row in artifact.manifest["metadata"]["trainingRun"]["history"]])


if __name__ == "__main__":
    unittest.main()
