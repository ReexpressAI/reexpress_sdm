# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reexpress_sdm import (TorchTrainer, TrainingConfig, TrainingControl, TrainingCancelled,
                           train_iterations, load_artifact, write_artifact)
from reexpress_sdm.training import (_balanced_mean, _run_epochs, _EpochEvaluator, _weight_arrays)
from reexpress_sdm.torch_backend import _q_d0_many
from reexpress_sdm.index import ExactL2Index


class TrainingLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.train = np.asarray([[-2, -1], [-1.7, -.8], [-1.2, -.7], [-.8, -.6], [1, .5], [2, 1]], dtype=np.float32)
        self.labels = np.array([0, 0, 0, 0, 1, 1])
        self.cal = np.asarray([[-1.9, -.9], [-1.4, -.6], [-.9, -.7], [1.1, .8]], dtype=np.float32)
        self.cal_labels = np.array([0, 0, 0, 1])
        self.config = TrainingConfig(2, exemplar_dimension=2, epochs=3, batch_size=6,
                                     learning_rate=.01, max_neighbors=4, alpha_resolution=.1)

    def fit(self, config=None, **kwargs):
        return TorchTrainer(config or self.config, device="cpu").fit(self.train, self.labels, self.cal,
            self.cal_labels, representation_fingerprint="lifecycle", **kwargs)

    def test_fit_and_saved_history_preserve_measured_timing(self):
        progress = []
        with patch("reexpress_sdm.torch_backend.perf_counter", side_effect=[11.0, 42.0]):
            result = self.fit(progress=progress.append)
        self.assertEqual(result.duration_seconds, 31.0)
        self.assertTrue(all(row["durationSeconds"] >= 0 for row in result.history))
        self.assertEqual([row["durationSeconds"] for row in progress],
                         [row["durationSeconds"] for row in result.history])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timed.sdmkitmodel"
            write_artifact(path, result.artifact)
            portable = load_artifact(path).manifest["metadata"]["trainingRun"]
        self.assertEqual(portable["durationSeconds"], result.duration_seconds)
        self.assertEqual([row["durationSeconds"] for row in portable["history"]],
                         [row["durationSeconds"] for row in result.history])
        self.assertEqual(portable["configuration"]["epochs"], 3)

    def test_ce_only_defers_all_matching_and_updates_only_winner(self):
        matches, progress = [], []
        def counted(*args, **kwargs):
            matches.append(True)
            return _q_d0_many(*args, **kwargs)
        def report(row):
            progress.append(dict(row))
            if row["balancedCalibrationSDMLoss"] is None:
                self.assertEqual(matches, [])
        with patch("reexpress_sdm.torch_backend._q_d0_many", side_effect=counted):
            result = self.fit(replace(self.config, cross_entropy_epochs=3), progress=report)
        self.assertEqual(len(result.history), 3)
        self.assertEqual(len(matches), 3, "CE winner matches both splits; finalization matches only calibration")
        best_ce = min(result.history, key=lambda r: (r["balancedCalibrationCELoss"], -r["epoch"]))
        self.assertEqual(result.best_epoch, best_ce["epoch"])
        self.assertEqual(sum(row["balancedCalibrationSDMLoss"] is not None for row in result.history), 1)
        self.assertEqual([row["epoch"] for row in progress], [1, 2, 3, result.best_epoch])
        for row in result.history:
            self.assertIsNotNone(row["balancedTrainingCELoss"])
            if row["epoch"] != result.best_epoch:
                self.assertIsNone(row["balancedMeanTrainingQ"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ce.sdmkitmodel"
            write_artifact(path, result.artifact)
            portable = load_artifact(path).manifest["metadata"]["trainingRun"]
            self.assertEqual(len(portable["history"]), 3)
            self.assertEqual(portable["bestEpoch"], result.best_epoch)

    def test_transition_preserves_last_ce_trajectory_and_only_best_ce_is_candidate(self):
        current, measurements, inputs = [0], [], []
        class Evaluator:
            def measure(inner, parameters=None, *, ce=False, training_only=False):
                epoch = current[0] if parameters is None else parameters[0]
                measurements.append((epoch, ce, training_only))
                q = np.full(2, epoch, dtype=np.float32)
                d = q / 10
                # Epoch 2 would be the SDM winner if mistakenly considered.
                values = {"balancedTrainingAccuracy": .5, "balancedCalibrationAccuracy": .5,
                          "balancedTrainingCELoss": None, "balancedCalibrationCELoss": None,
                          "balancedTrainingSDMLoss": None, "balancedCalibrationSDMLoss": None,
                          "balancedMeanTrainingQ": None, "balancedMeanCalibrationQ": None}
                if ce:
                    values.update(balancedTrainingCELoss=epoch, balancedCalibrationCELoss=epoch)
                elif not training_only:
                    loss = {1: .3, 2: .001, 3: .4}[epoch]
                    values.update(balancedTrainingSDMLoss=loss, balancedCalibrationSDMLoss=loss,
                                  balancedMeanTrainingQ=float(epoch), balancedMeanCalibrationQ=float(epoch))
                return values, q, d
        def step(q, d):
            inputs.append((current[0], q.copy(), d.copy()))
            current[0] += 1
            return 1.0
        with patch("reexpress_sdm.training.perf_counter", side_effect=[0., 2., 10., 13., 20., 25., 30., 37.]):
            params, epoch, loss, history = _run_epochs(replace(self.config, cross_entropy_epochs=2),
                2, step, lambda: (current[0],), Evaluator(), None, None)
        self.assertEqual(params, (1,))
        self.assertEqual(epoch, 1)
        self.assertEqual(loss, .3)
        self.assertEqual(inputs[2][0], 2, "No rewind of last CE parameters")
        np.testing.assert_array_equal(inputs[2][1], [2, 2])
        np.testing.assert_array_equal(inputs[2][2], np.array([.2, .2], dtype=np.float32))
        self.assertEqual(measurements, [(1, True, False), (2, True, False), (1, False, False),
                                       (2, False, True), (3, False, False)])
        self.assertIsNone(history[1]["balancedCalibrationSDMLoss"])
        self.assertEqual([row["durationSeconds"] for row in history], [7., 3., 7.],
                         "Deferred SDM scoring adds time to the CE winner whose metrics are updated")

    def test_stop_mid_scoring_completes_epoch_and_equals_smaller_epoch_limit(self):
        control = TrainingControl()
        def match(*args, **kwargs):
            if control.completed_epoch_count == 1:
                self.assertTrue(control.request_stop())
            return _q_d0_many(*args, **kwargs)
        with patch("reexpress_sdm.torch_backend._q_d0_many", side_effect=match):
            result = self.fit(control=control)
        expected = self.fit(replace(self.config, epochs=2))
        self.assertEqual(control.completed_epoch_count, 2)
        self.assertTrue(result.stopped_early)
        without_timing = lambda history: tuple({key: value for key, value in row.items()
                                               if key != "durationSeconds"} for row in history)
        self.assertEqual(without_timing(result.history), without_timing(expected.history))
        self.assertEqual(result.best_epoch, expected.best_epoch)
        self.assertEqual(result.artifact.calibration_rows, expected.artifact.calibration_rows)
        for actual, desired in zip(_weight_arrays(result.artifact.weights), _weight_arrays(expected.artifact.weights)):
            np.testing.assert_array_equal(actual, desired)

    def test_stop_during_ce_finalizes_winner_and_skips_later_iterations(self):
        control = TrainingControl()
        def report(row):
            if row["epoch"] == 2:
                self.assertTrue(control.request_stop())
        result = train_iterations(replace(self.config, epochs=5, cross_entropy_epochs=4),
            self.train, self.labels, self.cal, self.cal_labels, representation_fingerprint="lifecycle",
            number_of_random_shuffles=3, progress=report, control=control)
        self.assertTrue(result.stopped_early)
        self.assertEqual(result.number_of_iterations, 1)
        self.assertEqual(len(result.history), 2)
        self.assertEqual(control.completed_epoch_count, 2)
        self.assertIsNotNone(result.history[result.best_epoch - 1]["balancedCalibrationSDMLoss"])
        portable = result.artifact.manifest["metadata"]["trainingRun"]
        self.assertEqual(portable["configuration"]["iterations"], 3)
        self.assertEqual(portable["iterationCount"], 1)
        self.assertTrue(portable["stoppedEarly"])

    def test_stop_before_first_completed_score_cancels(self):
        control = TrainingControl()
        self.assertFalse(control.request_stop())
        with self.assertRaises(TrainingCancelled):
            self.fit(control=control)
        self.assertEqual(control.completed_epoch_count, 0)

    def test_cancel_during_best_ce_matching_returns_no_artifact(self):
        control = TrainingControl()
        def match(*args, **kwargs):
            control.cancel()
            return _q_d0_many(*args, **kwargs)
        with patch("reexpress_sdm.torch_backend._q_d0_many", side_effect=match), self.assertRaises(TrainingCancelled):
            self.fit(replace(self.config, cross_entropy_epochs=3), control=control)
        self.assertEqual(control.completed_epoch_count, 3)

    def test_stop_after_last_complete_epoch_is_not_truncation(self):
        control = TrainingControl()
        result = self.fit(control=control, progress=lambda row: control.request_stop() if row["epoch"] == 3 else None)
        self.assertFalse(result.stopped_early)
        self.assertEqual(control.completed_epoch_count, 3)

    def test_balanced_metrics_are_class_means_including_q(self):
        self.assertEqual(_balanced_mean(np.array([1, 1, 1, 0]), np.array([0, 0, 0, 1]), 2), .5)
        self.assertEqual(_balanced_mean(np.array([6, 6, 6, 2]), np.array([0, 0, 0, 1]), 2), 4.0)
        evaluator = _EpochEvaluator(self.config, self.labels, self.cal_labels,
            lambda calibration, saved=None, **kwargs: (
                np.tile(np.array([1, 0], dtype=np.float32), (len(self.cal if calibration else self.train), 1)),
                self.cal if calibration else self.train),
            ExactL2Index, _q_d0_many, None)
        for ce in (True, False):
            values, _, _ = evaluator.measure(ce=ce)
            self.assertEqual(values["balancedTrainingAccuracy"], .5)
            self.assertEqual(values["balancedCalibrationAccuracy"], .5)

    def test_continuation_preserves_normalization_representation_and_independent_adam(self):
        source = self.fit(replace(self.config, epochs=1)).artifact
        before = _weight_arrays(source.weights)
        seen = []
        original = torch.optim.Adam.__init__
        def capture(instance, arrays, *args, **kwargs):
            seen.append(tuple(value.detach().cpu().numpy().copy() for value in arrays))
            original(instance, arrays, *args, **kwargs)
            self.assertFalse(instance.state)
        with patch.object(torch.optim.Adam, "__init__", capture):
            result = train_iterations(replace(self.config, exemplar_dimension=9, epochs=1),
                self.train + 1, self.labels, self.cal + 1, self.cal_labels,
                representation_fingerprint="lifecycle", initial_artifact=source, number_of_random_shuffles=2)
        self.assertEqual(len(seen), 2)
        for parameters in seen:
            for actual, expected in zip(parameters, before):
                np.testing.assert_array_equal(actual, expected)
        self.assertEqual(result.artifact.manifest["normalization"], source.manifest["normalization"])
        self.assertEqual(result.artifact.manifest["representation"], source.manifest["representation"])
        self.assertEqual(result.artifact.manifest["configuration"]["exemplarDimension"], 2)
        for actual, expected in zip(_weight_arrays(source.weights), before):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(result.artifact.manifest["metadata"]["trainingRun"]["sourceModelID"], source.model_id)
        for bad in ({"representation_fingerprint": "different"}, {"class_names": ["wrong", "classes"]}):
            kwargs = {"representation_fingerprint": "lifecycle", "initial_artifact": source, **bad}
            with self.assertRaisesRegex(ValueError, "continuation requires"):
                TorchTrainer(self.config).fit(self.train, self.labels, self.cal, self.cal_labels, **kwargs)

    def test_cli_continuation_defaults_to_shuffling_updated_input_datasets(self):
        from reexpress_sdm.cli import main
        source = self.fit(replace(self.config, epochs=1)).artifact
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_path = directory / "source.sdmkitmodel"
            output = directory / "continued.sdmkitmodel"
            write_artifact(source_path, source)
            # Remove one row from each original split and add two training rows.
            # The new complete inputs have 7/3 rows; default pooling must produce 5/5.
            new_train = np.concatenate((self.train[1:], [[-3., -2.], [3., 2.]]))
            new_labels = np.concatenate((self.labels[1:], [0, 1]))
            for name, vectors, labels in (("training", new_train, new_labels),
                                         ("calibration", self.cal[1:], self.cal_labels[1:])):
                (directory / f"{name}.jsonl").write_text("".join(json.dumps({
                    "id": f"{name}-{i}", "embedding": row.tolist(), "label": int(labels[i])
                }) + "\n" for i, row in enumerate(vectors)))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main(["train", "--training", str(directory / "training.jsonl"),
                    "--calibration", str(directory / "calibration.jsonl"), "--output", str(output),
                    "--initial_model", str(source_path), "--epochs", "2", "--cross_entropy_epochs", "2",
                    "--number_of_classes", "2", "--representation_fingerprint", "lifecycle"])
            self.assertEqual(code, 0)
            artifact = load_artifact(output)
            self.assertEqual(artifact.manifest["normalization"], source.manifest["normalization"])
            run = artifact.manifest["metadata"]["trainingRun"]
            self.assertEqual(run["sourceModelID"], source.model_id)
            self.assertEqual(run["configuration"]["crossEntropyEpochs"], 2)
            self.assertTrue(run["configuration"]["shuffleTrainingAndCalibration"])
            splits = artifact.manifest["metadata"]["bestIterationSplits"]
            self.assertEqual(splits["originalTrainingCount"], 7)
            self.assertEqual(splits["originalCalibrationCount"], 3)
            expected_order = np.random.default_rng(self.config.seed).permutation(10)
            self.assertEqual(splits["trainingPoolIndices"], expected_order[:5].tolist())
            self.assertEqual(splits["calibrationPoolIndices"], expected_order[5:].tolist())
            self.assertEqual(len(artifact.support_records), 5)
            self.assertEqual(len(artifact.calibration_rows), 5)
            self.assertEqual(len(run["history"]), 2)

    def test_iteration_seeds_wrap_at_uint64_boundary(self):
        with self.assertRaisesRegex(ValueError, "UInt64"):
            replace(self.config, seed=2**64)
        result = train_iterations(replace(self.config, epochs=1, seed=2**64 - 1),
            self.train, self.labels, self.cal, self.cal_labels,
            representation_fingerprint="lifecycle", number_of_random_shuffles=2)
        self.assertEqual(result.artifact.manifest["metadata"]["iterationSeeds"], [2**64 - 1, 0])
        self.assertEqual(result.artifact.manifest["metadata"]["trainingRun"]["configuration"]["seed"], str(2**64 - 1))

    def test_configuration_ce_bounds(self):
        for value in (0, 4, True, 1.5):
            with self.assertRaises(ValueError):
                replace(self.config, cross_entropy_epochs=value)


class TorchLifecycleTests(unittest.TestCase):
    """Accelerated training stays batched through CE/SDM transitions."""
    fit = TrainingLifecycleTests.fit
    def setUp(self):
        TrainingLifecycleTests.setUp(self)
        self.torch = torch

    def test_torch_ce_matching_and_saved_normalization_cpu_and_mps(self):
        from reexpress_sdm.torch_backend import TorchTrainer, _q_d0_many, _torch_forward_numpy
        source = self.fit(replace(self.config, epochs=1)).artifact
        devices = ["cpu"] + (["mps"] if self.torch.backends.mps.is_available() else [])
        for device in devices:
            matches, transfers = [], []
            def forward(*args, **kwargs):
                result = _torch_forward_numpy(*args, **kwargs)
                transfers.append(kwargs.get("include_exemplars", True))
                self.assertEqual(result[1] is None, not kwargs.get("include_exemplars", True))
                return result
            def match(*args, **kwargs):
                matches.append(True)
                return _q_d0_many(*args, **kwargs)
            def progress(row):
                if row["balancedCalibrationSDMLoss"] is None:
                    self.assertEqual(matches, [])
            with self.subTest(device=device), patch("reexpress_sdm.torch_backend._q_d0_many", side_effect=match), patch("reexpress_sdm.torch_backend._torch_forward_numpy", side_effect=forward):
                result = TorchTrainer(replace(self.config, cross_entropy_epochs=3), device=device).fit(
                    self.train + 1, self.labels, self.cal + 1, self.cal_labels,
                    representation_fingerprint="lifecycle", initial_artifact=source, progress=progress)
            self.assertEqual(len(matches), 3)
            self.assertEqual(transfers, [False] * 6 + [True] * 4, "CE forwards transfer only logits, not full exemplar matrices")
            self.assertEqual(result.artifact.manifest["normalization"], source.manifest["normalization"])
            self.assertEqual(result.best_balanced_calibration_loss, result.history[result.best_epoch - 1]["balancedCalibrationSDMLoss"])
            self.assertIn("canonicalFinalBalancedCalibrationSDMLoss", result.artifact.manifest["metadata"])

    def test_torch_mixed_transition_stop_equals_shorter_training_cpu_and_mps(self):
        from reexpress_sdm.torch_backend import TorchTrainer
        devices = ["cpu"] + (["mps"] if self.torch.backends.mps.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                control = TrainingControl()
                config = replace(self.config, epochs=5, cross_entropy_epochs=2)
                result = TorchTrainer(config, device=device).fit(self.train, self.labels, self.cal,
                    self.cal_labels, representation_fingerprint="lifecycle", control=control,
                    progress=lambda row: control.request_stop() if row["epoch"] == 3 else None)
                reference = TorchTrainer(replace(config, epochs=3), device=device).fit(self.train, self.labels,
                    self.cal, self.cal_labels, representation_fingerprint="lifecycle")
                self.assertEqual(len(result.history), 3)
                self.assertEqual(control.completed_epoch_count, 3)
                self.assertTrue(result.stopped_early)
                self.assertEqual(result.best_epoch, reference.best_epoch)
                self.assertAlmostEqual(result.best_balanced_calibration_loss, reference.best_balanced_calibration_loss, places=6)
                for actual, expected in zip(_weight_arrays(result.artifact.weights), _weight_arrays(reference.artifact.weights)):
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_native_selection_loss_is_not_reranked_by_canonical_finalization(self):
        from reexpress_sdm.torch_backend import TorchTrainer
        source = self.fit(replace(self.config, epochs=1)).artifact
        matches = [0]
        def distorted(index, exemplars, *args):
            matches[0] += 1
            if matches[0] <= 4:  # Alter epoch measurements, then finalize the real winner.
                return np.zeros(len(exemplars), dtype=np.float32), np.zeros(len(exemplars), dtype=np.float32)
            return _q_d0_many(index, exemplars, *args)
        with patch("reexpress_sdm.torch_backend._q_d0_many", side_effect=distorted):
            result = TorchTrainer(replace(self.config, epochs=2), device="cpu").fit(
                self.train, self.labels, self.cal, self.cal_labels,
                representation_fingerprint="lifecycle", initial_artifact=source)
        selected = result.history[result.best_epoch - 1]["balancedCalibrationSDMLoss"]
        self.assertEqual(result.best_balanced_calibration_loss, selected)
        self.assertEqual(result.artifact.manifest["metadata"]["trainingRun"]["bestBalancedCalibrationLoss"], selected)
        self.assertNotAlmostEqual(result.artifact.manifest["metadata"]["canonicalFinalBalancedCalibrationSDMLoss"], selected, places=5)

    def test_continuation_first_epoch_matches_direct_pytorch_fresh_adam(self):
        from reexpress_sdm.torch_backend import TorchTrainer
        torch = self.torch
        source = self.fit(replace(self.config, epochs=1)).artifact
        vectors = self.train + 3
        norm = source.manifest["normalization"]
        parameters = [torch.nn.Parameter(torch.tensor(value)) for value in _weight_arrays(source.weights)]
        x = torch.tensor((vectors - np.float32(norm["mean"])) / np.float32(norm["standardDeviation"]))
        y = torch.tensor(self.labels)
        optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate)
        logits = (x @ parameters[0].T + parameters[1]) @ parameters[2].T + parameters[3]
        torch.nn.functional.cross_entropy(logits, y).backward()
        optimizer.step()
        expected = [parameter.detach().numpy() for parameter in parameters]
        for trainer in (TorchTrainer(replace(self.config, epochs=1), device="cpu"),):
            with self.subTest(trainer=type(trainer).__name__):
                result = trainer.fit(vectors, self.labels, self.cal + 3, self.cal_labels,
                    representation_fingerprint="lifecycle", initial_artifact=source)
                for actual, desired in zip(_weight_arrays(result.artifact.weights), expected):
                    np.testing.assert_allclose(actual, desired, rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
