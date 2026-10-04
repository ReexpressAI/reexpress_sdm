# Copyright Reexpress AI, Inc. All rights reserved.
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reexpress_sdm import ArtifactValidationError, TorchTrainer, TrainingConfig, load_artifact, train_iterations, write_artifact
from reexpress_sdm import artifact
from reexpress_sdm.capacity import check_training_capacity, estimate_training_capacity, memory_advisory, validate_training_capacity
from reexpress_sdm.cli import main

from helpers import make_artifact


class TrainingCapacityTests(unittest.TestCase):
    def setUp(self):
        self.config = TrainingConfig(2, exemplar_dimension=3, epochs=1, max_neighbors=4)
        self.train = np.random.default_rng(9).normal(size=(12, 3)).astype(np.float32)
        self.calibration = self.train + np.float32(0.1)
        self.labels = np.arange(12) % 2

    def test_two_million_rows_fit_metadata_and_existing_tensor_budgets(self):
        result = estimate_training_capacity(
            training_count=1_000_000, calibration_count=1_000_000,
            input_dimension=16130, exemplar_dimension=1000, number_of_classes=2,
            epochs=300, iterations=2, alpha_resolution=0.01, shuffle=True,
            encoded_id_bytes=2_000_000 * 38,
        )
        self.assertEqual(artifact.MAX_MANIFEST_BYTES, 4 * 1024**3)
        self.assertEqual(result.input_vector_bytes, 129_040_000_000)
        self.assertEqual(result.support_elements, 1_000_000_000)
        self.assertEqual(result.weight_elements, 16_133_002)
        self.assertEqual(result.estimated_manifest_bytes, 396_310_480)
        validate_training_capacity(result)

    def test_low_dimensions_allow_twenty_million_rows_without_relaxing_tensor_limits(self):
        arguments = dict(training_count=10_000_000, calibration_count=10_000_000,
                         input_dimension=128, exemplar_dimension=32, number_of_classes=2,
                         epochs=300, iterations=2, alpha_resolution=0.01, shuffle=True,
                         encoded_id_bytes=20_000_000 * 38)
        capacity = estimate_training_capacity(**arguments)
        self.assertEqual(capacity.support_elements, 320_000_000)
        self.assertGreater(capacity.estimated_manifest_bytes, 512 * 1024**2)
        self.assertLess(capacity.estimated_manifest_bytes, 4 * 1024**3)
        validate_training_capacity(capacity)
        # The manifest increase does not grant larger support payloads.
        with self.assertRaisesRegex(ArtifactValidationError, "support tensor"):
            validate_training_capacity(estimate_training_capacity(
                **dict(arguments, exemplar_dimension=201)))

    def test_odd_pool_and_unshuffled_counts_use_actual_selected_support(self):
        arguments = dict(training_count=8, calibration_count=3, input_dimension=7,
                         exemplar_dimension=5, number_of_classes=2, epochs=1,
                         iterations=1, alpha_resolution=0.1, encoded_id_bytes=33)
        shuffled = estimate_training_capacity(**arguments, shuffle=True)
        original = estimate_training_capacity(**arguments, shuffle=False)
        self.assertEqual((shuffled.support_elements, original.support_elements), (25, 40))
        self.assertEqual(shuffled.estimated_manifest_bytes - original.estimated_manifest_bytes, 3 * 80)

    def test_id_escaping_and_requested_history_are_budgeted(self):
        ids = ['a', '"\\\n', '\u00e9', '\U0001f600']
        result = check_training_capacity(self.config, training_count=2, calibration_count=2,
                                         input_dimension=3, ids=ids, iterations=3)
        quoted = sum(len(json.dumps(value, ensure_ascii=True)) for value in ids)
        expected = estimate_training_capacity(
            training_count=2, calibration_count=2, input_dimension=3,
            exemplar_dimension=3, number_of_classes=2, epochs=1, iterations=3,
            alpha_resolution=0.05, shuffle=False, encoded_id_bytes=quoted,
            additional_metadata_bytes=2,
        )
        self.assertEqual(result, expected)
        self.assertGreater(quoted, sum(len(value) + 2 for value in ids))
        with patch.object(artifact, "MAX_MANIFEST_BYTES", result.estimated_manifest_bytes - 1):
            with self.assertRaisesRegex(ArtifactValidationError, "pre-flight"):
                check_training_capacity(self.config, training_count=2, calibration_count=2,
                                        input_dimension=3, ids=ids, iterations=3)

    def test_shape_failures_happen_before_reading_ids(self):
        def forbidden_ids():
            raise AssertionError("IDs were scanned before impossible shapes were rejected")
            yield "unused"
        for input_dimension, count in ((3, 1_000_000_001), (1_000_000_001, 2)):
            with self.subTest(input_dimension=input_dimension), self.assertRaisesRegex(ArtifactValidationError, "tensor|weights"):
                check_training_capacity(self.config, training_count=count, calibration_count=2,
                                        input_dimension=input_dimension, ids=forbidden_ids())

    def test_over_budget_rejected_before_pooling_and_direct_normalization(self):
        with patch.object(artifact, "MAX_MANIFEST_BYTES", 1024), \
             patch("reexpress_sdm.iterative_training.np.concatenate", side_effect=AssertionError("pooled features")):
            with self.assertRaisesRegex(ArtifactValidationError, "pre-flight"):
                train_iterations(self.config, self.train, self.labels, self.calibration, self.labels)
        trainer = TorchTrainer(self.config, device="cpu")
        with patch.object(artifact, "MAX_MANIFEST_BYTES", 1024), \
             patch("reexpress_sdm.torch_backend.np.mean", side_effect=AssertionError("normalized features")):
            with self.assertRaisesRegex(ArtifactValidationError, "pre-flight"):
                trainer.fit(self.train, self.labels, self.calibration, self.labels)

    def test_single_id_scan_covers_multiple_iterations_and_bounds_actual_output(self):
        from reexpress_sdm.capacity import check_training_capacity as original
        ids = tuple(f'row-{index}-\U0001f600-"\\\n' for index in range(24))
        with patch("reexpress_sdm.iterative_training.check_training_capacity", wraps=original) as checks, \
             patch("reexpress_sdm.capacity.check_training_capacity", side_effect=AssertionError("repeated scan")):
            result = train_iterations(self.config, self.train, self.labels, self.calibration, self.labels,
                                      train_ids=ids[:12], calibration_ids=ids[12:],
                                      number_of_random_shuffles=2, shuffle_training_and_calibration=False,
                                      backend_options={"device": "cpu"})
        checks.assert_called_once()
        capacity = original(self.config, training_count=12, calibration_count=12,
                            input_dimension=3, ids=ids, iterations=2)
        with tempfile.TemporaryDirectory() as directory:
            model = write_artifact(Path(directory) / "model", result.artifact)
            self.assertLess((model / "manifest.json").stat().st_size, capacity.estimated_manifest_bytes)
            self.assertEqual(load_artifact(model).model_id, result.artifact.model_id)

    def test_continuation_checks_saved_exemplar_dimension(self):
        initial = make_artifact()
        config = TrainingConfig(2, exemplar_dimension=2_000_000_000, epochs=1, max_neighbors=4)
        with patch("reexpress_sdm.iterative_training.create_training_backend", side_effect=RuntimeError("past pre-flight")):
            with self.assertRaisesRegex(RuntimeError, "past pre-flight"):
                train_iterations(config, self.train[:, :2], self.labels,
                                 self.calibration[:, :2], self.labels,
                                 initial_artifact=initial, shuffle_training_and_calibration=False)

    def test_cli_advisory_is_separate_from_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, offset in (("train", 0), ("calibration", 12)):
                rows = [{"id": f"r{index + offset}", "label": int(self.labels[index]),
                         "embedding": vector.tolist()} for index, vector in enumerate(self.train)]
                (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr), \
                 patch("reexpress_sdm.capacity.physical_memory_bytes", return_value=128 * 1024**3):
                status = main(["train", "--training", str(root / "train.jsonl"),
                               "--calibration", str(root / "calibration.jsonl"),
                               "--output", str(root / "model"), "--number_of_classes", "2",
                               "--exemplar_dimension", "3", "--epochs", "1", "--max_neighbors", "4",
                               "--device", "cpu", "--do_not_shuffle_data"])
            self.assertEqual(status, 0, stderr.getvalue())
            self.assertIn("modelID", json.loads(stdout.getvalue()))
            advisory = json.loads(stderr.getvalue().splitlines()[0])
            self.assertEqual(advisory["type"], "trainingMemoryAdvisory")
            self.assertEqual(advisory["inputVectorBytes"], 24 * 3 * 4)
        self.assertNotIn("warning", memory_advisory(8, None))
        self.assertIn("warning", memory_advisory(64, 128))


class ManifestCapacityTests(unittest.TestCase):
    def test_oversized_export_rejects_before_payload_writes_and_preserves_destination(self):
        source = make_artifact()
        manifest = dict(source.manifest, metadata={"note": "x" * 10000})
        with tempfile.TemporaryDirectory() as directory:
            target = write_artifact(Path(directory) / "model", source)
            original = (target / "manifest.json").read_bytes()
            with patch.object(artifact, "MAX_MANIFEST_BYTES", len(original) + 100), \
                 patch("reexpress_sdm.artifact._write_f32", side_effect=AssertionError("payload write")):
                with self.assertRaisesRegex(ArtifactValidationError, "manifest exceeds"):
                    write_artifact(target, replace(source, manifest=manifest), overwrite=True)
            self.assertEqual((target / "manifest.json").read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_loader_rejects_sparse_file_over_4_gib_before_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            with path.open("wb") as stream:
                stream.truncate(artifact.MAX_MANIFEST_BYTES + 1)
            with patch("reexpress_sdm.artifact._decode_model_json", side_effect=AssertionError("decoded oversized input")):
                with self.assertRaisesRegex(ArtifactValidationError, "loader safety limit"):
                    artifact._load_json(path)

    def test_valid_manifest_above_old_64_mib_limit_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            target = write_artifact(Path(directory) / "model", make_artifact())
            path = target / "manifest.json"
            with path.open("ab") as stream:
                padding = b" " * 1024**2
                for _ in range(64):
                    stream.write(padding)
            self.assertGreater(path.stat().st_size, 64 * 1024**2)
            self.assertEqual(load_artifact(target).model_id, "fixture-model")

    def test_cli_large_model_advisory_precedes_decoding_and_keeps_stdout_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            target = write_artifact(Path(directory) / "model", make_artifact())
            stdout, stderr = io.StringIO(), io.StringIO()
            decode = artifact._decode_model_json
            def check_decode(source):
                self.assertEqual(json.loads(stderr.getvalue())["type"], "modelMemoryAdvisory")
                return decode(source)
            with redirect_stdout(stdout), redirect_stderr(stderr), \
                 patch("reexpress_sdm.cli._LARGE_MODEL_ADVISORY_BYTES", 1), \
                 patch("reexpress_sdm.artifact._decode_model_json", side_effect=check_decode):
                status = main(["artifact", "inspect", "--model", str(target)])
            self.assertEqual(status, 0, stderr.getvalue())
            self.assertIn("modelID", json.loads(stdout.getvalue()))
            events = stderr.getvalue().splitlines()
            self.assertEqual(len(events), 1)
            self.assertEqual(json.loads(events[0])["manifestBytes"],
                             (target / "manifest.json").stat().st_size)
            # Ordinary models and API loads stay quiet; exports do not prompt.
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                self.assertEqual(main(["artifact", "inspect", "--model", str(target)]), 0)
                load_artifact(target)
                write_artifact(Path(directory) / "another", make_artifact())
            self.assertEqual(stderr.getvalue(), "")

    def test_cli_advisory_covers_continuation_and_source_export_entrypoints(self):
        with tempfile.TemporaryDirectory() as directory:
            target = write_artifact(Path(directory) / "model", make_artifact())
            cases = [
                ("_run_train", ["train", "--initial_model", str(target), "--training", "train.jsonl",
                                "--calibration", "calibration.jsonl", "--number_of_classes", "2",
                                "--output", "unused"]),
                ("_run_dataset_export_sources", ["dataset", "export-sources", "--model", str(target),
                                                "--training", "train.jsonl", "--calibration", "calibration.jsonl",
                                                "--role", "selected-training", "--output", "unused.sdmdataset"]),
            ]
            for name, arguments in cases:
                stderr = io.StringIO()
                with self.subTest(command=name), redirect_stderr(stderr), \
                     patch("reexpress_sdm.cli._LARGE_MODEL_ADVISORY_BYTES", 1), \
                     patch("reexpress_sdm.cli." + name, return_value=0) as operation:
                    self.assertEqual(main(arguments), 0)
                    operation.assert_called_once()
                self.assertEqual(len(stderr.getvalue().splitlines()), 1)
                self.assertEqual(json.loads(stderr.getvalue())["type"], "modelMemoryAdvisory")

    def test_exact_writer_budget_includes_checksum_structure(self):
        source = make_artifact()
        with tempfile.TemporaryDirectory() as directory:
            first = write_artifact(Path(directory) / "first", source)
            byte_count = (first / "manifest.json").stat().st_size
            with patch.object(artifact, "MAX_MANIFEST_BYTES", byte_count):
                second = write_artifact(Path(directory) / "second", source)
                self.assertEqual((second / "manifest.json").stat().st_size, byte_count)
            with patch.object(artifact, "MAX_MANIFEST_BYTES", byte_count - 1):
                with self.assertRaisesRegex(ArtifactValidationError, "manifest exceeds"):
                    write_artifact(Path(directory) / "third", source)
            self.assertFalse((Path(directory) / "third").exists())


if __name__ == "__main__":
    unittest.main()
