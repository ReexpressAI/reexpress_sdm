# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reexpress_sdm import TrainingConfig, TorchTrainer, create_dense_index, create_training_backend, write_artifact
from reexpress_sdm.cli import build_parser, main

from helpers import make_artifact


class CLITests(unittest.TestCase):
    def test_public_defaults_use_torch_and_retired_backends_are_removed(self):
        import reexpress_sdm
        self.assertFalse(hasattr(reexpress_sdm, "NumPyTrainer"))
        parser = build_parser()
        training = ["train", "--training", "train.jsonl", "--calibration", "cal.jsonl",
                    "--output", "model.sdmkitmodel", "--number_of_classes", "2",
                    "--representation_fingerprint", "fixture"]
        scoring = ["score", "--model", "model.sdmkitmodel", "--input", "eval.jsonl"]
        self.assertEqual(parser.parse_args(training).backend, "torch")
        self.assertEqual(parser.parse_args(scoring).matching_backend, "torch")
        self.assertEqual(parser.parse_args(scoring).matching_device, "auto")
        self.assertIsInstance(create_training_backend("torch", TrainingConfig(2), device="cpu"), TorchTrainer)
        for backend in ("numpy", "mlx"):
            for arguments in (training + ["--backend", backend], scoring + ["--matching_backend", backend]):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args(arguments)
            with self.assertRaises(ValueError):
                create_training_backend(backend, TrainingConfig(2))
            with self.assertRaises(ValueError):
                create_dense_index(backend, [[0.0]])

    def _run(self, arguments):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(arguments)
        self.assertEqual(status, 0, stderr.getvalue())
        return stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def _write_jsonl(path: Path, rows) -> None:
        path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_artifact_inspect_preserves_metadata_without_constructing_runtime(self):
        artifact = make_artifact()
        expected = {
            key: artifact.manifest[key] for key in (
                "schemaVersion", "modelID", "createdAt", "producer", "configuration",
                "normalization", "representation", "regions",
            )
        }
        expected["supportCount"] = len(artifact.support_records)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "fixture.sdmkitmodel"
            write_artifact(model, artifact)
            with patch("reexpress_sdm.runtime.create_runtime", side_effect=AssertionError("inspection allocated a runtime")) as runtime, patch(
                "reexpress_sdm.cli.SDMModel", side_effect=AssertionError("inspection constructed a scoring model")
            ) as constructor:
                for extra in ([], ["--matching_device", "cuda:999999"]):
                    output, _ = self._run(["artifact", "inspect", "--model", str(model), *extra])
                    self.assertEqual(json.loads(output), expected)
                runtime.assert_not_called()
                constructor.assert_not_called()

    def test_monitor_command_is_removed(self):
        parser = build_parser()
        self.assertNotIn("monitor", parser.format_help())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parser.parse_args(["monitor"])
        self.assertEqual(error.exception.code, 2)

    def test_artifact_data_evaluation_and_training_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "fixture.sdmkitmodel"
            write_artifact(model, make_artifact())
            rows = [
                {"id": "a", "label": 0, "embedding": [1.0, 0.0]},
                {"id": "b", "label": 1, "embedding": [0.0, 1.0]},
                {"id": "c", "label": 0, "embedding": [2.0, 0.0]},
                {"id": "d", "label": 1, "embedding": [0.0, 2.0]},
            ]
            dataset = root / "data.jsonl"
            self._write_jsonl(dataset, rows)
            calibration_dataset = root / "calibration.jsonl"
            self._write_jsonl(
                calibration_dataset,
                [
                    {**row, "id": f"cal-{row['id']}"}
                    for row in rows
                ],
            )

            output, _ = self._run(["artifact", "validate", "--model", str(model)])
            self.assertTrue(json.loads(output)["valid"])
            output, _ = self._run(["artifact", "inspect", "--model", str(model)])
            self.assertEqual(json.loads(output)["modelID"], "fixture-model")

            scores = root / "scores.jsonl"
            self._run(
                ["score", "--model", str(model), "--input", str(dataset), "--output", str(scores)]
            )
            scored_rows = [json.loads(line) for line in scores.read_text().splitlines()]
            self.assertEqual(len(scored_rows), 4)
            self.assertIn("qPrimeLower", scored_rows[0])

            report = root / "evaluation.json"
            self._run(
                [
                    "evaluate",
                    "--model",
                    str(model),
                    "--input",
                    str(dataset),
                    "--output",
                    str(report),
                ]
            )
            evaluation = json.loads(report.read_text())
            self.assertEqual(set(evaluation), {"evaluation", "distribution"})
            self.assertEqual(evaluation["evaluation"]["evaluatedRows"], 4)
            self.assertEqual(evaluation["distribution"]["count"], 4)
            self.assertIn("predictedSDM", evaluation["distribution"]["signals"])

            trained = root / "trained.sdmkitmodel"
            training_report = root / "training.json"
            self._run(
                [
                    "train",
                    "--training",
                    str(dataset),
                    "--calibration",
                    str(calibration_dataset),
                    "--output",
                    str(trained),
                    "--report_output",
                    str(training_report),
                    "--number_of_classes",
                    "2",
                    "--exemplar_dimension",
                    "2",
                    "--epochs",
                    "1",
                    "--batch_size",
                    "2",
                    "--learning_rate",
                    "0.01",
                    "--max_neighbors",
                    "2",
                    "--alpha_resolution",
                    "0.1",
                    "--representation_fingerprint",
                    "cli-training-v1",
                ]
            )
            self.assertTrue((trained / "manifest.json").is_file())
            report = json.loads(training_report.read_text())
            self.assertEqual(report["bestEpoch"], 1)
            self.assertGreater(report["durationSeconds"], 0)
            manifest = json.loads((trained / "manifest.json").read_text())
            self.assertEqual(report["durationSeconds"], manifest["metadata"]["trainingRun"]["durationSeconds"])

if __name__ == "__main__":
    unittest.main()
