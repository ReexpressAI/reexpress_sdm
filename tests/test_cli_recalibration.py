# Copyright Reexpress AI, Inc. All rights reserved.
"""CLI recalibration exports fixed models and rejects incomplete workflows."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reexpress_sdm import load_artifact, write_artifact
from reexpress_sdm.cli import build_parser, main
from helpers import make_artifact


class CLIRecalibrationTests(unittest.TestCase):
    def run_cli(self, arguments, expected=0):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            status = main(arguments)
        self.assertEqual(status, expected, error.getvalue())
        return output.getvalue(), error.getvalue()

    def source(self, root):
        path = root / "source.sdmkitmodel"
        write_artifact(path, replace(make_artifact(), calibration_rows=(
            {"id": "c0", "label": 0, "prediction": 0, "sdm": [.99, .01], "qPrime": 2},
            {"id": "c1", "label": 1, "prediction": 1, "sdm": [.01, .99], "qPrime": 2},
        )))
        return path

    def test_recalibration_saves_loadable_model_without_training_inputs_or_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            original = {path.name: path.read_bytes() for path in source.iterdir()}
            destination, report = root / "updated.sdmkitmodel", root / "report.json"
            with patch("reexpress_sdm.cli.train_iterations", side_effect=AssertionError("training invoked")), \
                 patch("reexpress_sdm.cli.Dataset.load", side_effect=AssertionError("dataset loaded")), \
                 patch("reexpress_sdm.cli.SDMModel", side_effect=AssertionError("runtime constructed")):
                output, _ = self.run_cli([
                    "recalibrate", "--model", str(source),
                    "--alpha_resolution", ".05", "--output", str(destination),
                    "--report_output", str(report),
                ])
            self.assertEqual(output, "")
            result = load_artifact(destination)
            summary = json.loads(report.read_text())
            self.assertEqual(summary["sourceModelID"], "fixture-model")
            self.assertEqual(summary["modelID"], result.model_id)
            self.assertNotEqual(result.model_id, summary["sourceModelID"])
            self.assertEqual(summary["alphaResolution"], .05)
            self.assertEqual(summary["calibrationCount"], 2)
            self.assertEqual(summary["calibrationSource"], "savedCalibrationDiagnostics")
            self.assertEqual([region.alpha for region in result.regions], [.95])
            self.assertEqual({path.name: path.read_bytes() for path in source.iterdir()}, original)
            for name in ("weights.f32", "support.f32", "support.jsonl", "calibration.jsonl"):
                self.assertEqual((destination / name).read_bytes(), original[name])

    def test_recalibration_requires_source_and_ordinary_training_still_requires_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "absent.sdmkitmodel"
            for arguments, message in (
                (["recalibrate"], "--model"),
                (["train"], "--training, --calibration, --number_of_classes"),
                (["recalibrate", "--model", "unused", "--training", "unused"], "unrecognized arguments: --training"),
                (["recalibrate", "--initial_model", "unused"], "--model"),
            ):
                with self.subTest(arguments=arguments):
                    error = io.StringIO()
                    with redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                        main([*arguments, "--output", str(destination)])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(message, error.getvalue())
                    self.assertFalse(destination.exists())

    def test_commands_have_distinct_options_and_train_retains_continuation(self):
        parser = build_parser()
        common = ["train", "--training", "train", "--calibration", "cal",
                  "--number_of_classes", "2", "--output", "out"]
        continued = parser.parse_args([*common, "--initial_model", "source.sdmkitmodel"])
        self.assertEqual(continued.initial_model, "source.sdmkitmodel")
        self.assertFalse(hasattr(continued, "recalibrate"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([*common, "--recalibrate"])
        for flag, value in (("--epochs", "1"), ("--device", "cpu"), ("--calibration", "cal")):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["recalibrate", "--model", "model", "--output", "out", flag, value])
        for command in ("train", "recalibrate"):
            help_output = io.StringIO()
            with redirect_stdout(help_output), self.assertRaises(SystemExit) as raised:
                parser.parse_args([command, "--help"])
            self.assertEqual(raised.exception.code, 0)
            help_text = help_output.getvalue()
            self.assertNotIn("--recalibrate", help_text)
            if command == "train":
                self.assertIn("--initial_model", help_text)
                self.assertNotIn("unless recalibrating", help_text)
            else:
                self.assertIn("--model", help_text)
                self.assertIn("--alpha_resolution", help_text)
                self.assertNotIn("--initial_model", help_text)

    def test_invalid_resolution_missing_diagnostics_and_bad_checksums_write_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            destination = root / "absent.sdmkitmodel"
            base = ["recalibrate", "--model", str(source), "--output", str(destination)]
            for resolution in ("0", "nan", ".5"):
                _, error = self.run_cli([*base, "--alpha_resolution", resolution], expected=2)
                self.assertIn("alpha_resolution", error)
                self.assertFalse(destination.exists())
            write_artifact(source, make_artifact(), overwrite=True)
            _, error = self.run_cli(base, expected=2)
            self.assertIn("calibration.jsonl", error)
            self.assertFalse(destination.exists())
            with (source / "weights.f32").open("ab") as stream:
                stream.write(b"corrupted")
            self.run_cli(base, expected=2)
            self.assertFalse(destination.exists())

    def test_recalibration_respects_overwrite_and_reports_on_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            destination = root / "output.sdmkitmodel"
            arguments = ["recalibrate", "--model", str(source),
                         "--alpha_resolution", ".05", "--output", str(destination)]
            output, _ = self.run_cli(arguments)
            first = load_artifact(destination)
            self.assertEqual(json.loads(output)["modelID"], first.model_id)
            self.run_cli(arguments, expected=2)
            self.assertEqual(load_artifact(destination).model_id, first.model_id)
            self.run_cli([*arguments, "--overwrite"])
            self.assertNotEqual(load_artifact(destination).model_id, first.model_id)


if __name__ == "__main__":
    unittest.main()
