# Copyright Reexpress AI, Inc. All rights reserved.
"""Dataset commands work with raw-input bundles and portable scored bundles."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reexpress_sdm import DatasetBundle, SDMModel, load_artifact, write_artifact, write_dataset_bundle
from reexpress_sdm.cli import main
from reexpress_sdm.score_io import score_document
from helpers import make_artifact


class DatasetCLITests(unittest.TestCase):
    def run_cli(self, *arguments, expected=0):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = main(list(arguments))
        self.assertEqual(status, expected, errors.getvalue())
        return output.getvalue(), errors.getvalue()

    def test_validate_and_convert_reject_invalid_source_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, bundle = root / "bad.jsonl", root / "bad.sdmdataset"
            write_dataset_bundle(bundle, [{"id": "a", "label": -1}])
            invalid = ({"logits": "not logits"}, {"logits": [True, False]}, {"logits": []},
                       {"logits": [1]}, {"logits": [1e100, 0]}, {"sourceLogits": [False, 1]},
                       {"logits": [0, 1], "sourceLogits": [0, 1]})
            for fields in invalid:
                with self.subTest(fields=fields):
                    payload = json.dumps({"id": "a", "label": -1, **fields}) + "\n"
                    source.write_text(payload)
                    (bundle / "rows.jsonl").write_text(payload)
                    manifest = json.loads((bundle / "manifest.json").read_text())
                    manifest["rows"]["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
                    (bundle / "manifest.json").write_text(json.dumps(manifest))
                    for input_path in (source, bundle):
                        output, error = self.run_cli("dataset", "validate", "--input", str(input_path), expected=2)
                        self.assertEqual(output, "")
                        self.assertIn("logits", error.lower())
                        for suffix in (".jsonl", ".sdmdataset"):
                            destination = root / f"output{suffix}"
                            self.run_cli("dataset", "convert", "--input", str(input_path),
                                         "--output", str(destination), expected=2)
                            self.assertFalse(destination.exists())

    def test_metadata_only_conversion_inspection_and_jsonl_roundtrip(self):
        rows = [{"id": "raw-1", "label": 0, "document": "", "user_question": "Why?", "ai_response": "Answer."},
                {"id": "raw-2", "label": 1, "document": "Long Ω document\n" * 100, "metadata": {"source": "fixture"}}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, bundle, restored = root / "raw.jsonl", root / "raw.sdmdataset", root / "restored.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.run_cli("dataset", "convert", "--input", str(source), "--output", str(bundle), "--class_names", "zero,one")
            for command in ("inspect", "validate"):
                output, _ = self.run_cli("dataset", command, "--input", str(bundle))
                information = json.loads(output)
                self.assertEqual(information["rowCount"], 2)
                self.assertEqual(information["classNames"], ["zero", "one"])
                self.assertEqual(information["featureDimensions"], {})
                if command == "validate": self.assertTrue(information["valid"])
            self.run_cli("dataset", "convert", "--input", str(bundle), "--output", str(restored))
            self.assertEqual([json.loads(line) for line in restored.read_text().splitlines()], rows)
            self.run_cli("dataset", "convert", "--input", str(source), "--output", str(bundle), expected=2)
            self.assertEqual(DatasetBundle.open(bundle).row_count, 2)
            self.run_cli("dataset", "convert", "--input", str(bundle), "--output", str(bundle), "--overwrite", expected=2)

    def test_full_scoring_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, bundle = root / "model.sdmkitmodel", root / "evaluation.sdmdataset"
            write_artifact(model, make_artifact())
            rows = [{"id": "e0", "label": 0, "embedding": [1.5, 0.0], "document": "first"},
                    {"id": "e1", "label": 1, "embedding": [0.0, 1.5], "document": "second"}]
            write_dataset_bundle(bundle, rows, class_names=["zero", "one"], representation={"fingerprint": "fixture-v1"})
            scored_bundle, scored_jsonl = root / "scores.sdmdataset", root / "scores.jsonl"
            base = ("score", "--model", str(model), "--input", str(bundle), "--matching_device", "cpu")
            self.run_cli(*base, "--output", str(scored_bundle))
            self.run_cli(*base, "--output", str(scored_jsonl))
            binary_rows = list(DatasetBundle.open(scored_bundle).iter_rows())
            json_rows = [json.loads(line) for line in scored_jsonl.read_text().splitlines()]
            for binary, text in zip(binary_rows, json_rows):
                np.testing.assert_array_equal(binary.pop("embedding"), text.pop("embedding"))
                self.assertEqual(binary, text)
            output, _ = self.run_cli("evaluate", "--model", str(model), "--input", str(bundle), "--matching_device", "cpu")
            report = json.loads(output)
            self.assertEqual(set(report), {"evaluation", "distribution"})
            self.assertEqual(report["evaluation"]["evaluatedRows"], 2)
            self.assertEqual(report["distribution"]["count"], 2)
            self.assertIn("qPrime", report["distribution"]["signals"])
            loaded = SDMModel.load(model, device="cpu")
            direct = score_document(loaded, next(DatasetBundle.open(bundle).iter_rows()))
            self.assertEqual(direct["prediction"], binary_rows[0]["prediction"])
            self.assertEqual(json.loads(json.dumps(direct))["prediction"], direct["prediction"])

    def test_bundle_training_and_clear_missing_features_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, calibration = root / "train.sdmdataset", root / "cal.sdmdataset"
            values = [[2., 0.], [1., 0.], [0., 1.], [0., 2.]]
            for path, prefix in ((training, "t"), (calibration, "c")):
                write_dataset_bundle(path, [{"id": f"{prefix}{i}", "label": int(i >= 2), "embedding": vector}
                                            for i, vector in enumerate(values)],
                                     representation={"fingerprint": "training-fixture"}, class_names=["zero", "one"])
            output = root / "trained.sdmkitmodel"
            self.run_cli("train", "--training", str(training), "--calibration", str(calibration), "--output", str(output),
                         "--number_of_classes", "2", "--representation_fingerprint", "training-fixture",
                         "--epochs", "1", "--exemplar_dimension", "2", "--batch_size", "4", "--max_neighbors", "4", "--device", "cpu")
            self.assertEqual(load_artifact(output).configuration["classNames"], ["zero", "one"])
            _, error = self.run_cli("train", "--training", str(training), "--calibration", str(calibration),
                                   "--output", str(root / "bad.sdmkitmodel"), "--number_of_classes", "2",
                                   "--class_names", "reversed,labels", "--representation_fingerprint", "training-fixture", expected=2)
            self.assertIn("class names", error)
            raw = root / "raw.sdmdataset"
            write_dataset_bundle(raw, [{"id": "r", "label": 0, "document": "Raw input only"}])
            _, error = self.run_cli("score", "--model", str(output), "--input", str(raw), "--matching_device", "cpu", expected=2)
            self.assertRegex(error.lower(), "embedding|feature")

    def test_compact_binary_output_is_rejected_before_model_loading(self):
        with patch("reexpress_sdm.cli._controller", side_effect=AssertionError("loaded a model")) as controller:
            _, error = self.run_cli("score", "--model", "missing.sdmkitmodel", "--input", "missing.jsonl", "--output", "compact.sdmdataset", "--detail", "compact", expected=2)
            self.assertIn("--detail full", error)
            controller.assert_not_called()

    def test_model_commands_reject_reordered_classes_before_scoring_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, source = root / "model.sdmkitmodel", root / "reversed.sdmdataset"
            write_artifact(model, make_artifact())
            rows = [{"id": "e0", "label": 0, "embedding": [1.5, 0.0]},
                    {"id": "e1", "label": 1, "embedding": [0.0, 1.5]}]
            write_dataset_bundle(source, rows, class_names=["one", "zero"], representation={"fingerprint": "fixture-v1"})
            output = root / "existing.jsonl"
            original = b"preserve the previous result\n"
            output.write_bytes(original)
            common = ("--model", str(model), "--matching_device", "cpu")
            cases = [
                ("score", *common, "--input", str(source), "--output", str(root / "new.sdmdataset")),
                ("score", *common, "--input", str(source), "--output", str(output)),
                ("score", *common, "--input", str(source), "--output", str(output), "--detail", "compact"),
                ("evaluate", *common, "--input", str(source), "--output", str(output)),
            ]
            with patch.object(SDMModel, "score", side_effect=AssertionError("must validate classes before scoring")) as score:
                for arguments in cases:
                    with self.subTest(arguments=arguments):
                        _, error = self.run_cli(*arguments, expected=2)
                        self.assertIn("ordered classNames", error)
                        self.assertIn("same class names in the same order", error)
                        self.assertEqual(output.read_bytes(), original)
                        self.assertFalse((root / "new.sdmdataset").exists())
                score.assert_not_called()

    def test_conversion_normalizes_missing_jsonl_label_to_unlabeled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "unlabeled.jsonl", root / "unlabeled.sdmdataset"
            source.write_text(json.dumps({"id": "unlabeled", "document": "Raw input"}) + "\n")
            self.run_cli("dataset", "convert", "--input", str(source), "--output", str(target))
            self.assertEqual(next(DatasetBundle.open(target).iter_rows())["label"], -1)

    def test_conversion_preserves_bundle_provenance_and_rejects_relabeling_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.sdmdataset", root / "copy.sdmdataset"
            write_dataset_bundle(source, [{"id": "a", "label": 0, "embedding": [0., 1.]}], class_names=["zero", "one"],
                                 representation={"fingerprint": "source", "provider": "fixture"}, metadata={"note": "retained"})
            self.run_cli("dataset", "convert", "--input", str(source), "--output", str(target))
            before, after = DatasetBundle.open(source), DatasetBundle.open(target)
            self.assertEqual(before.class_names, after.class_names)
            self.assertEqual(before.representation, after.representation)
            self.assertEqual(before.metadata, after.metadata)
            self.run_cli("dataset", "convert", "--input", str(source), "--output", str(root / "bad.sdmdataset"),
                         "--representation_fingerprint", "different", expected=2)
