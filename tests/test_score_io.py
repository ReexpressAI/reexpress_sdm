# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reexpress_sdm import DatasetValidationError, Dataset, SDMModel, score_document, score_dataset_rows, write_scored_jsonl, write_artifact
from reexpress_sdm.cli import main
from helpers import make_artifact


class ScoreInterchangeTests(unittest.TestCase):
    def test_batch_scoring_checks_declared_class_order_and_accepts_undeclared_names(self):
        model = SDMModel(make_artifact(), device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            write_scored_jsonl([{"id": "a", "label": 0, "embedding": [1, 0]}], source)
            dataset = Dataset.load(source)
            for names in (None, ("zero", "one")):
                with self.subTest(names=names):
                    self.assertEqual(len(list(score_dataset_rows(model, replace(dataset, class_names=names)))), 1)
            with patch.object(model, "score", side_effect=AssertionError("must validate classes before scoring")) as score:
                for names in (("one", "zero"), ("different", "classes"), ("zero", "one", "extra")):
                    with self.subTest(names=names), self.assertRaisesRegex(DatasetValidationError, "ordered classNames"):
                        list(score_dataset_rows(model, replace(dataset, class_names=names)))
                score.assert_not_called()

    def test_single_document_retains_features_content_and_full_score_contract(self):
        model = SDMModel(make_artifact())
        source = {"id": "a", "embedding": [1], "attributes": [0], "document": "A\nsecond line", "metadata": {"name": "kept"}}
        row = score_document(model, source)
        self.assertEqual(row["label"], -1)
        self.assertEqual(row["scoreSchemaVersion"], 1)
        self.assertEqual(row["matchingSemanticsVersion"], 2)
        self.assertEqual(row["representationFingerprint"], model.representation_fingerprint)
        for key in ("embedding", "attributes", "document", "metadata"):
            self.assertEqual(row[key], source[key])
        self.assertNotIn("datasetID", row)
        self.assertNotIn("excludedSupportIndex", row)
        self.assertEqual({key: row[key] for key in model.score([[1, 0]], ids=["a"])[0].to_dict()}, model.score([[1, 0]], ids=["a"])[0].to_dict())
        self.assertNotIn("document", score_document(model, {"id": "empty", "embedding": [1, 0], "document": None}))

    def test_explicit_identity_is_validated_and_ids_alone_never_exclude_support(self):
        model = SDMModel(make_artifact())
        source = {"id": "s0", "label": 0, "embedding": [1, 0]}
        inference = score_document(model, source)
        identified = score_document(model, source, identity_support_index=0)
        self.assertEqual(inference["nearestSupportIndex"], 0)
        self.assertEqual(identified["excludedSupportIndex"], 0)
        self.assertEqual(identified["excludedSupportID"], "s0")
        self.assertNotEqual(identified["nearestSupportIndex"], 0)
        for invalid in [-1, 4, True, 0.0]:
            with self.subTest(index=invalid), self.assertRaises(DatasetValidationError):
                score_document(model, source, identity_support_index=invalid)
        with self.assertRaises(DatasetValidationError):
            score_document(model, {**source, "embedding": [9, 9]}, identity_support_index=0)
        with self.assertRaises(DatasetValidationError):
            score_document(model, {**source, "label": 1}, identity_support_index=0)

    def test_scoring_retains_prompt_without_using_it_as_document(self):
        model = SDMModel(make_artifact(), device="cpu")
        cases = ({}, {"document": None}, {"document": ""}, {"document": "Display text"})
        for fields in cases:
            with self.subTest(fields=fields):
                source = {"id": "query", "embedding": [1, 0], "prompt": "Independent prompt", **fields}
                scored = score_document(model, source)
                self.assertEqual(scored["prompt"], source["prompt"])
                if fields.get("document") is None:
                    self.assertNotIn("document", scored)
                else:
                    self.assertEqual(scored["document"], fields["document"])

    def test_selected_composition_drops_unused_features_and_replaces_old_scores(self):
        model = SDMModel(make_artifact())
        row = score_document(model, {"id": "a", "label": 0, "document": None,
                             "embedding": [1, 0], "attributes": [999], "modelID": "old",
                             "excludedSupportIndex": 2, "excludedSupportID": "old", "datasetID": "app-local",
                             "isOOD": True, "floorQPrime": 999, "floorQPrimeLower": 999}, composition="embedding")
        self.assertNotIn("attributes", row)
        self.assertNotIn("document", row)
        self.assertNotIn("excludedSupportID", row)
        self.assertTrue({"isOOD", "floorQPrime", "floorQPrimeLower"}.isdisjoint(row))
        self.assertEqual(row["modelID"], "fixture-model")

    def test_batch_cli_full_is_uploadable_and_compact_is_report_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "fixture.sdmkitmodel"
            write_artifact(model, make_artifact())
            source = root / "source.jsonl"
            rows = [{"id": "a", "embedding": [1, 0], "document": "a"}, {"id": "b", "embedding": [0, 1], "label": 1}]
            for row in rows:
                row.update({"isOOD": True, "floorQPrime": 999, "floorQPrimeLower": 999})
            write_scored_jsonl(rows, source)
            output = root / "scores.jsonl"
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["score", "--model", str(model), "--input", str(source), "--output", str(output), "--score_batch_size", "1"]), 0)
            actual = [json.loads(line) for line in output.read_text().splitlines()]
            expected = list(score_dataset_rows(SDMModel.load(model), Dataset.from_jsonl(source), batch_size=2))
            self.assertEqual(actual, expected)
            self.assertTrue(all({"isOOD", "floorQPrime", "floorQPrimeLower"}.isdisjoint(row) for row in actual))
            self.assertEqual(actual[0]["embedding"], rows[0]["embedding"])
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["score", "--model", str(model), "--input", str(source), "--output", str(output), "--detail", "compact"]), 0)
            compact = json.loads(output.read_text().splitlines()[0])
            self.assertNotIn("scoreSchemaVersion", compact)
            self.assertTrue({"isOOD", "floorQPrime", "floorQPrimeLower"}.isdisjoint(compact))

    def test_dataset_export_retains_actual_composition_and_rejects_mismatched_override(self):
        model = SDMModel(make_artifact())
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            write_scored_jsonl([{"id": "a", "embedding": [1, 0], "attributes": [9]}], source)
            dataset = Dataset.from_jsonl(source, composition="embedding")
            row = list(score_dataset_rows(model, dataset))[0]
            self.assertNotIn("attributes", row)
            with self.assertRaises(DatasetValidationError):
                list(score_dataset_rows(model, dataset, composition="auto"))
            write_scored_jsonl([{"id": "a", "embedding": [1], "attributes": [0]}], source)
            composed = Dataset.from_jsonl(source)
            with self.assertRaises(DatasetValidationError):
                list(score_dataset_rows(model, composed, composition="embedding"))

    def test_failed_output_generation_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "scores.jsonl"
            destination.write_text("prior result\n")
            def malformed():
                yield {"id": "first"}
                yield {"id": "bad", "q": float("nan")}
            with self.assertRaises(ValueError):
                write_scored_jsonl(malformed(), destination)
            self.assertEqual(destination.read_text(), "prior result\n")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
