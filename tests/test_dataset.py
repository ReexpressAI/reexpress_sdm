# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from reexpress_sdm import Dataset, DatasetValidationError


class DatasetTests(unittest.TestCase):
    def _write(self, directory: str, rows):
        path = Path(directory) / "data.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_auto_concatenates_embedding_and_attributes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                [{"id": "a", "label": 1, "embedding": [1, 2], "attributes": [3]}],
            )
            dataset = Dataset.from_jsonl(path, expected_dimension=3, require_labels=True, number_of_classes=2)
            self.assertEqual(dataset.vectors.tolist(), [[1.0, 2.0, 3.0]])

    def test_fractional_and_string_labels_are_rejected(self):
        for bad_label in (1.2, 1.0, "1", True):
            with self.subTest(label=bad_label), tempfile.TemporaryDirectory() as directory:
                path = self._write(directory, [{"id": "a", "label": bad_label, "embedding": [1]}])
                with self.assertRaises(DatasetValidationError):
                    Dataset.from_jsonl(path, number_of_classes=2)

    def test_strings_and_booleans_in_vectors_are_rejected(self):
        for vector in (["1.0"], [True]):
            with self.subTest(vector=vector), tempfile.TemporaryDirectory() as directory:
                path = self._write(directory, [{"id": "a", "label": 1, "embedding": vector}])
                with self.assertRaises(DatasetValidationError):
                    Dataset.from_jsonl(path, number_of_classes=2)

    def test_required_label_must_be_known(self):
        for bad_label in (-1, -99):
            with self.subTest(label=bad_label), tempfile.TemporaryDirectory() as directory:
                path = self._write(directory, [{"id": "a", "label": bad_label, "embedding": [1]}])
                with self.assertRaises(DatasetValidationError):
                    Dataset.from_jsonl(
                        path, number_of_classes=2, require_labels=True
                    )

    def test_only_reserved_negative_labels_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, [{"id": "a", "label": -2, "embedding": [1]}])
            with self.assertRaises(DatasetValidationError):
                Dataset.from_jsonl(path)

    def test_present_embedding_and_attributes_must_both_be_nonempty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                [{"id": "a", "label": -1, "embedding": [1], "attributes": []}],
            )
            with self.assertRaises(DatasetValidationError):
                Dataset.from_jsonl(path)

    def test_score_convenience_allows_missing_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, [{"id": "a", "embedding": [1]}])
            dataset = Dataset.from_jsonl(path)
            self.assertEqual(dataset.labels, (-1,))

    def test_jsonl_feature_loader_validates_reserved_source_logits(self):
        invalid = (None, "not logits", [], [0], [True, False], ["1", "2"],
                   [[0, 1], [2, 3]], [float("nan"), 0], [1e100, 0], [10 ** 1000, 0])
        with tempfile.TemporaryDirectory() as directory:
            for key in ("logits", "sourceLogits"):
                for values in invalid:
                    with self.subTest(key=key, values=values):
                        path = self._write(directory, [{"id": "a", "embedding": [1, 2], key: values}])
                        with self.assertRaises(DatasetValidationError):
                            Dataset.from_jsonl(path)
            path = self._write(directory, [{"id": "a", "embedding": [1, 2],
                                            "logits": [0, 1], "sourceLogits": [0, 1]}])
            with self.assertRaisesRegex(DatasetValidationError, "both logits"):
                Dataset.from_jsonl(path)
            values = [0.12345678901234568, 10 ** 30]
            path = self._write(directory, [{"id": "a", "embedding": [1, 2], "logits": values}])
            self.assertEqual(Dataset.from_jsonl(path).rows[0]["logits"], values)


if __name__ == "__main__":
    unittest.main()
