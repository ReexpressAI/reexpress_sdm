# Copyright Reexpress AI, Inc. All rights reserved.
"""Bundle persistence, mapped access, format rejection, and publication invariants."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reexpress_sdm import (Dataset, DatasetBundle, iter_dataset_rows,
                           write_dataset_bundle, write_dataset_jsonl)
from reexpress_sdm.errors import (DatasetValidationError, DimensionMismatchError,
                                 RepresentationMismatchError)


FIXTURES = Path(__file__).resolve().parent / "fixtures/contracts-v1/datasets"


class DatasetBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, name="dense"):
        return shutil.copytree(FIXTURES / f"{name}.sdmdataset", self.root / f"{name}.sdmdataset")

    @staticmethod
    def update(path, change, *, rows=None, matrix=None):
        manifest = json.loads((path / "manifest.json").read_text())
        if rows is not None:
            payload = b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows)
            (path / "rows.jsonl").write_bytes(payload)
            manifest["rows"]["sha256"] = hashlib.sha256(payload).hexdigest()
        if matrix is not None:
            (path / "embeddings.npy").write_bytes(matrix)
            manifest["matrices"]["embedding"]["sha256"] = hashlib.sha256(matrix).hexdigest()
        change(manifest)
        (path / "manifest.json").write_text(json.dumps(manifest))

    def test_golden_dense_is_readonly_and_shared_with_feature_dataset(self):
        dataset = Dataset.load(FIXTURES / "dense.sdmdataset", expected_dimension=4,
                               number_of_classes=2, require_labels=True)
        self.addCleanup(dataset.rows.close)
        self.assertIsInstance(dataset.vectors, np.memmap)
        self.assertFalse(dataset.vectors.flags.writeable)
        self.assertTrue(np.shares_memory(dataset.vectors, dataset.rows[2]["embedding"]))
        self.assertEqual(dataset.class_names, ("Negative", "Positive"))
        self.assertEqual(dataset.representation_fingerprint, "fixture:dense-v1")
        np.testing.assert_array_equal(dataset.vectors[2], [2, -2, .5, 1])
        np.testing.assert_array_equal(np.load(FIXTURES / "dense.sdmdataset/embeddings.npy", allow_pickle=False), dataset.vectors)

    def test_partial_roundtrip_keeps_missing_features_provenance_and_source(self):
        with DatasetBundle.open(FIXTURES / "partial.sdmdataset") as original:
            output = write_dataset_bundle(self.root / "roundtrip.sdmdataset", original,
                                          class_names=original.class_names,
                                          representation=original.representation, metadata=original.metadata)
            with DatasetBundle.open(output) as restored:
                # Hydration propagates the manifest fingerprint to each row;
                # metadata bytes/checksums may therefore change on export.
                self.assertEqual({k: v for k, v in restored.manifest.items() if k != "rows"},
                                 {k: v for k, v in original.manifest.items() if k != "rows"})
                self.assertEqual(restored.ids, original.ids)
                for expected, actual in zip(original, restored):
                    for key in ("embedding", "attributes"):
                        self.assertEqual(key in expected, key in actual)
                        if key in expected:
                            np.testing.assert_array_equal(expected.pop(key), actual.pop(key))
                    self.assertEqual(expected, actual)
                self.assertEqual(restored[1]["document"], "")
                self.assertNotIn("document", restored[3])
                self.assertNotIn("embedding", restored[1])
                self.assertNotIn("embeddingRow", restored[0])

    def test_metadata_only_arbitrary_classes_and_input_fields(self):
        names = [f"Class {i}" for i in range(12)]
        rows = [{"id": str(i), "label": i, "metadata": {"tokenIDs": [i, 10], "empty": None}} for i in range(12)]
        path = write_dataset_bundle(self.root / "raw.sdmdataset", rows, class_names=names)
        with DatasetBundle.open(path) as bundle:
            self.assertEqual(bundle.matrices, {})
            self.assertEqual(list(bundle), rows)
        with self.assertRaisesRegex(DatasetValidationError, "complete cached features"):
            Dataset.load(path)

    def test_document_and_prompt_remain_independent_through_jsonl_and_bundle(self):
        rows = [
            {"id": "absent", "label": -1, "prompt": "Prompt only"},
            {"id": "null", "label": -1, "document": None, "prompt": "Null document prompt"},
            {"id": "empty", "label": -1, "document": "", "prompt": "Empty document prompt"},
            {"id": "both", "label": -1, "document": "Display text", "prompt": "Independent prompt"},
        ]
        source = write_dataset_jsonl(self.root / "source.jsonl", rows)
        bundle_path = write_dataset_bundle(self.root / "source.sdmdataset", iter_dataset_rows(source))
        with DatasetBundle.open(bundle_path) as bundle:
            self.assertEqual(list(bundle), rows)
            restored = write_dataset_jsonl(self.root / "restored.jsonl", bundle)
        self.assertEqual(list(iter_dataset_rows(restored)), rows)

    def test_explicit_composition_allows_complete_column_with_other_partial(self):
        rows = [{"id": "a", "label": 0, "embedding": [1, 2], "attributes": [3]},
                {"id": "b", "label": 1, "embedding": [4, 5]}]
        path = write_dataset_bundle(self.root / "mixed.sdmdataset", rows)
        with self.assertRaisesRegex(DatasetValidationError, "complete cached features"):
            Dataset.load(path)
        dataset = Dataset.load(path, composition="embedding", expected_dimension=2)
        self.addCleanup(dataset.rows.close)
        self.assertIsInstance(dataset.vectors, np.memmap)
        np.testing.assert_array_equal(dataset.vectors, [[1, 2], [4, 5]])

    def test_combined_columns_match_jsonl_and_dimension_guards(self):
        rows = [{"id": str(i), "label": i % 2, "embedding": [i, -i], "attributes": [.25]} for i in range(4)]
        path = write_dataset_bundle(self.root / "complete.sdmdataset", rows)
        text = write_dataset_jsonl(self.root / "complete.jsonl", rows)
        dataset = Dataset.load(path, composition="embedding+attributes")
        self.addCleanup(dataset.rows.close)
        np.testing.assert_array_equal(dataset.vectors, Dataset.load(text).vectors)
        self.assertEqual(dataset.vectors.shape, (4, 3))
        with self.assertRaises(DimensionMismatchError):
            Dataset.load(path, expected_dimension=9)
        with self.assertRaises(DatasetValidationError):
            Dataset.load(path, composition="unknown")

    def test_identity_label_and_fingerprint_guards(self):
        path = self.fixture()
        with self.assertRaises(RepresentationMismatchError):
            Dataset.load(path, representation_fingerprint="different")
        with self.assertRaises(DatasetValidationError):
            Dataset.load(path, number_of_classes=3)
        with self.assertRaises(DatasetValidationError):
            Dataset.load(FIXTURES / "partial.sdmdataset", require_labels=True)
        raw = [{"id": "a", "label": -1, "representationFingerprint": "inferred"}]
        out = write_dataset_bundle(self.root / "inferred.sdmdataset", raw)
        with DatasetBundle.open(out) as bundle:
            self.assertEqual(bundle.representation, {"fingerprint": "inferred"})
        self.update(out, lambda m: m.pop("representation"))
        with DatasetBundle.open(out) as bundle:
            self.assertEqual(bundle.fingerprint, "inferred")

    def test_supported_npy_versions_and_key_order(self):
        path = self.fixture()
        payload = np.arange(24, dtype="<f4").reshape(6, 4).tobytes()
        for version in (1, 2, 3):
            with self.subTest(version=version):
                header = b'{"shape": (6,4,), "descr": "<f4", "fortran_order": False}   \n'
                prefix = b"\x93NUMPY" + bytes([version, 0]) + struct.pack("<H" if version == 1 else "<I", len(header))
                self.update(path, lambda _: None, matrix=prefix + header + payload)
                with DatasetBundle.open(path) as bundle:
                    np.testing.assert_array_equal(bundle.matrices["embedding"].reshape(-1), np.arange(24))

    def test_npy_rejects_unsupported_layouts_and_nonliteral_header(self):
        path = self.fixture()
        original = (path / "embeddings.npy").read_bytes()
        payload = original[256:]
        headers = [
            "{'descr': '<f8', 'fortran_order': False, 'shape': (6, 4)}",
            "{'descr': '>f4', 'fortran_order': False, 'shape': (6, 4)}",
            "{'descr': '|O', 'fortran_order': False, 'shape': (6, 4)}",
            "{'descr': '<f4', 'fortran_order': True, 'shape': (6, 4)}",
            "{'descr': '<f4', 'fortran_order': False, 'shape': [6, 4]}",
            "{'descr': '<f4', 'fortran_order': False, 'shape': (24,)}",
            "{'descr': '<f4', 'fortran_order': False, 'shape': (6, 4), 'extra': 1}",
            "{'descr': '<f4', 'descr': '<f4', 'shape': (6, 4)}",
            "{'descr': '<f4', 'fortran_order': False, 'shape': (True, 4)}",
            "{'descr': '<f4', 'fortran_order': False, 'shape': (6, 5)}",
            "__import__('os').getcwd()",
        ]
        for value in headers:
            with self.subTest(header=value):
                header = value.encode() + b"\n"
                matrix = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + payload
                self.update(path, lambda _: None, matrix=matrix)
                with self.assertRaises(DatasetValidationError):
                    DatasetBundle.open(path)

    def test_npy_truncated_trailing_nonfinite_and_oversized_headers(self):
        path = self.fixture()
        original = (path / "embeddings.npy").read_bytes()
        nan_payload = np.full((6, 4), np.nan, dtype="<f4").tobytes()
        cases = [original[:4], original[:9], original[:-1], original + b"\0", original[:256] + nan_payload,
                 b"\x93NUMPY\x02\x00" + struct.pack("<I", 65_537),
                 b"\x93NUMPY\x04\x00" + original[8:]]
        for i, matrix in enumerate(cases):
            with self.subTest(case=i):
                self.update(path, lambda _: None, matrix=matrix)
                with self.assertRaises(DatasetValidationError):
                    DatasetBundle.open(path)

    def test_checksums_can_only_be_skipped_explicitly(self):
        path = self.fixture()
        with (path / "rows.jsonl").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(DatasetValidationError, "checksum"):
            DatasetBundle.open(path)
        with DatasetBundle.open(path, verify_checksums=False) as bundle:
            self.assertEqual(len(bundle), 6)

    def test_manifest_validation(self):
        path = self.fixture()
        original = json.loads((path / "manifest.json").read_text())
        cases = [
            {"format": "unknown"}, {"schemaVersion": 2}, {"schemaVersion": True},
            {"rowCount": 0}, {"rowCount": 1.5}, {"rowCount": 5},
            {"matrices": {"unknown": {}}}, {"matrices": []},
            {"classNames": None}, {"classNames": ["Same", "Same"]}, {"classNames": [" Zero", "One"]},
            {"classNames": ["Only"]}, {"metadata": None}, {"representation": None},
            {"representation": {"fingerprint": ""}}, {"representation": {"fingerprint": "  "}},
            {"rows": {"path": "../rows.jsonl", "sha256": "0" * 64}},
            {"rows": {"path": "rows.jsonl", "sha256": "invalid"}},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                manifest = copy.deepcopy(original)
                manifest.update(changes)
                (path / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(DatasetValidationError):
                    DatasetBundle.open(path)

    def test_matrix_descriptors_must_match_profile(self):
        path = self.fixture()
        original = json.loads((path / "manifest.json").read_text())
        for changes in ({"dtype": "<f8"}, {"shape": [6, 5]}, {"shape": [0, 4]},
                        {"shape": [True, 4]}, {"shape": [6]}, {"path": "other.npy"}):
            with self.subTest(changes=changes):
                manifest = copy.deepcopy(original)
                manifest["matrices"]["embedding"].update(changes)
                (path / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(DatasetValidationError):
                    DatasetBundle.open(path)

    def test_metadata_refs_ids_labels_and_fingerprints_rejected(self):
        path = self.fixture()
        original = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines()]
        for changes in ({"embeddingRow": 1}, {"embeddingRow": None}, {"embeddingRow": False},
                        {"attributesRow": 0}, {"id": "dense-1"}, {"id": ""},
                        {"label": True}, {"label": 2}, {"label": -2},
                        {"embedding": [0, 1, 2, 3]}, {"metadata": []}, {"document": 5},
                        {"representationFingerprint": "conflict"}):
            with self.subTest(changes=changes):
                rows = copy.deepcopy(original)
                rows[0].update(changes)
                self.update(path, lambda _: None, rows=rows)
                with self.assertRaises((DatasetValidationError, RepresentationMismatchError)):
                    DatasetBundle.open(path)
        rows = copy.deepcopy(original)
        rows[-1].pop("embeddingRow")
        self.update(path, lambda _: None, rows=rows)
        with self.assertRaisesRegex(DatasetValidationError, "reference count"):
            DatasetBundle.open(path)

    def test_row_fingerprint_conflict_without_manifest_fingerprint(self):
        path = self.fixture()
        rows = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines()]
        rows[0]["representationFingerprint"] = "first"
        rows[1]["representationFingerprint"] = "second"
        self.update(path, lambda m: m.pop("representation"), rows=rows)
        with self.assertRaises(RepresentationMismatchError):
            DatasetBundle.open(path)

    def test_referenced_symlinks_rejected_unreferenced_finder_files_ignored(self):
        path = self.fixture()
        (path / ".DS_Store").write_bytes(b"finder metadata")
        with DatasetBundle.open(path) as bundle:
            self.assertEqual(len(bundle), 6)
        linked_root = self.root / "link.sdmdataset"
        linked_root.symlink_to(path, target_is_directory=True)
        with self.assertRaises(DatasetValidationError):
            DatasetBundle.open(linked_root)
        outside = self.root / "outside.npy"
        (path / "embeddings.npy").rename(outside)
        (path / "embeddings.npy").symlink_to(outside)
        with self.assertRaises(DatasetValidationError):
            DatasetBundle.open(path)

    def test_atomic_cancellation_preserves_existing_output(self):
        path = self.fixture()
        original = (path / "manifest.json").read_bytes()

        def interrupted():
            yield {"id": "new", "label": 0, "embedding": [4, 5]}
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            write_dataset_bundle(path, interrupted(), overwrite=True)
        self.assertEqual((path / "manifest.json").read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_failed_publish_restores_existing_destination(self):
        path = self.fixture()
        original = (path / "manifest.json").read_bytes()
        import os
        rename = os.rename

        def fail_publish(source, destination):
            if Path(destination) == path and not str(source).endswith(".backup"):
                raise OSError("simulated publication failure")
            return rename(source, destination)

        with patch("reexpress_sdm.bundle.os.rename", side_effect=fail_publish):
            with self.assertRaisesRegex(OSError, "publication failure"):
                write_dataset_bundle(path, [{"id": "x", "label": -1}], overwrite=True)
        self.assertEqual((path / "manifest.json").read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_writer_refuses_unrelated_destination_and_can_overwrite_bundle(self):
        path = self.root / "unrelated"
        path.mkdir()
        (path / "user.txt").write_text("preserve me")
        with self.assertRaises(DatasetValidationError):
            write_dataset_bundle(path, [{"id": "a", "label": 0}], overwrite=True)
        self.assertEqual((path / "user.txt").read_text(), "preserve me")
        out = self.fixture()
        write_dataset_bundle(out, [{"id": "new", "label": -1}], overwrite=True)
        with DatasetBundle.open(out) as bundle:
            self.assertEqual(bundle.ids, ("new",))

    def test_writer_rejects_invalid_features_and_does_not_publish(self):
        out = self.root / "bad.sdmdataset"
        for feature in ([], [True], [1, True], [False, 0.5], [float("inf")], [1e100], [[1, 2]], ["1", "2"]):
            with self.subTest(feature=feature):
                with self.assertRaises(DatasetValidationError):
                    write_dataset_bundle(out, [{"id": "a", "label": 0, "embedding": feature}])
                self.assertFalse(out.exists())
        with self.assertRaises(DimensionMismatchError):
            write_dataset_bundle(out, [{"id": "a", "label": 0, "embedding": [1]},
                                       {"id": "b", "label": 1, "embedding": [1, 2]}])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_bundle_reader_rejects_malformed_source_logits_with_valid_checksums(self):
        path = self.fixture()
        original = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines()]
        invalid = (None, "not logits", [], [0], [True, False], [1, False], ["1", "2"],
                   [[0, 1], [2, 3]], [float("nan"), 0], [float("inf"), 0], [1e100, 0],
                   [10 ** 1000, 0])
        for key in ("logits", "sourceLogits"):
            for values in invalid:
                with self.subTest(key=key, values=values):
                    rows = copy.deepcopy(original)
                    rows[0][key] = values
                    self.update(path, lambda _: None, rows=rows)
                    with self.assertRaises(DatasetValidationError):
                        DatasetBundle.open(path)
        rows = copy.deepcopy(original)
        rows[0].update(logits=[0, 1], sourceLogits=[0, 1])
        self.update(path, lambda _: None, rows=rows)
        with self.assertRaisesRegex(DatasetValidationError, "both logits"):
            DatasetBundle.open(path)

    def test_invalid_logits_preserve_existing_bundle_and_jsonl_outputs(self):
        original = [{"id": "original", "label": -1, "logits": [0, 1], "embedding": [1, 2]}]
        bundle_path = write_dataset_bundle(self.root / "existing.sdmdataset", original)
        jsonl_path = write_dataset_jsonl(self.root / "existing.jsonl", original)
        bundle_bytes = {item.name: item.read_bytes() for item in bundle_path.iterdir()}
        jsonl_bytes = jsonl_path.read_bytes()
        invalid = (None, "not logits", [], [0], [True, False], [1, False], ["1", "2"],
                   [[0, 1], [2, 3]], [float("nan"), 0], [float("inf"), 0], [1e100, 0],
                   [10 ** 1000, 0])
        for key in ("logits", "sourceLogits"):
            for values in invalid:
                rows = [original[0], {"id": "bad", "label": -1, key: values}]
                for writer, destination in ((write_dataset_bundle, bundle_path), (write_dataset_jsonl, jsonl_path)):
                    with self.subTest(key=key, values=values, output=destination.suffix):
                        with self.assertRaises(DatasetValidationError):
                            writer(destination, rows, overwrite=True)
                        self.assertEqual({item.name: item.read_bytes() for item in bundle_path.iterdir()}, bundle_bytes)
                        self.assertEqual(jsonl_path.read_bytes(), jsonl_bytes)
                        self.assertEqual(set(self.root.iterdir()), {bundle_path, jsonl_path})
        for writer, destination in ((write_dataset_bundle, bundle_path), (write_dataset_jsonl, jsonl_path)):
            with self.assertRaisesRegex(DatasetValidationError, "both logits"):
                writer(destination, [{"id": "bad", "label": -1, "logits": [0, 1],
                                      "sourceLogits": [0, 1]}], overwrite=True)
        self.assertEqual({item.name: item.read_bytes() for item in bundle_path.iterdir()}, bundle_bytes)
        self.assertEqual(jsonl_path.read_bytes(), jsonl_bytes)

    def test_valid_source_logits_retain_json_precision_and_legacy_field(self):
        values = [0.12345678901234568, -0.0, 2 ** 64 - 1, 10 ** 30, float(np.finfo(np.float32).max)]
        rows = [{"id": "canonical", "label": -1, "logits": values,
                 "metadata": {"precise": 0.12345678901234568, "logits": "ordinary metadata"}},
                {"id": "legacy", "label": -1, "sourceLogits": values},
                {"id": "numpy", "label": -1, "logits": np.asarray([0.25, 0.75], dtype=np.float64)},
                {"id": "tuple", "label": -1, "logits": (0.25, 0.75)}]
        path = write_dataset_bundle(self.root / "logits.sdmdataset", rows)
        with DatasetBundle.open(path) as bundle:
            exported = write_dataset_jsonl(self.root / "logits.jsonl", bundle)
            restored = list(iter_dataset_rows(exported))
            self.assertEqual(restored[0], rows[0])
            self.assertEqual(restored[1], rows[1])
            self.assertEqual(restored[2]["logits"], [0.25, 0.75])
            self.assertEqual(restored[3]["logits"], [0.25, 0.75])
            self.assertTrue(np.signbit(restored[0]["logits"][1]))

    def test_matrix_views_remain_valid_after_metadata_close(self):
        with DatasetBundle.open(FIXTURES / "dense.sdmdataset") as bundle:
            view = bundle[0]["embedding"]
            self.assertEqual(bundle[-1]["id"], "dense-5")
            self.assertEqual(len(bundle[1:3]), 2)
            with self.assertRaises(IndexError):
                bundle[6]
        np.testing.assert_array_equal(view, [0, 0, .5, 1])
        with self.assertRaisesRegex(ValueError, "closed"):
            bundle[0]

    def test_jsonl_export_is_atomic_and_preserves_scored_metadata(self):
        row = {"id": "a", "label": 0, "embedding": np.array([.25, .5], dtype=np.float32),
               "sdmModelID": "a-model", "prediction": 1, "custom": {"tokenIDs": [1, 2]}}
        bundle_path = write_dataset_bundle(self.root / "scored.sdmdataset", [row])
        path = write_dataset_jsonl(self.root / "scored.jsonl", iter_dataset_rows(bundle_path))
        actual = list(iter_dataset_rows(path))
        self.assertEqual(actual[0]["custom"], row["custom"])
        self.assertEqual(actual[0]["embedding"], [.25, .5])
        original = path.read_bytes()
        with self.assertRaises(DatasetValidationError):
            write_dataset_jsonl(path, [row, row], overwrite=True)
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
