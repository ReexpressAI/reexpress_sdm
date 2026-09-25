# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from reexpress_sdm import (DatasetBundle, DatasetValidationError, SDMModel, TorchTrainer,
                          TrainingConfig, export_source_dataset, feature_digest,
                          train_iterations, write_artifact, write_dataset_bundle)
from reexpress_sdm.cli import main
from reexpress_sdm.source_io import model_file_digests


class SourceAttachmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(12)
        cls.train = rng.normal(size=(20, 3)).astype(np.float32)
        cls.calibration = rng.normal(size=(18, 3)).astype(np.float32)
        cls.train_labels = np.arange(20) % 2
        cls.calibration_labels = np.arange(18) % 2
        cls.config = TrainingConfig(2, exemplar_dimension=3, epochs=2, batch_size=8, max_neighbors=8)
        cls.result = train_iterations(
            cls.config, cls.train, cls.train_labels, cls.calibration, cls.calibration_labels,
            representation_fingerprint="source-fixture", train_ids=[f"t{i}" for i in range(20)],
            calibration_ids=[f"c{i}" for i in range(18)], class_names=["zero", "one"],
            backend_options={"device": "cpu"}, number_of_random_shuffles=2,
            shuffle_training_and_calibration=True,
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model_path = self.root / "model.sdmkitmodel"
        write_artifact(self.model_path, self.result.artifact)
        self.paths = {}
        self.rows = {}
        for name, prefix, vectors, labels in (("training", "t", self.train, self.train_labels),
                                              ("calibration", "c", self.calibration, self.calibration_labels)):
            rows = [{"id": f"{prefix}{i}", "label": int(label), "document": f"Original {prefix}{i}",
                     "embedding": vector[:2].tolist(), "attributes": vector[2:].tolist(),
                     "metadata": {"user": "retained"}}
                    for i, (vector, label) in enumerate(zip(vectors, labels))]
            self.rows[name] = rows
            path = self.root / f"{name}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.paths[name] = path

    def export(self, *, role="original-training", **kwargs):
        return export_source_dataset(self.model_path, self.root / f"{role}.sdmdataset", role=role, **kwargs)

    def test_digests_are_portable_raw_inputs_and_selected_order(self):
        self.assertEqual(feature_digest([1, -0.0, -2.5]), hashlib.sha256(struct.pack("<fff", 1, 0, -2.5)).hexdigest())
        self.assertEqual(feature_digest(np.asarray([1, 0., -2.5], dtype=">f4")), feature_digest([1, -0., -2.5]))
        for invalid in ([True, 1], [float("inf")], [], [[1., 2.]]):
            with self.subTest(invalid=invalid), self.assertRaises(DatasetValidationError):
                feature_digest(invalid)
        pool = np.concatenate((self.train, self.calibration))
        metadata = self.result.artifact.manifest["metadata"]["sourceData"]
        self.assertEqual(metadata["trainingFeatureDigests"], [feature_digest(pool[i]) for i in self.result.training_pool_indices])
        self.assertEqual(metadata["calibrationFeatureDigests"], [feature_digest(pool[i]) for i in self.result.calibration_pool_indices])

    def test_original_sources_are_independent_text_only_and_keep_both_selected_memberships(self):
        before = model_file_digests(self.model_path)
        for name in ("training", "calibration"):
            with DatasetBundle.open(self.export(role=f"original-{name}", text_only=True, **{name: self.paths[name]})) as bundle:
                self.assertFalse(bundle.matrices)
                self.assertEqual(bundle.metadata["modelSource"]["modelFiles"], before)
                self.assertEqual(list(bundle.ids), [row["id"] for row in self.rows[name]])
                self.assertEqual({row["metadata"]["sdmSource"]["selectedSplit"] for row in bundle}, {"training", "calibration"})
                for row in bundle:
                    self.assertEqual(row["metadata"]["user"], "retained")
                    self.assertIn("featureSHA256", row["metadata"]["sdmSource"])
                    self.assertIn("Original", row["document"])
        self.assertEqual(before, model_file_digests(self.model_path))

    def test_selected_export_reorders_sources_and_reports_missing_other_file(self):
        self.paths["training"].write_text("".join(json.dumps(row) + "\n" for row in reversed(self.rows["training"])))
        with self.assertRaisesRegex(DatasetValidationError, "missing .* source rows"):
            self.export(role="selected-training", training=self.paths["training"])
        self.assertFalse((self.root / "selected-training.sdmdataset").exists())
        with DatasetBundle.open(self.export(role="selected-training", **self.paths)) as bundle:
            self.assertEqual(list(bundle.ids), [row.id for row in self.result.artifact.support_records])
            for index, row in enumerate(bundle):
                self.assertEqual(row["metadata"]["sdmSource"]["selectedIndex"], index)

    def test_feature_and_label_mismatches_fail_without_publishing(self):
        for field, value, message in (("embedding", [99., 99.], "training input digest"), ("label", 1, "source label")):
            with self.subTest(field=field):
                rows = [dict(row) for row in self.rows["training"]]
                rows[0][field] = value
                self.paths["training"].write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaisesRegex(DatasetValidationError, message):
                    self.export(training=self.paths["training"])
                self.assertFalse((self.root / "original-training.sdmdataset").exists())

    def test_with_scores_validates_exclusion_for_shuffled_original_calibration(self):
        with DatasetBundle.open(self.export(role="original-calibration", calibration=self.paths["calibration"],
                                           with_scores=True, device="cpu")) as bundle:
            model = SDMModel(self.result.artifact, device="cpu")
            for row in bundle:
                source = row["metadata"]["sdmSource"]
                index = source["selectedIndex"] if source["selectedSplit"] == "training" else None
                self.assertEqual(row.get("excludedSupportIndex"), index)
                expected = model.score([np.concatenate((row["embedding"], row["attributes"]))],
                                       identity_support_indices=[index])[0]
                self.assertEqual(row["q"], expected.q)
                self.assertEqual(row["nearestSupportID"], expected.nearest_support_id)
                self.assertEqual(row["scoreSchemaVersion"], 1)

    def test_text_only_removes_incoming_scores_and_requires_no_feature_inputs(self):
        rows = [{key: value for key, value in row.items() if key not in ("embedding", "attributes")}
                for row in self.rows["training"]]
        for row in rows:
            row.update(scoreSchemaVersion=1, modelID="unrelated", q=99, excludedSupportIndex=999)
        self.paths["training"].write_text("".join(json.dumps(row) + "\n" for row in rows))
        with DatasetBundle.open(self.export(training=self.paths["training"], text_only=True)) as bundle:
            self.assertTrue(all("scoreSchemaVersion" not in row and "excludedSupportIndex" not in row for row in bundle))
        with self.assertRaisesRegex(DatasetValidationError, "cannot be combined"):
            self.export(training=self.paths["training"], text_only=True, with_scores=True)

    def test_direct_fit_exports_selected_roles_without_invented_pool_indexes(self):
        direct = TorchTrainer(self.config, device="cpu").fit(
            self.train, self.train_labels, self.calibration, self.calibration_labels,
            representation_fingerprint="source-fixture", train_ids=[f"t{i}" for i in range(20)],
            calibration_ids=[f"c{i}" for i in range(18)], class_names=["zero", "one"])
        write_artifact(self.model_path, direct.artifact, overwrite=True)
        with DatasetBundle.open(self.export(role="selected-training", training=self.paths["training"])) as bundle:
            self.assertTrue(all("poolIndex" not in row["metadata"]["sdmSource"] for row in bundle))
        with self.assertRaisesRegex(DatasetValidationError, "bestIterationSplits"):
            self.export(training=self.paths["training"])

    def test_legacy_models_verify_projected_features_and_accept_text_only(self):
        metadata = dict(self.result.artifact.manifest["metadata"])
        metadata.pop("sourceData")
        legacy = replace(self.result.artifact, manifest={**self.result.artifact.manifest, "metadata": metadata})
        write_artifact(self.model_path, legacy, overwrite=True)
        with DatasetBundle.open(self.export(role="selected-calibration", device="cpu", **self.paths)) as bundle:
            self.assertEqual(bundle.row_count, len(legacy.calibration_rows))
        rows = self.rows["training"]
        rows[0]["embedding"] = [100., 100.]
        self.paths["training"].write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(DatasetValidationError, "projected source values"):
            self.export(training=self.paths["training"], device="cpu")
        self.export(training=self.paths["training"], text_only=True)

    def test_bundle_names_fingerprints_and_cli(self):
        bad_path = write_dataset_bundle(self.root / "bad.sdmdataset", self.rows["training"],
                                        class_names=["one", "zero"])
        with self.assertRaisesRegex(DatasetValidationError, "classNames"):
            self.export(training=bad_path)
        output = self.root / "cli.sdmdataset"
        errors = io.StringIO()
        with redirect_stderr(errors):
            status = main(["dataset", "export-sources", "--model", str(self.model_path),
                           "--training", str(self.paths["training"]), "--role", "original-training",
                           "--output", str(output), "--text_only"])
        self.assertEqual(status, 0, errors.getvalue())
        with DatasetBundle.open(output) as bundle:
            self.assertEqual(bundle.fingerprint, "source-fixture")
            self.assertEqual(bundle.class_names, ("zero", "one"))

    def test_corrupt_membership_and_wrong_ids_cannot_fall_back_to_id_matching(self):
        metadata = json.loads(json.dumps(self.result.artifact.manifest["metadata"]))
        metadata["bestIterationSplits"]["trainingPoolIndices"][0] = metadata["bestIterationSplits"]["calibrationPoolIndices"][0]
        malformed = replace(self.result.artifact, manifest={**self.result.artifact.manifest, "metadata": metadata})
        write_artifact(self.model_path, malformed, overwrite=True)
        with self.assertRaisesRegex(DatasetValidationError, "partition"):
            self.export(training=self.paths["training"], text_only=True)

    def test_unknown_index_convention_and_huge_counts_fail_before_pool_allocation(self):
        for key, value, message in (("indexConvention", "one-based", "indexConvention"),
                                    ("originalTrainingCount", 10**18, "saved source record count")):
            with self.subTest(key=key):
                metadata = json.loads(json.dumps(self.result.artifact.manifest["metadata"]))
                metadata["bestIterationSplits"][key] = value
                malformed = replace(self.result.artifact, manifest={**self.result.artifact.manifest, "metadata": metadata})
                write_artifact(self.model_path, malformed, overwrite=True)
                with self.assertRaisesRegex(DatasetValidationError, message):
                    self.export(training=self.paths["training"], text_only=True)


if __name__ == "__main__":
    unittest.main()
