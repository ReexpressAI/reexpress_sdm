# Copyright Reexpress AI, Inc. All rights reserved.
"""Release boundary regressions shared with the native implementation."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from reexpress_sdm import (ArtifactValidationError, Dataset, DatasetBundle, DatasetValidationError,
    RepresentationMismatchError, SDMModel, load_artifact, score_document, validate_manifest,
    write_artifact, write_dataset_bundle)
from reexpress_sdm.training import TrainingConfig, _balanced_loss, _EpochEvaluator, _run_epochs
from helpers import make_artifact

class ReleaseRegressionTests(unittest.TestCase):
    def test_model_json_rejects_ambiguous_keys_at_every_read_boundary(self):
        row = {"id": "cal", "label": 0, "prediction": 0, "sdm": [.8, .2], "qPrime": 1}
        base = replace(make_artifact(), calibration_rows=(row,))
        for filename, field, token in (("manifest.json", "schemaVersion", "1"),
                ("support.jsonl", "label", "1"), ("calibration.jsonl", "qPrime", "2")):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.sdmkitmodel"
                write_artifact(path, base)
                source = path / filename
                source.write_text(source.read_text().replace("{", '{"' + field + '":' + token + ',', 1))
                with self.assertRaisesRegex(ArtifactValidationError, "duplicate"):
                    load_artifact(path, verify_checksums=False)

    def test_model_metadata_rejects_lossy_keys_and_integer_tokens(self):
        for metadata in ({"é": 1, "e\u0301": 2}, {"n": 1 << 64}, {"n": -(1 << 63) - 1}):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as directory:
                base = make_artifact()
                manifest = copy.deepcopy(base.manifest)
                manifest["metadata"] = metadata
                with self.assertRaises(ArtifactValidationError):
                    SDMModel(replace(base, manifest=manifest))
                path = Path(directory) / "model.sdmkitmodel"
                write_artifact(path, base)
                stored = json.loads((path / "manifest.json").read_text())
                stored["metadata"] = metadata
                (path / "manifest.json").write_text(json.dumps(stored))
                with self.assertRaises(ArtifactValidationError):
                    load_artifact(path)

    def test_model_metadata_integer_endpoints_roundtrip_exactly(self):
        base = make_artifact()
        manifest = copy.deepcopy(base.manifest)
        manifest["metadata"] = {"signed": -(1 << 63), "unsigned": (1 << 64) - 1, "decimal": 1e20}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.sdmkitmodel"
            write_artifact(path, replace(base, manifest=manifest))
            self.assertEqual(load_artifact(path).manifest["metadata"], manifest["metadata"])

    def test_cached_counts_cannot_exceed_matching_population(self):
        base = make_artifact()
        row = {"id": "cal", "label": 0, "prediction": 0, "sdm": [.8, .2], "qPrime": 1, "q": 2}
        for change in ({"qPrime": 1e20}, {"q": 5}, {"qPrime": 3}, {"qPrime": 5, "q": 5}):
            with self.subTest(change=change), self.assertRaises(ArtifactValidationError):
                SDMModel(replace(base, calibration_rows=({**row, **change},)))
        for field in ("rescaledSimilarityCDFs", "regions"):
            manifest = copy.deepcopy(base.manifest)
            if field == "regions":
                manifest[field][0]["minimumRescaledSimilarity"] = 5
            else:
                manifest[field][0].append(5)
            with self.subTest(field=field), self.assertRaises(ArtifactValidationError):
                validate_manifest(manifest)

    def test_manifest_semantic_bounds_apply_after_float32_conversion(self):
        for section, field, value in (("configuration", "qOffset", 1.00000000001),
                                     ("normalization", "standardDeviation", 1e-50)):
            manifest = copy.deepcopy(make_artifact().manifest)
            manifest[section][field] = value
            with self.subTest(field=field), self.assertRaises(ArtifactValidationError):
                validate_manifest(manifest)
        manifest = copy.deepcopy(make_artifact().manifest)
        manifest["schemaVersion"] = True
        with self.assertRaises(ArtifactValidationError):
            validate_manifest(manifest)

    def test_schema_integers_use_signed_range_while_metadata_preserves_unsigned(self):
        for key in ("maxNeighbors", "oodLimit"):
            manifest = copy.deepcopy(make_artifact().manifest)
            manifest["configuration"][key] = (1 << 63) - 1
            validate_manifest(manifest)
            manifest["configuration"][key] = 1 << 63
            with self.subTest(key=key), self.assertRaises(ArtifactValidationError):
                validate_manifest(manifest)

    def test_balanced_loss_retains_log_sum_at_large_common_logit_offset(self):
        labels = np.array([0, 0, 1])
        q = np.zeros(3, dtype=np.float32)
        d = np.ones(3, dtype=np.float32)
        for offset in (0, 1e8, -1e8):
            with self.subTest(offset=offset):
                loss = _balanced_loss(np.full((3, 2), offset, dtype=np.float32), labels, q, d, 2, 2)
                self.assertAlmostEqual(loss, 1.0, places=6)

    def test_checkpoint_selection_does_not_reward_large_uniform_logits(self):
        config = TrainingConfig(2, exemplar_dimension=2, epochs=2, max_neighbors=2)
        labels = np.array([0, 1])
        for offset in (0, 1e8):
            current = [0]
            def step(q, d):
                current[0] += 1
                return 1.0
            def forward(calibration, parameters, *, logits_only=False):
                epoch = current[0] if parameters is None else parameters[0]
                logits = np.array([[1, 0], [0, 1]], dtype=np.float32) if epoch == 1 else np.full((2, 2), offset, dtype=np.float32)
                return logits, np.eye(2, dtype=np.float32)
            def match(*args):
                return np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)
            evaluator = _EpochEvaluator(config, labels, labels, forward, lambda exemplars: exemplars, match, None)
            parameters, epoch, loss, history = _run_epochs(config, 2, step, lambda: (current[0],), evaluator, None, None)
            with self.subTest(offset=offset):
                self.assertEqual((parameters, epoch), ((1,), 1))
                self.assertLess(loss, history[1]["balancedCalibrationSDMLoss"])

    def test_prompt_only_scoring_preserves_absent_document(self):
        source = {"id": "prompt-only", "label": -1, "prompt": "Instructions", "embedding": [1, 0]}
        result = score_document(SDMModel(make_artifact()), source)
        self.assertEqual(result["prompt"], "Instructions")
        self.assertNotIn("document", result)
        self.assertEqual(result["embedding"], source["embedding"])

    def test_nested_fingerprints_have_the_same_raw_bundle_and_scoring_policy(self):
        row = {"id": "a", "label": 0, "embedding": [1, 0],
               "metadata": {"representationFingerprint": "fixture-v1"}}
        model = SDMModel(make_artifact())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "source.jsonl"
            raw.write_text(json.dumps(row) + "\n")
            self.assertEqual(Dataset.load(raw).representation_fingerprint, "fixture-v1")
            path = write_dataset_bundle(root / "source.sdmdataset", [row])
            with DatasetBundle.open(path) as bundle:
                self.assertEqual(bundle.fingerprint, "fixture-v1")
            self.assertEqual(score_document(model, row)["representationFingerprint"], "fixture-v1")
            with self.assertRaises(RepresentationMismatchError):
                Dataset.load(raw, representation_fingerprint="different")
            with self.assertRaises(RepresentationMismatchError):
                write_dataset_bundle(root / "bad.sdmdataset", [row], representation={"fingerprint": "different"})
            explicit = {**row, "representationFingerprint": "fixture-v1", "metadata": {"representationFingerprint": "legacy"}}
            self.assertEqual(score_document(model, explicit)["representationFingerprint"], "fixture-v1")
            for invalid in (None, "", " ", 42):
                with self.subTest(invalid=invalid), self.assertRaises(DatasetValidationError):
                    score_document(model, {**row, "representationFingerprint": invalid})
            with self.assertRaises(DatasetValidationError):
                score_document(model, {**row, "metadata": {"representationFingerprint": "different"}}, representation_fingerprint="fixture-v1")

if __name__ == "__main__":
    unittest.main()
