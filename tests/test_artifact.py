# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reexpress_sdm import (
    AdapterWeights,
    ArtifactValidationError,
    ExactL2Index,
    SDMArtifact,
    SDMModel,
    SupportRecord,
    build_artifact,
    load_artifact,
    validate_manifest,
    write_artifact,
)

from helpers import make_artifact


class ArtifactTests(unittest.TestCase):
    @staticmethod
    def _rebuild(
        artifact,
        *,
        support_records=None,
        calibration_rows=None,
        metadata=None,
    ):
        configuration = artifact.manifest["configuration"]
        representation = artifact.manifest["representation"]
        normalization = artifact.manifest["normalization"]
        return build_artifact(
            weights=artifact.weights,
            support_vectors=artifact.support_vectors,
            support_records=(
                artifact.support_records if support_records is None else support_records
            ),
            distance_cdfs=artifact.manifest["distanceCDFs"],
            rescaled_similarity_cdfs=artifact.manifest["rescaledSimilarityCDFs"],
            regions=artifact.regions,
            embedding_dimension=configuration["embeddingDimension"],
            exemplar_dimension=configuration["exemplarDimension"],
            number_of_classes=configuration["numberOfClasses"],
            class_names=configuration["classNames"],
            max_neighbors=configuration["maxNeighbors"],
            q_offset=configuration["qOffset"],
            ood_limit=configuration["oodLimit"],
            alpha_resolution=configuration["alphaResolution"],
            normalization_mean=normalization["mean"],
            normalization_standard_deviation=normalization["standardDeviation"],
            representation_provider=representation["provider"],
            representation_model=representation["model"],
            representation_revision=representation["revision"],
            representation_input_template=representation["inputTemplate"],
            representation_fingerprint=representation["fingerprint"],
            model_id=artifact.model_id,
            metadata=metadata,
            calibration_rows=calibration_rows,
        )

    def test_round_trip_and_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.sdmkitmodel"
            write_artifact(path, make_artifact())
            artifact = load_artifact(path)
            self.assertEqual(artifact.model_id, "fixture-model")
            self.assertEqual(artifact.support_vectors.shape, (4, 2))
            score = SDMModel(artifact).score([[1.0, 0.0]], ids=["x"])[0]
            self.assertEqual(score.id, "x")
            self.assertEqual(score.prediction, 0)
            self.assertEqual(score.nearest_support_id, "s0")
            self.assertEqual(score.floor_q_prime_lower, 2)
            self.assertEqual(score.centroid_region_alpha, 0.8)
            self.assertEqual(score.lower_region_alpha, 0.0)
            self.assertFalse(score.is_in_most_conservative_region)
            self.assertFalse(score.is_in_most_conservative_region_lower)
            serialized = score.to_dict()
            self.assertTrue({"isOOD", "floorQPrime", "floorQPrimeLower"}.isdisjoint(serialized))
            self.assertIn("cumulativeEffectiveSampleSizes", serialized)
            self.assertNotIn("effectiveSampleSizes", serialized)
            self.assertFalse(serialized["isInMostConservativeRegion"])
            self.assertFalse(serialized["isInMostConservativeRegionLower"])
            compact = score.to_dict(detail="compact")
            self.assertTrue({"isOOD", "floorQPrime", "floorQPrimeLower"}.isdisjoint(compact))
            self.assertFalse(compact["isInMostConservativeRegion"])
            self.assertFalse(compact["isInMostConservativeRegionLower"])

            rejected = SDMModel(artifact).score([[50.0, 50.0]])[0]
            self.assertEqual(rejected.centroid_region_alpha, 0.0)
            self.assertEqual(rejected.lower_region_alpha, 0.0)
            self.assertFalse(rejected.is_in_most_conservative_region)
            self.assertFalse(rejected.is_in_most_conservative_region_lower)

    def test_optional_swift_artifact_interop(self):
        path = os.environ.get("SDMKIT_SWIFT_ARTIFACT")
        if path is None:
            return
        model = SDMModel.load(path)
        score = model.score([[1.0, 0.0]])[0]
        self.assertTrue(np.all(np.isfinite(score.sdm)))
        if model.artifact.model_id == "swift-golden-export":
            self.assertEqual(score.prediction, 0)
            self.assertEqual(score.q, 1)
            self.assertEqual(score.d0, 1.0)
            self.assertEqual(score.d, 1.0)
            self.assertAlmostEqual(score.sdm[0], 0.75, places=6)
            self.assertEqual(score.q_prime, 1.0)
            self.assertEqual(score.centroid_region_alpha, 0.0)
            self.assertEqual(score.lower_region_alpha, 0.0)

    def test_checksum_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.sdmkitmodel"
            write_artifact(path, make_artifact())
            support_path = path / "support.f32"
            data = bytearray(support_path.read_bytes())
            data[0] ^= 1
            support_path.write_bytes(data)
            with self.assertRaises(ArtifactValidationError):
                load_artifact(path)

            write_artifact(path, make_artifact(), overwrite=True)
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["checksums"]["support.jsonl"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactValidationError, "checksums"):
                load_artifact(path)

    def test_support_rows_reject_unknown_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.sdmkitmodel"
            write_artifact(path, make_artifact(), include_checksums=False)
            records_path = path / "support.jsonl"
            rows = records_path.read_text(encoding="utf-8").splitlines()
            first = json.loads(rows[0])
            first["unexpected"] = True
            rows[0] = json.dumps(first)
            records_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactValidationError, "unsupported fields"):
                load_artifact(path)

    def test_manifest_rejects_unknown_and_unsorted_values(self):
        manifest = copy.deepcopy(dict(make_artifact().manifest))
        manifest["unknown"] = True
        with self.assertRaises(ArtifactValidationError):
            validate_manifest(manifest)

    def test_manifest_hardens_runtime_configuration(self):
        for field, bad_value in (
            ("qOffset", 1.0),
            ("oodLimit", -1),
            ("alphaResolution", 0.5),
        ):
            with self.subTest(field=field):
                manifest = copy.deepcopy(dict(make_artifact().manifest))
                manifest["configuration"][field] = bad_value
                with self.assertRaises(ArtifactValidationError):
                    validate_manifest(manifest)
        manifest = copy.deepcopy(dict(make_artifact().manifest))
        manifest["configuration"]["classNames"] = ["same", "same"]
        with self.assertRaises(ArtifactValidationError):
            validate_manifest(manifest)

        manifest = copy.deepcopy(dict(make_artifact().manifest))
        manifest["regions"] = [
            {
                "alpha": 0.93,
                "minimumRescaledSimilarity": 1.0,
                "outputThresholds": [0.95, 0.95],
            }
        ]
        with self.assertRaisesRegex(ArtifactValidationError, "alpha ladder"):
            validate_manifest(manifest)

        manifest = copy.deepcopy(dict(make_artifact().manifest))
        manifest["regions"][0]["outputThresholds"][1] = 0.1
        with self.assertRaisesRegex(ArtifactValidationError, r"\[alpha, 1\]"):
            validate_manifest(manifest)

    def test_manifest_accepts_float32_alpha_boundary_thresholds(self):
        manifest = copy.deepcopy(dict(make_artifact().manifest))
        manifest["regions"] = [
            {
                "alpha": 0.8,
                "minimumRescaledSimilarity": 1.0,
                "outputThresholds": [0.8, 0.8],
            },
            {
                "alpha": 0.7,
                "minimumRescaledSimilarity": 1.0,
                "outputThresholds": [0.7, 0.7],
            },
        ]
        validated = validate_manifest(manifest)
        self.assertEqual(
            [region["alpha"] for region in validated["regions"]],
            [0.8, 0.7],
        )

    def test_writer_and_loader_validate_cached_calibration_rows(self):
        valid = {
            "id": "cal-0",
            "label": 0,
            "prediction": 0,
            "sdm": [0.8, 0.2],
            "qPrime": 2.0,
            "q": 2,
            "d0": 0.25,
            "d": 0.75,
            "zPrime": [1.0, 0.0],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.sdmkitmodel"
            artifact = replace(make_artifact(), calibration_rows=(valid,))
            write_artifact(path, artifact)
            loaded = load_artifact(path)
            self.assertEqual(loaded.calibration_rows, (valid,))

        invalid_rows = (
            {**valid, "extra": True},
            {**valid, "sdm": [0.8, 0.8]},
            {**valid, "q": 1.5},
            {**valid, "d": 1.5},
            {**valid, "prediction": 1},
        )
        for index, row in enumerate(invalid_rows):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                artifact = replace(make_artifact(), calibration_rows=(row,))
                with self.assertRaises(ArtifactValidationError):
                    write_artifact(Path(directory) / "invalid.sdmkitmodel", artifact)

    def test_direct_model_construction_validates_support_records(self):
        base = make_artifact()
        first = base.support_records[0]
        second = base.support_records[1]
        invalid_records = {
            "empty id": (replace(first, id=""), *base.support_records[1:]),
            "duplicate id": (
                first,
                replace(second, id=first.id),
                *base.support_records[2:],
            ),
            "invalid label": (replace(first, label=-1), *base.support_records[1:]),
            "invalid prediction": (
                replace(first, predicted_label=2),
                *base.support_records[1:],
            ),
            "non-string document": (
                replace(first, document=123),
                *base.support_records[1:],
            ),
            "non-finite metadata": (
                replace(first, metadata={"value": float("nan")}),
                *base.support_records[1:],
            ),
            "non-string metadata key": (
                replace(first, metadata={1: "value"}),
                *base.support_records[1:],
            ),
            "non-json metadata value": (
                replace(first, metadata={"value": {1, 2}}),
                *base.support_records[1:],
            ),
        }
        for name, records in invalid_records.items():
            with self.subTest(name=name), self.assertRaises(ArtifactValidationError):
                SDMModel(replace(base, support_records=tuple(records)))

        ood_records = (replace(first, label=-99), *base.support_records[1:])
        self.assertIsInstance(
            SDMModel(replace(base, support_records=tuple(ood_records))),
            SDMModel,
        )

    def test_direct_model_construction_validates_cached_calibration(self):
        base = make_artifact()
        valid = {
            "id": "cal-0",
            "label": 0,
            "prediction": 0,
            "sdm": [0.8, 0.2],
            "qPrime": 2.0,
            "q": 2,
            "d0": 0.25,
            "d": 0.75,
            "zPrime": [1.0, 0.0],
        }
        invalid_rows = {
            "empty id": ({**valid, "id": ""},),
            "duplicate id": (valid, dict(valid)),
            "invalid label": ({**valid, "label": -1},),
            "invalid prediction": ({**valid, "prediction": 2},),
            "non-categorical sdm": ({**valid, "sdm": [0.8, 0.8]},),
            "negative qPrime": ({**valid, "qPrime": -1.0},),
            "fractional q": ({**valid, "q": 1.5},),
            "invalid distance": ({**valid, "d": 1.5},),
            "prediction-logit mismatch": ({**valid, "prediction": 1},),
            "unknown field": ({**valid, "unknown": True},),
        }
        for name, rows in invalid_rows.items():
            with self.subTest(name=name), self.assertRaises(ArtifactValidationError):
                SDMModel(replace(base, calibration_rows=rows))

        self.assertIsInstance(
            SDMModel(replace(base, calibration_rows=(valid,))),
            SDMModel,
        )

    def test_build_artifact_uses_full_in_memory_validation(self):
        base = make_artifact()
        duplicate_records = list(base.support_records)
        duplicate_records[1] = replace(duplicate_records[1], id=duplicate_records[0].id)
        with self.assertRaises(ArtifactValidationError):
            self._rebuild(base, support_records=tuple(duplicate_records))

        invalid_calibration = {
            "id": "cal-0",
            "label": 0,
            "prediction": 0,
            "sdm": [0.6, 0.6],
            "qPrime": 1.0,
        }
        with self.assertRaises(ArtifactValidationError):
            self._rebuild(base, calibration_rows=(invalid_calibration,))

        with self.assertRaises(ArtifactValidationError):
            self._rebuild(base, metadata={"notJSON": object()})

    def test_model_owns_read_only_arrays_and_detached_json_data(self):
        base = make_artifact()
        manifest = copy.deepcopy(dict(base.manifest))
        manifest["metadata"] = {"nested": ["original"]}
        projection_weight = np.array(base.weights.projection_weight, copy=True)
        projection_bias = np.array(base.weights.projection_bias, copy=True)
        classifier_weight = np.array(base.weights.classifier_weight, copy=True)
        classifier_bias = np.array(base.weights.classifier_bias, copy=True)
        support_vectors = np.array(base.support_vectors, copy=True)
        support_metadata = {"nested": ["original"]}
        support_records = (
            replace(base.support_records[0], metadata=support_metadata),
            *base.support_records[1:],
        )
        calibration_row = {
            "id": "cal-0",
            "label": 0,
            "prediction": 0,
            "sdm": [0.8, 0.2],
            "qPrime": 2.0,
        }
        caller_artifact = SDMArtifact(
            manifest=manifest,
            weights=AdapterWeights(
                projection_weight=projection_weight,
                projection_bias=projection_bias,
                classifier_weight=classifier_weight,
                classifier_bias=classifier_bias,
            ),
            support_vectors=support_vectors,
            support_records=tuple(support_records),
            calibration_rows=(calibration_row,),
        )
        model = SDMModel(caller_artifact)
        before = model.score([[1.0, 0.0]])[0].to_dict()

        projection_weight.fill(100.0)
        projection_bias.fill(100.0)
        classifier_weight.fill(100.0)
        classifier_bias.fill(100.0)
        support_vectors.fill(100.0)
        manifest["modelID"] = "mutated"
        manifest["metadata"]["nested"][0] = "mutated"
        support_metadata["nested"][0] = "mutated"
        calibration_row["sdm"][0] = 0.0

        self.assertEqual(model.score([[1.0, 0.0]])[0].to_dict(), before)
        self.assertEqual(model.artifact.model_id, "fixture-model")
        self.assertEqual(model.artifact.manifest["metadata"]["nested"], ["original"])
        self.assertEqual(
            model.artifact.support_records[0].metadata["nested"], ["original"]
        )
        self.assertEqual(model.artifact.calibration_rows[0]["sdm"], [0.8, 0.2])
        for array in (
            model.artifact.weights.projection_weight,
            model.artifact.weights.projection_bias,
            model.artifact.weights.classifier_weight,
            model.artifact.weights.classifier_bias,
            model.artifact.support_vectors,
        ):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = 0.0

    def test_exact_index_owns_caller_support_matrix(self):
        support_vectors = np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        index = ExactL2Index(support_vectors)
        before = index.search_one(np.asarray([0.0, 0.0], dtype=np.float32), 2)
        support_vectors.fill(100.0)
        after = index.search_one(np.asarray([0.0, 0.0], dtype=np.float32), 2)
        np.testing.assert_array_equal(after[0], before[0])
        np.testing.assert_array_equal(after[1], before[1])

    def test_writer_uses_whole_second_rfc3339_timestamp(self):
        created_at = str(make_artifact().manifest["createdAt"])
        self.assertRegex(created_at, r"T\d{2}:\d{2}:\d{2}Z$")

    def test_model_rejects_non_numeric_and_overflowing_vectors(self):
        model = SDMModel(make_artifact())
        for bad in (["1", "0"], [True, False], [1.0e300, 0.0]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model.score([bad])

    def test_writer_does_not_coerce_string_or_boolean_tensors(self):
        base = make_artifact()
        for projection in (
            np.asarray([["1", "0"], ["0", "1"]]),
            np.asarray([[True, False], [False, True]]),
        ):
            weights = AdapterWeights(
                projection,
                base.weights.projection_bias,
                base.weights.classifier_weight,
                base.weights.classifier_bias,
            )
            with self.subTest(dtype=str(projection.dtype)), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ArtifactValidationError):
                    write_artifact(
                        Path(directory) / "invalid.sdmkitmodel",
                        replace(base, weights=weights),
                    )

    def test_identity_scoring_reserves_one_neighbor_slot(self):
        class RecordingIndex(ExactL2Index):
            requested = None

            def search_one(self, query, k, exclude_index=None):
                self.requested = k
                return super().search_one(query, k, exclude_index)

        artifact = make_artifact()
        artifact.manifest["configuration"]["maxNeighbors"] = 2
        index = RecordingIndex(artifact.support_vectors)
        model = SDMModel(artifact, index=index)
        # Inspect calibration's search without the independent display search.
        model.score([[1.0, 0.0]], identity_support_indices=[0], nearest_exemplars=0)
        self.assertEqual(index.requested, 1)
        model.score([[1.0, 0.0]], nearest_exemplars=0)
        self.assertEqual(index.requested, 2)

    def test_model_uses_batch_matching_when_backend_explicitly_supports_it(self):
        class BatchOnlyIndex(ExactL2Index):
            calls = 0

            def search_many(self, queries, k, exclude_indices=None):
                self.calls += 1
                return super().search_many(queries, k, exclude_indices)

            def search_one(self, query, k, exclude_index=None):
                raise AssertionError("score should use the batch path")

        artifact = make_artifact()
        index = BatchOnlyIndex(
            artifact.support_vectors, query_tile_size=1, support_tile_size=1
        )
        scores = SDMModel(artifact, index=index).score(
            [[1.0, 0.0], [0.0, 1.0]], ids=["one", "two"]
        )
        self.assertEqual(index.calls, 1)
        self.assertEqual([score.id for score in scores], ["one", "two"])

        index.calls = 0
        SDMModel(artifact, index=index).score(
            [[1.0, 0.0], [0.0, 1.0]], identity_support_indices=[0, 1]
        )
        self.assertEqual(index.calls, 1)

    def test_model_falls_back_for_mixed_identity_exclusions(self):
        class RecordingIndex(ExactL2Index):
            calls = 0

            def search_one(self, query, k, exclude_index=None):
                self.calls += 1
                return super().search_one(query, k, exclude_index)

        artifact = make_artifact()
        index = RecordingIndex(artifact.support_vectors)
        SDMModel(artifact, index=index).score(
            [[1.0, 0.0], [0.0, 1.0]], identity_support_indices=[0, None]
        )
        self.assertEqual(index.calls, 2)

    def test_invalid_manifest_distance_cdf_order_is_rejected(self):
        manifest = copy.deepcopy(dict(make_artifact().manifest))
        manifest["distanceCDFs"][0] = [2.0, 1.0]
        with self.assertRaises(ArtifactValidationError):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
