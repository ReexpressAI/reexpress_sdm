# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reexpress_sdm import (
    ArtifactValidationError, CalibrationError, SDMModel,
    load_artifact, recalibrate_artifact, validate_artifact, write_artifact,
)

from helpers import make_artifact


class RecalibrationTests(unittest.TestCase):
    def artifact(self):
        source = make_artifact()
        manifest = deepcopy(source.manifest)
        manifest["createdAt"] = "2026-01-01T00:00:00Z"
        history_path = Path(__file__).resolve().parent / "fixtures/contracts-v1/portable-training-run.json"
        manifest["metadata"] = {
            "trainingRun": json.loads(history_path.read_text()),
            "custom": {"retained": [True, None, "text"]},
            "recalibration": {"calibrationDataset": {"identity": "previous-calibration-population"}},
        }
        return validate_artifact(replace(source, manifest=manifest, calibration_rows=(
            {"id": "cal-0", "label": 0, "prediction": 0, "sdm": [0.99, 0.01], "qPrime": 2.0},
            {"id": "cal-1", "label": 1, "prediction": 1, "sdm": [0.01, 0.99], "qPrime": 2.0},
        )))

    def test_saved_diagnostics_produce_new_ladder_without_runtime_or_source_mutation(self):
        source = self.artifact()
        original_manifest = deepcopy(source.manifest)
        with patch("reexpress_sdm.torch_backend._load_torch", side_effect=AssertionError("no runtime needed")):
            result = recalibrate_artifact(source, alpha_resolution=0.05)
        self.assertEqual(source.manifest, original_manifest)
        self.assertNotEqual(result.model_id, source.model_id)
        self.assertEqual(result.configuration["alphaResolution"], 0.05)
        self.assertEqual([region.alpha for region in result.regions], [0.95])
        self.assertEqual(result.regions[0].minimum_rescaled_similarity, 2.0)
        np.testing.assert_array_equal(result.regions[0].output_thresholds, np.asarray([0.99, 0.99], np.float32))
        for name in ("normalization", "representation", "distanceCDFs", "rescaledSimilarityCDFs"):
            self.assertEqual(result.manifest[name], source.manifest[name])
        for name in ("trainingRun", "custom"):
            self.assertEqual(result.manifest["metadata"][name], source.manifest["metadata"][name])
        self.assertEqual(result.manifest["metadata"]["recalibration"]["calibrationDataset"],
                         source.manifest["metadata"]["recalibration"]["calibrationDataset"])
        self.assertEqual(result.calibration_rows, source.calibration_rows)
        self.assertEqual(result.support_records, source.support_records)
        self.assertEqual(result.manifest["producer"]["name"], "reexpress_sdm")
        result.manifest["metadata"]["custom"]["retained"].append("result-only")
        self.assertEqual(source.manifest, original_manifest)

    def test_roundtrip_preserves_payloads_and_numerical_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            original_path = Path(directory) / "original.sdmkitmodel"
            result_path = Path(directory) / "recalibrated.sdmkitmodel"
            write_artifact(original_path, self.artifact())
            source = load_artifact(original_path)
            result = recalibrate_artifact(source, alpha_resolution=0.05)
            self.assertNotIn("checksums", result.manifest)
            write_artifact(result_path, result)
            restored = load_artifact(result_path)
            for name in ("weights.f32", "support.f32", "support.jsonl", "calibration.jsonl"):
                self.assertEqual((original_path / name).read_bytes(), (result_path / name).read_bytes(), name)
            before = SDMModel(source, device="cpu").score([[1.0, 0.0], [0.0, 1.0], [0.3, 0.7]])
            after = SDMModel(restored, device="cpu").score([[1.0, 0.0], [0.0, 1.0], [0.3, 0.7]])
            changed_fields = {"modelID", "centroidRegionAlpha", "lowerRegionAlpha",
                              "isInMostConservativeRegion", "isInMostConservativeRegionLower"}
            for old_score, new_score in zip(before, after):
                self.assertEqual({k: v for k, v in old_score.to_dict().items() if k not in changed_fields},
                                 {k: v for k, v in new_score.to_dict().items() if k not in changed_fields})

    def test_repeated_recalibration_preserves_training_lineage(self):
        source = self.artifact()
        first = recalibrate_artifact(source, alpha_resolution=0.05)
        second = recalibrate_artifact(first, alpha_resolution=0.01)
        lineage = second.manifest["metadata"]["workbenchRecalibration"]
        self.assertEqual(lineage, {
            "sourceModelID": first.model_id, "trainingModelID": source.model_id,
            "trainingCreatedAt": source.manifest["createdAt"],
            "previousAlphaResolution": 0.05, "calibrationCount": 2,
        })
        self.assertEqual(second.manifest["metadata"]["recalibration"]["sourceModelID"], first.model_id)
        self.assertNotEqual(second.model_id, first.model_id)

    def test_missing_diagnostics_and_invalid_inputs_fail_clearly(self):
        for rows in (None, ()):
            with self.subTest(rows=rows), self.assertRaisesRegex(CalibrationError, "calibration.jsonl"):
                recalibrate_artifact(replace(make_artifact(), calibration_rows=rows), alpha_resolution=0.05)
        for resolution in (0, -0.1, 0.5, 1, float("nan"), float("inf"), True, "0.05", None):
            with self.subTest(resolution=resolution), self.assertRaisesRegex(CalibrationError, "alpha_resolution"):
                recalibrate_artifact(self.artifact(), alpha_resolution=resolution)
        source = self.artifact()
        malformed = ({**source.calibration_rows[0], "label": -1}, source.calibration_rows[1])
        with self.assertRaises(ArtifactValidationError):
            recalibrate_artifact(replace(source, calibration_rows=malformed), alpha_resolution=0.05)


if __name__ == "__main__":
    unittest.main()
