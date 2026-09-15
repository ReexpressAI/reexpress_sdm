# Copyright Reexpress AI, Inc. All rights reserved.
import copy
import json
import os
from pathlib import Path
import unittest

from reexpress_sdm.training_metadata import validate_training_run


class PortableTrainingMetadataTests(unittest.TestCase):
    def setUp(self):
        self.value = json.loads((Path(__file__).resolve().parent / "fixtures/contracts-v1/portable-training-run.json").read_text())

    def test_common_fixture_preserves_balanced_metrics_nulls_and_uint64_seed(self):
        restored = validate_training_run(self.value)
        self.assertEqual(restored, self.value)
        self.assertEqual(int(restored["configuration"]["seed"]), 2**64 - 1)
        self.assertIsNone(restored["history"][0]["balancedMeanTrainingQ"])
        self.assertEqual(restored["history"][1]["balancedMeanTrainingQ"], 13 / 6)
        self.assertEqual(restored["history"][1]["balancedTrainingAccuracy"], .5)

    def test_rejects_partial_scores_invalid_selection_and_ambiguous_seeds(self):
        changes = [
            lambda v: v["history"][1].update(balancedMeanTrainingQ=None),
            lambda v: v.update(bestEpoch=1),
            lambda v: v.update(bestBalancedCalibrationLoss=.1),
            lambda v: v.update(stoppedEarly=False),
            lambda v: v["history"][0].update(epoch=2),
            lambda v: v["history"][0].update(balancedTrainingAccuracy=1.01),
            lambda v: v["configuration"].update(seed=2**64 - 1),
            lambda v: v["configuration"].update(seed=str(2**64)),
            lambda v: v.update(sourceModelID=None),
            lambda v: v["configuration"].update(learningRate=1e-50),
            lambda v: v["configuration"].update(learningRate=1e50),
        ]
        for change in changes:
            with self.subTest(change=change):
                value = copy.deepcopy(self.value)
                change(value)
                with self.assertRaises(ValueError):
                    validate_training_run(value)

    def test_optional_timing_accepts_older_exports_and_rejects_invalid_values(self):
        for duration in (None, 0, 12.5):
            with self.subTest(duration=duration):
                value = copy.deepcopy(self.value)
                value["durationSeconds"] = duration
                value["history"][0]["durationSeconds"] = duration
                self.assertEqual(validate_training_run(value), value)
        for duration in (-1, float("nan"), float("inf"), True, "1.0"):
            for epoch_field in (False, True):
                with self.subTest(duration=duration, epoch_field=epoch_field):
                    value = copy.deepcopy(self.value)
                    target = value["history"][0] if epoch_field else value
                    target["durationSeconds"] = duration
                    with self.assertRaisesRegex(ValueError, "durationSeconds"):
                        validate_training_run(value)

    def test_training_rejects_learning_rates_that_cannot_round_trip_float32(self):
        from reexpress_sdm import TrainingConfig
        for rate in (1e-50, 1e50):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, "Float32"):
                TrainingConfig(2, learning_rate=rate)

    @unittest.skipUnless(os.environ.get("SDMKIT_SWIFT_TRAINING_ARTIFACT"), "optional native Swift training export")
    def test_native_swift_training_export_history_and_scores(self):
        import numpy as np
        from reexpress_sdm import SDMModel, load_artifact
        path = Path(os.environ["SDMKIT_SWIFT_TRAINING_ARTIFACT"])
        expected = json.loads(path.with_suffix(".expected.json").read_text())
        artifact = load_artifact(path)
        self.assertEqual(artifact.model_id, expected["modelID"])
        self.assertEqual(artifact.manifest["metadata"]["trainingRun"], expected["trainingRun"])
        validate_training_run(artifact.manifest["metadata"]["trainingRun"])
        model = SDMModel(artifact)
        scores = model.score([row["embedding"] for row in expected["evaluation"]])
        for result, reference in zip(scores, expected["scores"]):
            actual = result.to_dict()
            for key, value in reference.items():
                if isinstance(value, bool) or key in ("prediction", "q", "nearestSupportIndex", "centroidRegionAlpha", "lowerRegionAlpha", "cumulativeEffectiveSampleSizes"):
                    self.assertEqual(actual[key], value, key)
                else:
                    np.testing.assert_allclose(actual[key], value, atol=expected["floatTolerance"], rtol=expected["floatTolerance"], err_msg=key)


if __name__ == "__main__":
    unittest.main()
