# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from reexpress_sdm import AdapterWeights, ExactL2Index, SDMModel, SupportRecord, build_artifact
from reexpress_sdm.calibration import NestedCalibrator
from reexpress_sdm.errors import CalibrationError
from reexpress_sdm.math import (
    assigned_region_alpha,
    distance_band,
    distance_quantile,
    dkw_errors,
    effective_sample_sizes,
    region_accepts,
    rescaled_similarity,
    sdm_probabilities,
)
from reexpress_sdm.types import Region


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sdm-golden-v1.json"


class GoldenContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        cls.absolute = cls.fixture["tolerances"]["absolute"]
        cls.relative = cls.fixture["tolerances"]["relative"]

    def assertFloatsClose(self, actual, expected):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), atol=self.absolute, rtol=self.relative
        )

    def test_activation_cases(self):
        for case in self.fixture["activationCases"]:
            with self.subTest(case=case["name"]):
                logits = np.asarray(case["logits"], dtype=np.float32)
                prediction = int(np.argmax(logits))
                probabilities = sdm_probabilities(logits, case["q"], case["d"])
                q_prime = rescaled_similarity(
                    case["q"], float(probabilities[prediction]), self.fixture["constants"]["qOffset"]
                )
                self.assertEqual(prediction, case["expectedPrediction"])
                self.assertFloatsClose(probabilities, case["expectedProbabilities"])
                self.assertFloatsClose(q_prime, case["expectedQPrime"])

    def test_distance_cdf_cases(self):
        for case in self.fixture["distanceCDFCases"]:
            actual = [distance_quantile(query, case["classCDFs"]) for query in case["queries"]]
            self.assertFloatsClose(actual, case["expectedDistanceQuantiles"])

    def test_effective_sample_cases(self):
        for case in self.fixture["effectiveSampleCases"]:
            actual = [effective_sample_sizes(query, case["classQPrimeCDFs"]) for query in case["queries"]]
            self.assertEqual(actual, [tuple(row) for row in case["expectedSampleSizes"]])

    def test_neighbor_cases(self):
        for case in self.fixture["neighborCases"]:
            support = np.asarray(case["supportVectors"], dtype=np.float32)
            index = ExactL2Index(support)
            labels = case["supportLabels"]
            predictions = case["supportPredictions"]
            for query in case["queries"]:
                distances, order = index.search_one(query["vector"], len(support))
                q = 0
                for support_index in order:
                    i = int(support_index)
                    if labels[i] == predictions[i] == query["prediction"]:
                        q += 1
                    else:
                        break
                self.assertEqual(order.tolist(), query["expectedOrder"])
                self.assertEqual(q, query["expectedQ"])
                self.assertFloatsClose(distances[0], query["expectedD0"])
            training = case["trainingQuery"]
            distances, order = index.search_one(
                training["vector"], len(support), exclude_index=training["supportIndex"]
            )
            q = 0
            for support_index in order:
                i = int(support_index)
                if labels[i] == predictions[i] == training["prediction"]:
                    q += 1
                else:
                    break
            self.assertEqual(order.tolist(), training["expectedOrderAfterIdentityExclusion"])
            self.assertEqual(q, training["expectedQ"])
            self.assertFloatsClose(distances[0], training["expectedD0"])

    def test_high_offset_squared_l2_does_not_collapse_distinct_neighbors(self):
        support = np.asarray(
            [[10_000.0, 10_000.0], [10_001.0, 10_000.0], [10_002.0, 10_000.0]],
            dtype=np.float32,
        )
        distances, order = ExactL2Index(support).search_one(
            np.asarray([10_000.0, 10_000.0], dtype=np.float32), 3
        )
        self.assertEqual(order.tolist(), [0, 1, 2])
        self.assertFloatsClose(distances, [0.0, 1.0, 4.0])

    def test_adaptor_case(self):
        case = self.fixture["adaptorCases"][0]
        weights = AdapterWeights(
            np.asarray(case["projectionWeight"], dtype=np.float32),
            np.asarray(case["projectionBias"], dtype=np.float32),
            np.asarray(case["classifierWeight"], dtype=np.float32),
            np.asarray(case["classifierBias"], dtype=np.float32),
        )
        artifact = build_artifact(
            weights=weights,
            support_vectors=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            support_records=(SupportRecord("a", 0, 0), SupportRecord("b", 1, 1)),
            distance_cdfs=((1.0,), (1.0,)),
            rescaled_similarity_cdfs=((1.0,), (1.0,)),
            regions=(),
            embedding_dimension=3,
            exemplar_dimension=2,
            number_of_classes=2,
            max_neighbors=2,
            normalization_mean=case["normalization"]["mean"],
            normalization_standard_deviation=case["normalization"]["standardDeviation"],
            representation_fingerprint="golden-adaptor",
        )
        logits, exemplar = SDMModel(artifact).transform([case["input"]])
        self.assertFloatsClose(exemplar[0], case["expectedExemplar"])
        self.assertFloatsClose(logits[0], case["expectedLogits"])
        self.assertEqual(int(np.argmax(logits[0])), case["expectedPrediction"])

    def test_nested_calibration_and_membership(self):
        for case in self.fixture["nestedCalibrationCases"]:
            regions = NestedCalibrator(
                case["numberOfClasses"], case["alphaResolution"]
            ).fit(case["probabilities"], case["qPrime"], case["labels"])
            expected = case["expectedRegions"]
            self.assertEqual([region.alpha for region in regions], [row["alpha"] for row in expected])
            for actual, wanted in zip(regions, expected):
                self.assertFloatsClose(
                    actual.minimum_rescaled_similarity, wanted["minimumRescaledSimilarity"]
                )
                self.assertFloatsClose(actual.output_thresholds, wanted["outputThresholds"])
            for row in case.get("inference", []):
                self.assertEqual(
                    assigned_region_alpha(
                        row["qPrime"], row["probabilities"], row["prediction"], regions, 0
                    ),
                    row["expectedAlpha"],
                )

    def test_region_membership(self):
        for case in self.fixture["regionMembershipCases"]:
            value = case["region"]
            region = Region(
                value["alpha"], value["minimumRescaledSimilarity"], tuple(value["outputThresholds"])
            )
            for row in case["rows"]:
                self.assertEqual(
                    region_accepts(
                        row["qPrime"], row["probabilities"], row["prediction"], region, 0
                    ),
                    row["expected"],
                )

    def test_calibrator_rejects_non_categorical_rows(self):
        with self.assertRaises(CalibrationError):
            NestedCalibrator(2, 0.1).fit(
                [[0.8, 0.8], [0.2, 0.8]], [1.0, 1.0], [0, 1]
            )

    def test_calibrator_does_not_coerce_cached_values(self):
        calibrator = NestedCalibrator(2, 0.1)
        with self.assertRaises(CalibrationError):
            calibrator.fit([[0.8, 0.2]], ["1.0"], [0])
        with self.assertRaises(CalibrationError):
            calibrator.fit([[0.8, 0.2]], [1.0], [0.0])
        with self.assertRaises(CalibrationError):
            calibrator.fit([[0.8, 0.2]], [1.0], [0], predictions=[0.0])

    def test_dkw_cases(self):
        for case in self.fixture["dkwCases"]:
            errors = dkw_errors(case["sampleSizes"], case["alpha"])
            lower, upper = distance_band(case["distanceQuantile"], errors)
            self.assertFloatsClose(errors, case.get("expectedErrors", errors))
            self.assertFloatsClose(max(errors), case["expectedMaximumError"])
            self.assertFloatsClose(lower, case["expectedLowerDistance"])
            self.assertFloatsClose(upper, case["expectedUpperDistance"])

    def test_public_math_rejects_out_of_contract_inputs(self):
        for kwargs in (
            {"q": -1, "d": 1.0},
            {"q": 1, "d": -0.1},
            {"q": 1, "d": 1.1},
            {"q": 1, "d": 1.0, "q_offset": 1.0},
            {"q": "1", "d": 1.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sdm_probabilities([1.0, 0.0], **kwargs)
        for arguments in (
            (-1, 0.8, 2.0),
            (1, 1.1, 2.0),
            (1, 0.8, 1.0),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                rescaled_similarity(*arguments)
        with self.assertRaises(ValueError):
            dkw_errors([1], 1.0)
        with self.assertRaises(ValueError):
            dkw_errors([1.0], 0.9)


if __name__ == "__main__":
    unittest.main()
