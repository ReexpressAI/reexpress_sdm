# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

from dataclasses import replace
import unittest

from reexpress_sdm import DimensionMismatchError, EstimatorKind, Evaluator, SDMController, SelectionPolicy
from reexpress_sdm.monitoring import distribution_summary
from reexpress_sdm.model import SDMModel

from helpers import make_artifact


class ControlAndMetricsTests(unittest.TestCase):
    def setUp(self):
        model = SDMModel(make_artifact())
        base = model.score([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
        self.scores = (
            replace(base[0], prediction=0, centroid_region_alpha=0.95, lower_region_alpha=0.95),
            replace(base[1], prediction=1, centroid_region_alpha=0.90, lower_region_alpha=0.0),
            replace(base[2], prediction=1, centroid_region_alpha=0.95, lower_region_alpha=0.90),
            replace(base[3], prediction=0, centroid_region_alpha=0.0, lower_region_alpha=0.0),
        )
        self.labels = (0, 0, 1, 1)

    def test_evaluator_reports_all_conditional_cells_and_minima(self):
        report = Evaluator(2, alphas=(0.95, 0., 0.90)).evaluate(self.scores, self.labels)
        centroid_95 = report["centroid"]["perAlphaCumulative"][0]
        self.assertEqual(centroid_95["coverageCount"], 2)
        self.assertEqual(centroid_95["coverage"], 0.5)
        self.assertEqual(centroid_95["marginal"]["accuracy"], 1.0)
        self.assertEqual(centroid_95["minimumConditionalAccuracy"], 1.0)
        centroid_90 = report["centroid"]["perAlphaCumulative"][1]
        self.assertAlmostEqual(centroid_90["marginal"]["accuracy"], 2.0 / 3.0)
        self.assertEqual(centroid_90["minimumConditionalAccuracy"], 0.5)
        lower_95 = report["lower"]["perAlphaCumulative"][0]
        self.assertFalse(lower_95["allConditionalCellsDefined"])
        self.assertIsNone(lower_95["minimumConditionalAccuracy"])
        self.assertEqual(lower_95["minimumDefinedTrueClassAccuracy"], 1.0)
        self.assertEqual(lower_95["byTrueClass"][1]["count"], 0)
        self.assertFalse(lower_95["byTrueClass"][1]["defined"])

    def test_policy_does_not_change_score_facts(self):
        policy = SelectionPolicy(0.95, EstimatorKind.CENTROID)
        decision = policy.decide(self.scores[0])
        self.assertEqual(decision.action, "allow")
        self.assertEqual(decision.reason, "accepted")
        rejected = policy.decide(self.scores[1])
        self.assertEqual(rejected.action, "review")
        self.assertEqual(rejected.reason, "below-required-alpha")
        self.assertEqual(self.scores[1].centroid_region_alpha, 0.90)

    def test_default_controller_decides_with_centroid_and_explicit_lower_is_preserved(self):
        controller = SDMController(SDMModel(make_artifact()))
        default, = controller.decide([self.scores[2]])
        lower, = controller.decide([self.scores[2]], policy=SelectionPolicy(estimator=EstimatorKind.LOWER))
        self.assertEqual((default.estimator, default.observed_alpha, default.action),
                         (EstimatorKind.CENTROID, 0.95, 'allow'))
        self.assertEqual((lower.estimator, lower.observed_alpha, lower.action),
                         (EstimatorKind.LOWER, 0.90, 'review'))

    def test_evaluator_does_not_coerce_labels(self):
        with self.assertRaises(ValueError):
            Evaluator(2).evaluate(self.scores, (0.0, 0.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            Evaluator(2).evaluate(self.scores, (0, -2, 1, 1))

    def test_zero_similarity_policy_is_preserved_without_serialized_gate(self):
        policy = SelectionPolicy(0.95, EstimatorKind.CENTROID)
        score = replace(self.scores[0], q=0, q_prime=0.0, floor_q_prime=0,
                        q_prime_lower=0.0, floor_q_prime_lower=0, is_ood=True,
                        centroid_region_alpha=0.0, lower_region_alpha=0.0)
        decision = policy.decide(score)
        self.assertEqual(decision.action, "review")
        self.assertEqual(decision.reason, "zero-similarity")
        self.assertNotIn("isOOD", score.to_dict())

    def test_evaluator_excludes_and_counts_sentinel_labels(self):
        scores = (
            replace(self.scores[0], centroid_region_alpha=0.95, lower_region_alpha=0.95),
            replace(self.scores[1], centroid_region_alpha=0.90, lower_region_alpha=0.90),
            replace(self.scores[2], centroid_region_alpha=0.80, lower_region_alpha=0.80),
            replace(self.scores[3], prediction=1, centroid_region_alpha=0.95, lower_region_alpha=0.95),
        )
        report = Evaluator(2).evaluate(scores, (0, -1, -99, 1))

        self.assertEqual(report["totalRows"], 4)
        self.assertEqual(report["evaluatedRows"], 2)
        self.assertEqual(report["unlabeledRows"], 1)
        self.assertEqual(report["explicitOODRows"], 1)
        self.assertEqual(report["overall"]["marginal"]["count"], 2)
        self.assertEqual(report["overall"]["marginal"]["accuracy"], 1.0)
        self.assertEqual(
            [row["alpha"] for row in report["centroid"]["perAlphaCumulative"]],
            [0.95],
        )
        self.assertEqual(report["centroid"]["perAlphaCumulative"][0]["coverage"], 1.0)

    def test_evaluator_rejects_malformed_scores(self):
        with self.assertRaises(ValueError):
            Evaluator(2).evaluate(
                (replace(self.scores[0], prediction=2),),
                (0,),
            )
        with self.assertRaises(ValueError):
            Evaluator(2).evaluate(
                (replace(self.scores[0], prediction=True),),
                (0,),
            )

        class_sized_vectors = (
            "z_prime",
            "sdm",
            "effective_sample_sizes",
            "effective_sample_size_errors",
            "sdm_lower",
            "sdm_upper",
        )
        for field in class_sized_vectors:
            with self.subTest(field=field), self.assertRaises(DimensionMismatchError):
                Evaluator(2).evaluate(
                    (replace(self.scores[0], **{field: (1,)}),),
                    (0,),
                )

        with self.assertRaises(ValueError):
            Evaluator(2).evaluate(
                (replace(self.scores[0], z_prime=(float("nan"), 0.0)),),
                (0,),
            )
        with self.assertRaises(ValueError):
            Evaluator(2).evaluate(
                (replace(self.scores[0], effective_sample_sizes=(1, -1)),),
                (0,),
            )

    def test_distribution_summary_preserves_signals_and_prediction_frequencies(self):
        summary = distribution_summary(self.scores, number_of_classes=2, histogram_bins=4)
        self.assertEqual(set(summary), {"count", "signals", "predictionFrequencies"})
        self.assertEqual(summary["count"], 4)
        self.assertEqual(set(summary["signals"]), {
            "q", "qPrime", "d", "d0", "predictedLogit", "logitMargin", "predictedSDM",
            "centroidAssignedAlpha", "lowerAssignedAlpha",
        })
        self.assertEqual(summary["predictionFrequencies"], [
            {"class": 0, "count": 2, "proportion": 0.5},
            {"class": 1, "count": 2, "proportion": 0.5},
        ])
        for signal in summary["signals"].values():
            self.assertEqual(set(signal), {"count", "minimum", "maximum", "mean", "quantiles", "histogram"})
            self.assertEqual(signal["count"], 4)
            self.assertEqual(sum(signal["histogram"]["counts"]), 4)
        controller = SDMController(SDMModel(make_artifact()))
        self.assertEqual(controller.summarize_distribution(self.scores, histogram_bins=4), summary)

    def test_comparison_api_is_removed(self):
        import reexpress_sdm
        from reexpress_sdm import monitoring
        for api in (reexpress_sdm, monitoring, SDMController):
            self.assertFalse(hasattr(api, "compare_distributions"))


if __name__ == "__main__":
    unittest.main()
