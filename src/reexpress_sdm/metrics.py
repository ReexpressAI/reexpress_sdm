# Copyright Reexpress AI, Inc. All rights reserved.
"""Evaluation metrics for calibrated selective classification."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .errors import DimensionMismatchError
from .types import SDMScore


def _cell(correct: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    count = int(np.sum(mask))
    correct_count = int(np.sum(correct[mask])) if count else 0
    return {
        "count": count,
        "correct": correct_count,
        "accuracy": correct_count / count if count else None,
        "defined": count > 0,
    }


def _conditional_summary(
    predictions: np.ndarray,
    labels: np.ndarray,
    selected: np.ndarray,
    number_of_classes: int,
) -> dict[str, Any]:
    correct = predictions == labels
    marginal = _cell(correct, selected)
    by_true = [
        {"class": class_index, **_cell(correct, selected & (labels == class_index))}
        for class_index in range(number_of_classes)
    ]
    by_prediction = [
        {"class": class_index, **_cell(correct, selected & (predictions == class_index))}
        for class_index in range(number_of_classes)
    ]

    def minima(cells: list[dict[str, Any]]) -> tuple[float | None, float | None, bool]:
        defined = [float(cell["accuracy"]) for cell in cells if cell["defined"]]
        all_defined = all(bool(cell["defined"]) for cell in cells)
        return (
            min(defined) if defined else None,
            min(defined) if defined and all_defined else None,
            all_defined,
        )

    min_true_defined, min_true_complete, all_true_defined = minima(by_true)
    min_prediction_defined, min_prediction_complete, all_prediction_defined = minima(by_prediction)
    complete_minimum = None
    if min_true_complete is not None and min_prediction_complete is not None:
        complete_minimum = min(min_true_complete, min_prediction_complete)
    return {
        "marginal": marginal,
        "byTrueClass": by_true,
        "byPrediction": by_prediction,
        "minimumDefinedTrueClassAccuracy": min_true_defined,
        "minimumTrueClassAccuracy": min_true_complete,
        "allTrueClassCellsDefined": all_true_defined,
        "minimumDefinedPredictionConditionalAccuracy": min_prediction_defined,
        "minimumPredictionConditionalAccuracy": min_prediction_complete,
        "allPredictionConditionalCellsDefined": all_prediction_defined,
        "minimumConditionalAccuracy": complete_minimum,
        "allConditionalCellsDefined": all_true_defined and all_prediction_defined,
    }


def evaluate_scores(
    scores: Sequence[SDMScore],
    labels: Sequence[int],
    *,
    number_of_classes: int,
    alphas: Sequence[float] | None = None,
) -> dict[str, Any]:
    if isinstance(number_of_classes, bool) or not isinstance(number_of_classes, int) or number_of_classes < 2:
        raise ValueError("number_of_classes must be at least two")
    if len(scores) != len(labels):
        raise DimensionMismatchError("scores and labels must have equal lengths")
    raw_labels = np.asarray(labels)
    if raw_labels.ndim != 1 or raw_labels.dtype.kind not in "iu":
        raise ValueError("labels must be a one-dimensional array of actual integers")
    label_array_all = raw_labels.astype(np.int64, copy=False)
    allowed_labels = (
        (label_array_all == -1)
        | (label_array_all == -99)
        | ((label_array_all >= 0) & (label_array_all < number_of_classes))
    )
    if not np.all(allowed_labels):
        raise ValueError("labels must be -1, -99, or a configured class")

    predictions: list[int] = []
    for score in scores:
        prediction = score.prediction
        if (
            isinstance(prediction, (bool, np.bool_))
            or not isinstance(prediction, (int, np.integer))
            or prediction < 0
            or prediction >= number_of_classes
        ):
            raise ValueError("score predictions must be configured class indices")
        predictions.append(int(prediction))

        float_vectors = {
            "z_prime": score.z_prime,
            "sdm": score.sdm,
            "effective_sample_size_errors": score.effective_sample_size_errors,
            "sdm_lower": score.sdm_lower,
            "sdm_upper": score.sdm_upper,
        }
        for name, values in float_vectors.items():
            array = np.asarray(values)
            if array.ndim != 1 or array.size != number_of_classes:
                raise DimensionMismatchError(
                    f"score {name} must contain number_of_classes values"
                )
            if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
                raise ValueError(f"score {name} must contain only finite numbers")

        sample_sizes = np.asarray(score.effective_sample_sizes)
        if sample_sizes.ndim != 1 or sample_sizes.size != number_of_classes:
            raise DimensionMismatchError(
                "score effective_sample_sizes must contain number_of_classes values"
            )
        if sample_sizes.dtype.kind not in "iu" or np.any(sample_sizes < 0):
            raise ValueError("score effective_sample_sizes must contain nonnegative integers")

    known = (label_array_all >= 0) & (label_array_all < number_of_classes)
    predictions_all = np.asarray(predictions, dtype=np.int64)
    labels_known = label_array_all[known]
    predictions_known = predictions_all[known]
    centroid = np.asarray([score.centroid_region_alpha for score in scores], dtype=np.float64)[known]
    lower = np.asarray([score.lower_region_alpha for score in scores], dtype=np.float64)[known]
    if alphas is None:
        alphas = sorted(
            {float(value) for value in np.concatenate((centroid, lower)) if value > 0.0},
            reverse=True,
        )
    else:
        converted_alphas: set[float] = set()
        for value in alphas:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(value)
            ):
                raise ValueError("alphas must contain finite numbers")
            if value == 0.0:
                continue
            if not 0.5 < value < 1.0:
                raise ValueError("nonzero alphas must be in (0.5, 1)")
            converted_alphas.add(float(value))
        alphas = sorted(converted_alphas, reverse=True)

    def estimator_report(assignments: np.ndarray) -> dict[str, Any]:
        per_alpha: list[dict[str, Any]] = []
        for alpha in alphas or ():
            selected = assignments >= alpha
            summary = _conditional_summary(
                predictions_known, labels_known, selected, number_of_classes
            )
            summary.update(
                {
                    "alpha": alpha,
                    "admissionCount": int(np.sum(selected)),
                    "admission": float(np.mean(selected)) if selected.size else None,
                }
            )
            per_alpha.append(summary)
        rejected = assignments == 0.0
        return {
            "perAlphaCumulative": per_alpha,
            "rejected": {
                "count": int(np.sum(rejected)),
                "proportion": float(np.mean(rejected)) if rejected.size else None,
            },
        }

    overall_selected = np.ones(labels_known.size, dtype=bool)
    return {
        "totalRows": len(scores),
        "evaluatedRows": int(np.sum(known)),
        "unlabeledRows": int(np.sum(label_array_all == -1)),
        "explicitOODRows": int(np.sum(label_array_all == -99)),
        "overall": _conditional_summary(
            predictions_known, labels_known, overall_selected, number_of_classes
        ),
        "centroid": estimator_report(centroid),
        "lower": estimator_report(lower),
    }


class Evaluator:
    def __init__(self, number_of_classes: int, alphas: Sequence[float] | None = None):
        if number_of_classes < 2:
            raise ValueError("number_of_classes must be at least two")
        self.number_of_classes = number_of_classes
        self.alphas = tuple(alphas) if alphas is not None else None

    def evaluate(self, scores: Sequence[SDMScore], labels: Sequence[int]) -> dict[str, Any]:
        return evaluate_scores(
            scores,
            labels,
            number_of_classes=self.number_of_classes,
            alphas=self.alphas,
        )
