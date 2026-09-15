# Copyright Reexpress AI, Inc. All rights reserved.
"""Descriptive SDM score summaries for individual datasets."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .types import SDMScore


def _signal_arrays(scores: Sequence[SDMScore]) -> dict[str, np.ndarray]:
    def predicted_logit(score: SDMScore) -> float:
        return score.z_prime[score.prediction]

    def margin(score: SDMScore) -> float:
        ordered = sorted(score.z_prime, reverse=True)
        return ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]

    return {
        "q": np.asarray([score.q for score in scores], dtype=np.float64),
        "qPrime": np.asarray([score.q_prime for score in scores], dtype=np.float64),
        "d": np.asarray([score.d for score in scores], dtype=np.float64),
        "d0": np.asarray([score.d0 for score in scores], dtype=np.float64),
        "predictedLogit": np.asarray([predicted_logit(score) for score in scores]),
        "logitMargin": np.asarray([margin(score) for score in scores]),
        "predictedSDM": np.asarray([score.sdm[score.prediction] for score in scores]),
        "centroidAssignedAlpha": np.asarray(
            [score.centroid_region_alpha for score in scores], dtype=np.float64
        ),
        "lowerAssignedAlpha": np.asarray(
            [score.lower_region_alpha for score in scores], dtype=np.float64
        ),
    }


def _summary(values: np.ndarray, histogram_bins: int) -> dict[str, Any]:
    if values.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "quantiles": {},
            "histogram": {"edges": [], "counts": []},
        }
    quantile_levels = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    quantiles = np.quantile(values, quantile_levels)
    if float(np.min(values)) == float(np.max(values)):
        edges = np.asarray([float(values[0]) - 0.5, float(values[0]) + 0.5])
    else:
        edges = np.linspace(float(np.min(values)), float(np.max(values)), histogram_bins + 1)
    counts, edges = np.histogram(values, bins=edges)
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "quantiles": {
            str(level): float(value) for level, value in zip(quantile_levels, quantiles)
        },
        "histogram": {
            "edges": [float(value) for value in edges],
            "counts": [int(value) for value in counts],
        },
    }


def distribution_summary(
    scores: Sequence[SDMScore], *, number_of_classes: int, histogram_bins: int = 10
) -> dict[str, Any]:
    if histogram_bins < 1:
        raise ValueError("histogram_bins must be positive")
    predictions = np.asarray([score.prediction for score in scores], dtype=np.int64)
    return {
        "count": len(scores),
        "signals": {
            name: _summary(values, histogram_bins)
            for name, values in _signal_arrays(scores).items()
        },
        "predictionFrequencies": [
            {
                "class": class_index,
                "count": int(np.sum(predictions == class_index)),
                "proportion": float(np.mean(predictions == class_index)) if predictions.size else None,
            }
            for class_index in range(number_of_classes)
        ],
    }
