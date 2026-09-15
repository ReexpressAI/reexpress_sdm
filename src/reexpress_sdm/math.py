# Copyright Reexpress AI, Inc. All rights reserved.
"""Numerical primitives matching ``code/reexpress/sdm_model.py``."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .errors import DimensionMismatchError
from .types import Region


MIN_ALPHA_RESOLUTION = 0.00005


def _validate_alpha_resolution(alpha_resolution: float) -> None:
    """Bound the candidate ladder to fewer than 10,000 levels."""
    if (
        isinstance(alpha_resolution, (bool, np.bool_))
        or not isinstance(alpha_resolution, (int, float, np.integer, np.floating))
        or not MIN_ALPHA_RESOLUTION <= alpha_resolution < 0.5
        or not math.isfinite(alpha_resolution)
    ):
        raise ValueError("alpha_resolution must be finite and in [0.00005, 0.5)")


def ladder_alphas(alpha_resolution: float) -> tuple[float, ...]:
    _validate_alpha_resolution(alpha_resolution)
    result: list[float] = []
    k = 1
    while True:
        alpha = round(1.0 - k * alpha_resolution, 10)
        if alpha <= 0.5:
            break
        result.append(alpha)
        k += 1
    return tuple(result)


def stable_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
        squeeze = True
    elif values.ndim == 2:
        squeeze = False
    else:
        raise DimensionMismatchError("softmax input must be one- or two-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("softmax input contains a non-finite value")
    maxima = np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(values - maxima).astype(np.float32, copy=False)
    result = exponentials / np.sum(exponentials, axis=1, keepdims=True, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("softmax produced a non-finite value")
    return result[0] if squeeze else result


def sdm_probabilities(
    logits: np.ndarray,
    q: int | float | np.ndarray,
    d: float | np.ndarray,
    q_offset: float = 2.0,
) -> np.ndarray:
    """Return ``softmax(z' * d * ln(q_offset + q))`` in Float32."""

    if (
        isinstance(q_offset, (bool, np.bool_))
        or not isinstance(q_offset, (int, float, np.integer, np.floating))
        or not math.isfinite(q_offset)
        or q_offset <= 1.0
    ):
        raise ValueError("q_offset must be a finite number greater than one")
    try:
        raw_logits = np.asarray(logits)
        raw_q = np.asarray(q)
        raw_d = np.asarray(d)
    except ValueError as error:
        raise DimensionMismatchError("logits, q, and d must be rectangular numeric arrays") from error
    if raw_logits.dtype.kind not in "iuf" or raw_q.dtype.kind not in "iuf" or raw_d.dtype.kind not in "iuf":
        raise ValueError("logits, q, and d must contain numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        logits_array = raw_logits.astype(np.float32, copy=False)
        q_array = raw_q.astype(np.float32, copy=False).reshape(-1)
        d_array = raw_d.astype(np.float32, copy=False).reshape(-1)
    one_row = logits_array.ndim == 1
    if one_row:
        logits_array = logits_array.reshape(1, -1)
    if logits_array.ndim != 2:
        raise DimensionMismatchError("logits must be one- or two-dimensional")
    if logits_array.shape[1] < 2:
        raise DimensionMismatchError("logits must contain at least two classes")
    if q_array.size == 1:
        q_array = np.repeat(q_array, logits_array.shape[0])
    if d_array.size == 1:
        d_array = np.repeat(d_array, logits_array.shape[0])
    if q_array.size != logits_array.shape[0] or d_array.size != logits_array.shape[0]:
        raise DimensionMismatchError("q and d must have one value per logits row")
    if not np.all(np.isfinite(logits_array)) or not np.all(np.isfinite(q_array)) or not np.all(np.isfinite(d_array)):
        raise ValueError("logits, q, and d must be finite Float32 values")
    if np.any(q_array < 0.0):
        raise ValueError("q cannot be negative")
    if np.any(d_array < 0.0) or np.any(d_array > 1.0):
        raise ValueError("d must be in [0, 1]")
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = logits_array * d_array[:, None] * np.log(q_array + np.float32(q_offset))[:, None]
    result = stable_softmax(scaled)
    return result[0] if one_row else result


def rescaled_similarity(q: int | float, predicted_probability: float, q_offset: float = 2.0) -> float:
    for value, name in ((q, "q"), (predicted_probability, "predicted_probability"), (q_offset, "q_offset")):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")
    if q < 0.0:
        raise ValueError("q cannot be negative")
    if not 0.0 <= predicted_probability <= 1.0:
        raise ValueError("predicted_probability must be in [0, 1]")
    if q_offset <= 1.0:
        raise ValueError("q_offset must be greater than one")
    q_value = np.float32(q)
    base = np.float32(q_offset) + q_value
    if not np.isfinite(q_value) or not np.isfinite(base):
        raise ValueError("q and q_offset must be representable as Float32")
    exponent = np.float32(predicted_probability)
    result = np.minimum(q_value, np.power(base, exponent, dtype=np.float32))
    if not np.isfinite(result):
        raise ValueError("rescaled similarity is non-finite")
    return float(result)


def distance_quantile(
    d0: float,
    distance_cdfs: Sequence[Sequence[float]],
) -> float:
    """Compute the minimum reverse eCDF using left insertion."""

    quantiles: list[float] = []
    for cdf in distance_cdfs:
        if len(cdf) == 0:
            quantiles.append(0.0)
        else:
            cdf_array = np.asarray(cdf, dtype=np.float32)
            insertion = int(np.searchsorted(cdf_array, np.float32(max(d0, 0.0)), side="left"))
            quantiles.append(
                float(
                    np.float32(1.0)
                    - np.float32(insertion) / np.float32(len(cdf))
                )
            )
    return float(np.float32(min(quantiles))) if quantiles else 0.0


def effective_sample_sizes(
    q_prime: float,
    rescaled_similarity_cdfs: Sequence[Sequence[float]],
) -> tuple[int, ...]:
    """Count calibration values ``<= q_prime`` using right insertion."""

    return tuple(
        int(np.searchsorted(np.asarray(cdf, dtype=np.float32), np.float32(q_prime), side="right"))
        if len(cdf) > 0
        else 0
        for cdf in rescaled_similarity_cdfs
    )


def dkw_errors(sample_sizes: Sequence[int], alpha: float) -> tuple[float, ...]:
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not math.isfinite(alpha)
    ):
        raise ValueError("alpha must be a finite number")
    if not 0.5 < alpha < 1.0:
        raise ValueError("alpha must be in (0.5, 1)")
    delta = 1.0 - alpha
    converted_sizes: list[int] = []
    for size in sample_sizes:
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or int(size) < 0:
            raise ValueError("sample sizes must be nonnegative integers")
        converted_sizes.append(int(size))
    numerator = np.log(np.float32(2.0 / delta))
    return tuple(
        float(np.sqrt(numerator / np.float32(2 * size))) if size > 0 else 1.0
        for size in converted_sizes
    )


def region_accepts(
    q_prime: float,
    probabilities: Sequence[float],
    prediction: int,
    region: Region,
    ood_limit: int,
) -> bool:
    q_prime_value = np.float32(q_prime)
    minimum = np.float32(region.minimum_rescaled_similarity)
    if math.floor(float(q_prime_value)) <= ood_limit or q_prime_value < minimum:
        return False
    if len(probabilities) != len(region.output_thresholds):
        raise DimensionMismatchError("region threshold count does not match probabilities")
    if not 0 <= prediction < len(probabilities):
        raise ValueError("prediction is outside the probability classes")
    probability_values = np.asarray(probabilities, dtype=np.float32)
    thresholds = np.asarray(region.output_thresholds, dtype=np.float32)
    above = probability_values >= thresholds
    return int(np.sum(above)) == 1 and bool(above[prediction])


def assigned_region_alpha(
    q_prime: float,
    probabilities: Sequence[float],
    prediction: int,
    regions: Sequence[Region],
    ood_limit: int,
) -> float:
    for region in regions:
        if region_accepts(q_prime, probabilities, prediction, region, ood_limit):
            return region.alpha
    return 0.0


def distance_band(
    distance: float,
    errors: Sequence[float],
) -> tuple[float, float]:
    maximum_error = np.float32(max((float(x) for x in errors), default=1.0))
    distance_value = np.float32(distance)
    lower = np.clip(distance_value - maximum_error, np.float32(0.0), np.float32(1.0))
    upper = np.clip(distance_value + maximum_error, np.float32(0.0), np.float32(1.0))
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Row-batched counterparts of the scalar primitives above. Each performs the
# same Float32 operations per row, so a batch of one row reproduces the scalar
# function; they exist so scoring and training avoid per-row Python overhead.


def distance_quantiles(
    d0: np.ndarray,
    distance_cdfs: Sequence[Sequence[float]],
) -> np.ndarray:
    """Vectorized :func:`distance_quantile` for a vector of nearest distances."""

    d0_array = np.maximum(np.asarray(d0, dtype=np.float32).reshape(-1), np.float32(0.0))
    if len(distance_cdfs) == 0:
        return np.zeros(d0_array.shape[0], dtype=np.float32)
    result = np.full(d0_array.shape[0], np.float32(1.0), dtype=np.float32)
    for cdf in distance_cdfs:
        if len(cdf) == 0:
            return np.zeros(d0_array.shape[0], dtype=np.float32)
        cdf_array = np.asarray(cdf, dtype=np.float32)
        insertion = np.searchsorted(cdf_array, d0_array, side="left").astype(np.float32)
        quantile = np.float32(1.0) - insertion / np.float32(len(cdf))
        np.minimum(result, quantile, out=result)
    return result


def rescaled_similarities(
    q: np.ndarray, predicted_probability: np.ndarray, q_offset: float = 2.0
) -> np.ndarray:
    """Vectorized :func:`rescaled_similarity`: ``min(q, (q_offset + q) ** p)`` in Float32."""

    q_array = np.asarray(q, dtype=np.float32).reshape(-1)
    probability = np.asarray(predicted_probability, dtype=np.float32).reshape(-1)
    base = np.float32(q_offset) + q_array
    result = np.minimum(q_array, np.power(base, probability, dtype=np.float32))
    if not np.all(np.isfinite(result)):
        raise ValueError("rescaled similarity is non-finite")
    return result


def effective_sample_size_matrix(
    q_prime: np.ndarray,
    rescaled_similarity_cdfs: Sequence[Sequence[float]],
) -> np.ndarray:
    """Vectorized :func:`effective_sample_sizes`; returns an ``[N, C]`` int64 matrix."""

    values = np.asarray(q_prime, dtype=np.float32).reshape(-1)
    result = np.zeros((values.shape[0], len(rescaled_similarity_cdfs)), dtype=np.int64)
    for class_index, cdf in enumerate(rescaled_similarity_cdfs):
        if len(cdf) > 0:
            result[:, class_index] = np.searchsorted(
                np.asarray(cdf, dtype=np.float32), values, side="right"
            )
    return result


def dkw_error_matrix(sample_sizes: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorized :func:`dkw_errors` for an ``[N, C]`` sample-size matrix."""

    if not 0.5 < alpha < 1.0:
        raise ValueError("alpha must be in (0.5, 1)")
    sizes = np.asarray(sample_sizes, dtype=np.int64)
    numerator = np.log(np.float32(2.0 / (1.0 - alpha)))
    result = np.ones(sizes.shape, dtype=np.float32)
    positive = sizes > 0
    with np.errstate(divide="ignore"):
        result[positive] = np.sqrt(numerator / (2 * sizes[positive]).astype(np.float32))
    return result


def distance_bands(distance: np.ndarray, errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`distance_band`: clamp ``d -/+ max_c error`` to ``[0, 1]``."""

    distance_array = np.asarray(distance, dtype=np.float32).reshape(-1)
    error_matrix = np.asarray(errors, dtype=np.float32)
    maximum_error = (
        np.max(error_matrix, axis=1) if error_matrix.size else np.ones(distance_array.shape[0], dtype=np.float32)
    ).astype(np.float32, copy=False)
    lower = np.clip(distance_array - maximum_error, np.float32(0.0), np.float32(1.0))
    upper = np.clip(distance_array + maximum_error, np.float32(0.0), np.float32(1.0))
    return lower, upper


def region_accepts_many(
    q_prime: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    region: Region,
    ood_limit: int,
) -> np.ndarray:
    """Vectorized :func:`region_accepts`; returns an ``[N]`` boolean mask."""

    q_prime_array = np.asarray(q_prime, dtype=np.float32).reshape(-1)
    probability_matrix = np.asarray(probabilities, dtype=np.float32)
    prediction_array = np.asarray(predictions, dtype=np.int64).reshape(-1)
    thresholds = np.asarray(region.output_thresholds, dtype=np.float32)
    if probability_matrix.ndim != 2 or probability_matrix.shape[1] != thresholds.shape[0]:
        raise DimensionMismatchError("region threshold count does not match probabilities")
    passes_gate = (np.floor(q_prime_array) > ood_limit) & (
        q_prime_array >= np.float32(region.minimum_rescaled_similarity)
    )
    above = probability_matrix >= thresholds
    singleton = np.sum(above, axis=1) == 1
    predicted_above = above[np.arange(prediction_array.shape[0]), prediction_array]
    return passes_gate & singleton & predicted_above
