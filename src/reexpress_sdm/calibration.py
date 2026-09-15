# Copyright Reexpress AI, Inc. All rights reserved.
"""Nested high-reliability region calibration."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .errors import CalibrationError, DimensionMismatchError
from .math import _validate_alpha_resolution, ladder_alphas
from .types import Region


def _class_threshold(values: Sequence[float], alpha: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    quantile_index = min(int(round(1.0 - alpha, 10) * len(ordered)), len(ordered) - 1)
    return max(ordered[quantile_index], 0.0)


def _candidate_region(
    probabilities: np.ndarray,
    q_prime: np.ndarray,
    labels: np.ndarray,
    candidate_rows: np.ndarray,
    alpha: float,
    number_of_classes: int,
) -> Region | None:
    """Find the first passing q-prime cutoff in one linear candidate sweep."""

    if candidate_rows.size == 0:
        return None
    # A stable order is not needed for the class counts, but makes grouping and
    # diagnostics invariant when equal q-prime values originate in many rows.
    order = np.argsort(q_prime[candidate_rows], kind="stable")
    sorted_rows = candidate_rows[order]
    sorted_q_prime = q_prime[sorted_rows]
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(sorted_q_prime[1:] != sorted_q_prime[:-1]).astype(np.int64)
            + 1,
        )
    )
    sorted_labels = labels[sorted_rows]
    alpha_gate = np.float32(alpha)
    below_gate = (
        probabilities[sorted_rows, sorted_labels] < alpha_gate
    )
    counts = np.bincount(sorted_labels, minlength=number_of_classes).astype(
        np.int64, copy=False
    )
    below_counts = np.bincount(
        sorted_labels[below_gate], minlength=number_of_classes
    ).astype(np.int64, copy=False)
    tail_fraction = round(1.0 - alpha, 10)
    allowed_below = np.asarray(
        [int(tail_fraction * int(count)) for count in counts], dtype=np.int64
    )
    valid = (counts > 0) & (below_counts <= allowed_below)
    invalid_count = int(np.count_nonzero(~valid))

    for group_index, start in enumerate(starts):
        if invalid_count == 0:
            retained_rows = sorted_rows[int(start) :]
            retained_labels = sorted_labels[int(start) :]
            thresholds: list[float] = []
            for true_class in range(number_of_classes):
                class_values = probabilities[
                    retained_rows[retained_labels == true_class], true_class
                ]
                quantile_index = min(
                    int(tail_fraction * int(class_values.size)),
                    int(class_values.size) - 1,
                )
                # The gate guarantees that every class is represented here.
                threshold = float(np.partition(class_values, quantile_index)[quantile_index])
                thresholds.append(max(threshold, 0.0))
            return Region(
                alpha,
                float(sorted_q_prime[int(start)]),
                tuple(thresholds),
            )

        stop = (
            int(starts[group_index + 1])
            if group_index + 1 < starts.size
            else int(sorted_rows.size)
        )
        group_labels = sorted_labels[int(start) : stop]
        affected, removed_counts = np.unique(group_labels, return_counts=True)
        before = valid[affected]
        counts[affected] -= removed_counts.astype(np.int64, copy=False)
        if np.any(below_gate[int(start) : stop]):
            bad_labels, removed_bad = np.unique(
                group_labels[below_gate[int(start) : stop]], return_counts=True
            )
            below_counts[bad_labels] -= removed_bad.astype(np.int64, copy=False)
        allowed_below = np.asarray(
            [int(tail_fraction * int(counts[class_index])) for class_index in affected],
            dtype=np.int64,
        )
        after = (counts[affected] > 0) & (
            below_counts[affected] <= allowed_below
        )
        invalid_count += int(np.count_nonzero(~after)) - int(np.count_nonzero(~before))
        valid[affected] = after
    return None


class NestedCalibrator:
    def __init__(self, number_of_classes: int, alpha_resolution: float, ood_limit: int = 0):
        if isinstance(number_of_classes, bool) or not isinstance(number_of_classes, int):
            raise CalibrationError("number_of_classes must be an integer")
        if number_of_classes < 2:
            raise CalibrationError("number_of_classes must be at least two")
        if isinstance(ood_limit, bool) or not isinstance(ood_limit, int) or ood_limit < 0:
            raise CalibrationError("ood_limit must be a nonnegative integer")
        try:
            _validate_alpha_resolution(alpha_resolution)
        except ValueError as error:
            raise CalibrationError(str(error)) from error
        self.number_of_classes = number_of_classes
        self.alpha_resolution = alpha_resolution
        self.ood_limit = ood_limit

    def fit(
        self,
        probabilities: Sequence[Sequence[float]] | np.ndarray,
        q_prime: Sequence[float] | np.ndarray,
        labels: Sequence[int] | np.ndarray,
        predictions: Sequence[int] | np.ndarray | None = None,
    ) -> tuple[Region, ...]:
        """Construct the descending residual alpha ladder.

        ``predictions`` must be the argmax of raw logits when parity is
        possible. It defaults to probability argmax for cached data that does
        not contain such cases.
        """

        try:
            raw_probabilities = np.asarray(probabilities)
            raw_q_prime = np.asarray(q_prime)
            raw_labels = np.asarray(labels)
        except ValueError as error:
            raise CalibrationError("calibration inputs must be rectangular arrays") from error
        if raw_probabilities.dtype.kind not in "iuf" or raw_q_prime.dtype.kind not in "iuf":
            raise CalibrationError("probabilities and q_prime must be numeric")
        if raw_labels.dtype.kind not in "iu":
            raise CalibrationError("labels must be integers")
        with np.errstate(over="ignore", invalid="ignore"):
            probabilities_array = raw_probabilities.astype(np.float32, copy=False)
            q_prime_array = raw_q_prime.astype(np.float32, copy=False).reshape(-1)
        labels_array = raw_labels.astype(np.int64, copy=False).reshape(-1)
        if probabilities_array.ndim != 2 or probabilities_array.shape[1] != self.number_of_classes:
            raise DimensionMismatchError(
                f"probabilities must have shape [N, {self.number_of_classes}]"
            )
        row_count = probabilities_array.shape[0]
        if q_prime_array.size != row_count or labels_array.size != row_count:
            raise DimensionMismatchError("probabilities, q_prime, and labels must have equal row counts")
        if predictions is None:
            predictions_array = np.argmax(probabilities_array, axis=1).astype(np.int64)
        else:
            raw_predictions = np.asarray(predictions)
            if raw_predictions.dtype.kind not in "iu":
                raise CalibrationError("predictions must contain actual integers")
            predictions_array = raw_predictions.astype(np.int64, copy=False).reshape(-1)
            if predictions_array.size != row_count:
                raise DimensionMismatchError("predictions must have one value per row")
        if row_count == 0:
            raise CalibrationError("calibration data cannot be empty")
        if not np.all(np.isfinite(probabilities_array)) or not np.all(np.isfinite(q_prime_array)):
            raise CalibrationError("calibration values must be finite")
        if np.any(probabilities_array < 0.0) or np.any(probabilities_array > 1.0):
            raise CalibrationError("probabilities must be in [0, 1]")
        if np.any(
            np.abs(
                np.sum(probabilities_array, axis=1, dtype=np.float32) - np.float32(1.0)
            )
            > np.float32(1.0e-5)
        ):
            raise CalibrationError(
                "each categorical probability row must sum to one within absolute tolerance 1e-5"
            )
        if np.any(q_prime_array < 0.0):
            raise CalibrationError("q_prime values cannot be negative")
        if np.any(labels_array < 0) or np.any(labels_array >= self.number_of_classes):
            raise CalibrationError("calibration labels must be known class labels")
        if np.any(predictions_array < 0) or np.any(predictions_array >= self.number_of_classes):
            raise CalibrationError("predictions are outside the configured classes")

        residual = np.ones(row_count, dtype=bool)
        passes_ood_gate = np.floor(q_prime_array) > self.ood_limit
        row_indices = np.arange(row_count, dtype=np.int64)
        regions: list[Region] = []
        for alpha in ladder_alphas(self.alpha_resolution):
            candidate_mask = residual & passes_ood_gate
            candidate_rows = np.flatnonzero(candidate_mask)
            if candidate_rows.size == 0:
                break
            chosen = _candidate_region(
                probabilities_array,
                q_prime_array,
                labels_array,
                candidate_rows,
                alpha,
                self.number_of_classes,
            )
            if chosen is None:
                continue
            regions.append(chosen)
            thresholds = np.asarray(chosen.output_thresholds, dtype=np.float32)
            above = probabilities_array >= thresholds
            accepted = (
                residual
                & passes_ood_gate
                & (q_prime_array >= np.float32(chosen.minimum_rescaled_similarity))
                & (np.sum(above, axis=1) == 1)
                & above[row_indices, predictions_array]
            )
            residual[accepted] = False
        return tuple(regions)


def calibrate_nested_regions(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    q_prime: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    number_of_classes: int,
    alpha_resolution: float,
    ood_limit: int = 0,
    predictions: Sequence[int] | np.ndarray | None = None,
) -> tuple[Region, ...]:
    return NestedCalibrator(number_of_classes, alpha_resolution, ood_limit).fit(
        probabilities, q_prime, labels, predictions
    )
