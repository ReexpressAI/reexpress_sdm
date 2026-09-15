# Copyright Reexpress AI, Inc. All rights reserved.
"""Framework-neutral evidence linking saved models to raw input features."""
from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import DatasetValidationError

FEATURE_DIGEST_ALGORITHM = "sha256-f32le-v1"


def feature_digest(vector: Sequence[float] | np.ndarray) -> str:
    """SHA-256 of a finite raw Float32 vector in LE order, canonicalizing -0."""
    raw = np.asarray(vector)
    if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "iuf" or (
        isinstance(vector, (list, tuple)) and any(isinstance(x, (bool, np.bool_)) for x in vector)
    ):
        raise DatasetValidationError("feature digest requires a nonempty finite numeric vector")
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.array(raw, dtype="<f4", order="C", copy=True)
    if not np.isfinite(value).all():
        raise DatasetValidationError("feature digest requires finite Float32 values")
    value[value == 0] = 0  # Numerically identical signed zeros have one portable encoding.
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def build_source_data(training_vectors: np.ndarray, calibration_vectors: np.ndarray) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "featureDigestAlgorithm": FEATURE_DIGEST_ALGORITHM,
        "trainingFeatureDigests": [feature_digest(row) for row in training_vectors],
        "calibrationFeatureDigests": [feature_digest(row) for row in calibration_vectors],
    }


def validate_source_data(value: Any, *, training_count: int, calibration_count: int | None = None) -> None:
    if not isinstance(value, Mapping) or type(value.get("schemaVersion")) is not int or value["schemaVersion"] != 1:
        raise DatasetValidationError("sourceData requires schemaVersion 1")
    if value.get("featureDigestAlgorithm") != FEATURE_DIGEST_ALGORITHM:
        raise DatasetValidationError("sourceData uses an unsupported feature digest algorithm")
    for key, count in (("trainingFeatureDigests", training_count), ("calibrationFeatureDigests", calibration_count)):
        digests = value.get(key)
        if not isinstance(digests, list) or (count is not None and len(digests) != count) or any(
            not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None for item in digests
        ):
            raise DatasetValidationError(f"sourceData.{key} must contain an aligned lowercase SHA-256 per row")
