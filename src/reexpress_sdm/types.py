# Copyright Reexpress AI, Inc. All rights reserved.
"""Public, framework-neutral value types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


class EstimatorKind(str, Enum):
    CENTROID = "centroid"
    LOWER = "lower"


@dataclass(frozen=True)
class Region:
    alpha: float
    minimum_rescaled_similarity: float
    output_thresholds: tuple[float, ...]

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "minimumRescaledSimilarity": self.minimum_rescaled_similarity,
            "outputThresholds": list(self.output_thresholds),
        }


@dataclass(frozen=True)
class SupportRecord:
    id: str
    label: int
    predicted_label: int
    document: str | None = None
    metadata: Mapping[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "predictedLabel": self.predicted_label,
        }
        if self.document is not None:
            result["document"] = self.document
        if self.metadata is not None:
            result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True)
class SupportMatch:
    """One exact nearest exemplar, ordered by distance then support index."""

    support_index: int
    id: str
    label: int
    predicted_label: int
    squared_distance: float
    document: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "supportIndex": self.support_index,
            "id": self.id,
            "label": self.label,
            "predictedLabel": self.predicted_label,
            "squaredDistance": self.squared_distance,
        }
        if self.document is not None:
            result["document"] = self.document
        return result


@dataclass(frozen=True)
class AdapterWeights:
    projection_weight: np.ndarray
    projection_bias: np.ndarray
    classifier_weight: np.ndarray
    classifier_bias: np.ndarray


@dataclass(frozen=True)
class SDMArtifact:
    manifest: Mapping[str, Any]
    weights: AdapterWeights
    support_vectors: np.ndarray
    support_records: tuple[SupportRecord, ...]
    calibration_rows: tuple[Mapping[str, Any], ...] | None = None

    @property
    def model_id(self) -> str:
        return str(self.manifest["modelID"])

    @property
    def configuration(self) -> Mapping[str, Any]:
        return self.manifest["configuration"]

    @property
    def regions(self) -> tuple[Region, ...]:
        return tuple(
            Region(
                alpha=float(value["alpha"]),
                minimum_rescaled_similarity=float(value["minimumRescaledSimilarity"]),
                output_thresholds=tuple(float(x) for x in value["outputThresholds"]),
            )
            for value in self.manifest["regions"]
        )


@dataclass(frozen=True)
class SDMScore:
    model_id: str
    id: str | None
    z_prime: tuple[float, ...]
    prediction: int
    q: int
    d0: float
    d: float
    sdm: tuple[float, ...]
    q_prime: float
    floor_q_prime: int
    is_ood: bool
    nearest_support_index: int
    nearest_support_id: str
    effective_sample_sizes: tuple[int, ...]
    effective_sample_size_errors: tuple[float, ...]
    d_lower: float
    d_upper: float
    sdm_lower: tuple[float, ...]
    sdm_upper: tuple[float, ...]
    q_prime_lower: float
    floor_q_prime_lower: int
    centroid_region_alpha: float
    lower_region_alpha: float
    most_conservative_region_alpha: float
    nearest_support_matches: tuple[SupportMatch, ...] = ()

    @property
    def is_in_most_conservative_region(self) -> bool:
        """Whether the centroid estimate has the artifact's highest alpha."""

        return (
            self.most_conservative_region_alpha > 0.0
            and self.centroid_region_alpha == self.most_conservative_region_alpha
        )

    @property
    def is_in_most_conservative_region_lower(self) -> bool:
        """Whether the DKW-lower estimate has the artifact's highest alpha."""

        return (
            self.most_conservative_region_alpha > 0.0
            and self.lower_region_alpha == self.most_conservative_region_alpha
        )

    def to_dict(self, detail: str = "full") -> dict[str, Any]:
        result: dict[str, Any] = {
            "modelID": self.model_id,
            "id": self.id,
            "prediction": self.prediction,
            "q": self.q,
            "d0": self.d0,
            "d": self.d,
            "qPrime": self.q_prime,
            "nearestSupportIndex": self.nearest_support_index,
            "nearestSupportID": self.nearest_support_id,
            "centroidRegionAlpha": self.centroid_region_alpha,
            "lowerRegionAlpha": self.lower_region_alpha,
            "isInMostConservativeRegion": self.is_in_most_conservative_region,
            "isInMostConservativeRegionLower": self.is_in_most_conservative_region_lower,
        }
        if detail == "full":
            result.update(
                {
                    "zPrime": list(self.z_prime),
                    "sdm": list(self.sdm),
                    "cumulativeEffectiveSampleSizes": list(self.effective_sample_sizes),
                    "effectiveSampleSizeErrors": list(self.effective_sample_size_errors),
                    "dLower": self.d_lower,
                    "dUpper": self.d_upper,
                    "sdmLower": list(self.sdm_lower),
                    "sdmUpper": list(self.sdm_upper),
                    "qPrimeLower": self.q_prime_lower,
                }
            )
        elif detail != "compact":
            raise ValueError("detail must be 'full' or 'compact'")
        if self.nearest_support_matches:
            result["nearestSupportMatches"] = [match.to_dict() for match in self.nearest_support_matches]
        return result


@dataclass(frozen=True)
class ControlDecision:
    model_id: str
    id: str | None
    action: str
    required_alpha: float
    observed_alpha: float
    estimator: EstimatorKind
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "modelID": self.model_id,
            "id": self.id,
            "action": self.action,
            "requiredAlpha": self.required_alpha,
            "observedAlpha": self.observed_alpha,
            "estimator": self.estimator.value,
            "reason": self.reason,
        }


def floats(values: Sequence[float]) -> tuple[float, ...]:
    """Convert NumPy values to JSON-safe built-in floats."""

    return tuple(float(x) for x in values)
