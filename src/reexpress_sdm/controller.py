# Copyright Reexpress AI, Inc. All rights reserved.
"""High-level SDK facade suitable for application and agent integrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .metrics import Evaluator
from .model import SDMModel
from .monitoring import distribution_summary
from .policy import SelectionPolicy
from .types import ControlDecision, SDMArtifact, SDMScore


class SDMController:
    def __init__(self, model: SDMModel, policy: SelectionPolicy | None = None):
        self.model = model
        self.policy = policy or SelectionPolicy()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        policy: SelectionPolicy | None = None,
        verify_checksums: bool = True,
        backend: str = "torch",
        device: str = "auto",
        query_batch_size: int = 256,
        support_tile_size: int = 16384,
    ) -> "SDMController":
        return cls(
            SDMModel.load(path, verify_checksums=verify_checksums, backend=backend, device=device,
                          query_batch_size=query_batch_size, support_tile_size=support_tile_size),
            policy=policy,
        )

    def score(
        self,
        embeddings: Sequence[Sequence[float]] | np.ndarray,
        *,
        ids: Sequence[str | None] | None = None,
        representation_fingerprint: str | None = None,
        identity_support_indices: Sequence[int | None] | None = None,
        nearest_exemplars: int = 25,
    ) -> tuple[SDMScore, ...]:
        return self.model.score(
            embeddings,
            ids=ids,
            representation_fingerprint=representation_fingerprint,
            identity_support_indices=identity_support_indices,
            nearest_exemplars=nearest_exemplars,
        )

    def decide(
        self,
        scores: Sequence[SDMScore],
        *,
        policy: SelectionPolicy | None = None,
    ) -> tuple[ControlDecision, ...]:
        return (policy or self.policy).decide_many(scores)

    def evaluate(self, scores: Sequence[SDMScore], labels: Sequence[int]) -> dict[str, Any]:
        return Evaluator(
            self.model.number_of_classes,
            alphas=[region.alpha for region in self.model.regions],
        ).evaluate(scores, labels)

    def summarize_distribution(
        self, scores: Sequence[SDMScore], *, histogram_bins: int = 10
    ) -> dict[str, Any]:
        return distribution_summary(
            scores,
            number_of_classes=self.model.number_of_classes,
            histogram_bins=histogram_bins,
        )

    def model_information(self) -> dict[str, Any]:
        return _artifact_information(self.model.artifact)


def _artifact_information(artifact: SDMArtifact) -> dict[str, Any]:
    """Describe a validated artifact without constructing a compute runtime."""
    return {
        "schemaVersion": artifact.manifest["schemaVersion"],
        "modelID": artifact.model_id,
        "createdAt": artifact.manifest["createdAt"],
        "producer": artifact.manifest["producer"],
        "configuration": artifact.manifest["configuration"],
        "normalization": artifact.manifest["normalization"],
        "representation": artifact.manifest["representation"],
        "regions": [region.to_manifest_dict() for region in artifact.regions],
        "supportCount": len(artifact.support_records),
    }
