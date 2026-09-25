# Copyright Reexpress AI, Inc. All rights reserved.
"""Explicit control decisions built on immutable SDM scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .errors import PolicyValidationError
from .types import ControlDecision, EstimatorKind, SDMScore


@dataclass(frozen=True)
class SelectionPolicy:
    minimum_alpha: float = 0.95
    estimator: EstimatorKind = EstimatorKind.CENTROID
    accepted_action: str = "allow"
    rejected_action: str = "review"

    def __post_init__(self) -> None:
        if not 0.5 < self.minimum_alpha < 1.0:
            raise PolicyValidationError("minimum_alpha must be in (0.5, 1)")
        if not isinstance(self.estimator, EstimatorKind):
            try:
                object.__setattr__(self, "estimator", EstimatorKind(self.estimator))
            except ValueError as error:
                raise PolicyValidationError("estimator must be centroid or lower") from error
        if self.accepted_action not in {"allow", "review", "reject"}:
            raise PolicyValidationError("accepted_action is invalid")
        if self.rejected_action not in {"allow", "review", "reject"}:
            raise PolicyValidationError("rejected_action is invalid")

    def decide(self, score: SDMScore) -> ControlDecision:
        observed = (
            score.centroid_region_alpha
            if self.estimator == EstimatorKind.CENTROID
            else score.lower_region_alpha
        )
        if score.is_ood:
            accepted = False
            reason = "zero-similarity"
        elif observed == 0.0:
            accepted = False
            reason = "no-calibrated-region"
        elif observed < self.minimum_alpha:
            accepted = False
            reason = "below-required-alpha"
        else:
            accepted = True
            reason = "accepted"
        return ControlDecision(
            model_id=score.model_id,
            id=score.id,
            action=self.accepted_action if accepted else self.rejected_action,
            required_alpha=self.minimum_alpha,
            observed_alpha=observed,
            estimator=self.estimator,
            reason=reason,
        )

    def decide_many(self, scores: Sequence[SDMScore]) -> tuple[ControlDecision, ...]:
        return tuple(self.decide(score) for score in scores)

