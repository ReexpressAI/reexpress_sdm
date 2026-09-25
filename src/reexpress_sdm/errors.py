# Copyright Reexpress AI, Inc. All rights reserved.
"""Typed errors exposed by the SDK."""


class SDMError(Exception):
    """Base class for expected SDK failures."""


class ArtifactValidationError(SDMError):
    """Raised when a model artifact violates the v1 contract."""


class DimensionMismatchError(SDMError):
    """Raised when an input has the wrong vector or class dimension."""


class RepresentationMismatchError(SDMError):
    """Raised when an input and model use different representation spaces."""


class CalibrationError(SDMError):
    """Raised when calibration inputs are invalid."""


class DatasetValidationError(SDMError):
    """Raised when an interchange dataset is malformed."""


class PolicyValidationError(SDMError):
    """Raised when a control policy is invalid."""

