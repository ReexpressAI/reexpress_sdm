# Copyright Reexpress AI, Inc. All rights reserved.
"""Training capacity checks shared by the CLI and the Python training API.

This budgets generated model metadata, not feature storage or available RAM.
Keep the coefficients aligned with SDMKit's SDMTrainingCapacity. The final
writer separately checks exact encoded bytes, including caller-added metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping

from . import artifact
from .errors import ArtifactValidationError
from .math import ladder_alphas


@dataclass(frozen=True)
class TrainingCapacity:
    estimated_manifest_bytes: int
    input_vector_bytes: int
    support_elements: int
    weight_elements: int


def estimate_training_capacity(
    *, training_count: int, calibration_count: int, input_dimension: int,
    exemplar_dimension: int, number_of_classes: int, epochs: int,
    iterations: int, alpha_resolution: float, shuffle: bool,
    encoded_id_bytes: int, additional_metadata_bytes: int = 0,
) -> TrainingCapacity:
    """Calculate a conservative generated-manifest budget without feature arrays.

    ``encoded_id_bytes`` is the sum of ASCII-escaped, quoted JSON IDs. Counts
    describe the original input splits; the winning support uses the requested
    shuffle policy. There is deliberately no row-count or input-element cap.
    """
    values = (training_count, calibration_count, input_dimension, exemplar_dimension,
              number_of_classes, epochs, iterations)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise ValueError("training capacity counts and dimensions must be positive integers")
    if encoded_id_bytes < 0 or additional_metadata_bytes < 0:
        raise ValueError("encoded metadata sizes must be nonnegative")
    total = training_count + calibration_count
    support = total // 2 if shuffle else training_count
    calibration = total - support if shuffle else calibration_count
    weights = exemplar_dimension * input_dimension + exemplar_dimension + number_of_classes * exemplar_dimension + number_of_classes
    # Each row reserves its checksum, quoted ID, pool index and pretty-print
    # whitespace. CDF families contain at most two values per calibration row.
    # Region/history allowances cover finite numeric JSON and field names.
    estimate = (
        1024 * 1024
        + total * (112 + len(str(total - 1))) + encoded_id_bytes
        + 2 * calibration * 40
        + len(ladder_alphas(alpha_resolution)) * (512 + 40 * number_of_classes)
        + epochs * iterations * 2048 + iterations * 2048
        + additional_metadata_bytes
    )
    return TrainingCapacity(estimate, 4 * total * input_dimension,
                            support * exemplar_dimension, weights)


def validate_training_capacity(capacity: TrainingCapacity) -> None:
    """Apply the existing tensor limits and the shared manifest byte budget."""
    if capacity.support_elements > artifact.MAX_TENSOR_ELEMENTS:
        raise ArtifactValidationError("training support tensor exceeds the model safety limit")
    if capacity.weight_elements > artifact.MAX_TENSOR_ELEMENTS:
        raise ArtifactValidationError("training adaptor weights exceed the model safety limit")
    if capacity.estimated_manifest_bytes > artifact.MAX_MANIFEST_BYTES:
        raise ArtifactValidationError(
            "training pre-flight: conservative manifest budget "
            f"({capacity.estimated_manifest_bytes / 1024**2:.1f} MiB) exceeds the "
            f"{artifact.MAX_MANIFEST_BYTES / 1024**2:g} MiB model safety limit; "
            "reduce source ID lengths, data, requested history, or calibration ladder size"
        )


def check_training_capacity(
    configuration: Any, *, training_count: int, calibration_count: int,
    input_dimension: int, ids: Iterable[str], iterations: int = 1,
    shuffle: bool = False, metadata: Mapping[str, Any] | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> TrainingCapacity:
    """Check shapes first, then scan only IDs before allocating training copies."""
    arguments = dict(
        training_count=int(training_count), calibration_count=int(calibration_count),
        input_dimension=int(input_dimension), exemplar_dimension=int(configuration.exemplar_dimension),
        number_of_classes=int(configuration.number_of_classes), epochs=int(configuration.epochs),
        iterations=int(iterations), alpha_resolution=configuration.alpha_resolution, shuffle=shuffle,
    )
    # Reject impossible shapes or even the minimum manifest budget before IDs.
    validate_training_capacity(estimate_training_capacity(**arguments, encoded_id_bytes=0))
    encoded_id_bytes = 0
    for index, value in enumerate(ids):
        if check_cancelled is not None and index % 4096 == 0:
            check_cancelled()
        encoded_id_bytes += len(json.dumps(value, ensure_ascii=True, allow_nan=False))
    # Includes arbitrary-length representation strings/class names supplied by
    # the caller; the fixed allowance covers the ordinary generated structure.
    extra = len(json.dumps(metadata or {}, ensure_ascii=True, indent=2, allow_nan=False))
    result = estimate_training_capacity(**arguments, encoded_id_bytes=encoded_id_bytes,
                                        additional_metadata_bytes=extra)
    validate_training_capacity(result)
    if check_cancelled is not None:
        check_cancelled()
    return result


def physical_memory_bytes() -> int | None:
    """Best-effort installed RAM, using only platform/standard-library APIs."""
    try:
        pages, page_size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages * page_size)
    except (OSError, ValueError, AttributeError):
        pass
    if sys.platform == "darwin":
        try:
            result = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"],
                                    capture_output=True, text=True, check=True, timeout=2)
            value = int(result.stdout.strip())
            return value if value > 0 else None
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return None


def memory_advisory(input_vector_bytes: int, installed_memory_bytes: int | None) -> dict[str, Any]:
    """A lower-bound advisory, never a permission or a peak-memory promise."""
    result: dict[str, Any] = {
        "type": "trainingMemoryAdvisory",
        "inputVectorBytes": input_vector_bytes,
        "installedMemoryBytes": installed_memory_bytes,
        "message": (
            "Float32 input vectors only; training also needs pooled/shuffled and normalized "
            "copies, projected vectors, matching workspace, and metadata. JSONL retains "
            "decoded feature lists; .sdmdataset avoids that overhead. Installed RAM is not "
            "available RAM or GPU memory. This is not a peak-memory estimate or a guarantee of fit."
        ),
    }
    if installed_memory_bytes is not None and input_vector_bytes >= installed_memory_bytes // 2:
        result["warning"] = "Input vectors alone occupy at least half of installed RAM; substantial memory pressure is likely."
    return result
