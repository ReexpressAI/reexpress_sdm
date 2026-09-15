# Copyright Reexpress AI, Inc. All rights reserved.
"""Portable SDMKit model artifact I/O."""

from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Mapping
import unicodedata
import uuid

import numpy as np

from .errors import ArtifactValidationError
from .math import _validate_alpha_resolution, ladder_alphas
from .types import AdapterWeights, SDMArtifact, SupportRecord


MANIFEST_FILE = "manifest.json"
WEIGHTS_FILE = "weights.f32"
SUPPORT_VECTORS_FILE = "support.f32"
SUPPORT_RECORDS_FILE = "support.jsonl"
CALIBRATION_FILE = "calibration.jsonl"
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_TENSOR_ELEMENTS = 2_000_000_000
MIN_JSON_INTEGER = -(1 << 63)
MAX_JSON_INTEGER = (1 << 64) - 1


def _fail(message: str) -> None:
    raise ArtifactValidationError(message)


def _require_unicode(value: str, name: str) -> str:
    """Accept Unicode scalar strings without rewriting their spelling."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _fail(f"{name} contains an invalid Unicode surrogate")
    return value


def _identity_key(value: str, name: str) -> str:
    # Swift String equality uses canonical equivalence. This is only a
    # uniqueness key; the accepted source spelling remains unchanged.
    return unicodedata.normalize("NFC", _require_unicode(value, name))


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object")
    return value


def _check_keys(value: Mapping[str, Any], name: str, required: set[str], optional: set[str] | None = None) -> None:
    optional = optional or set()
    missing = required - set(value)
    unexpected = set(value) - required - optional
    if missing:
        _fail(f"{name} is missing fields: {sorted(missing)}")
    if unexpected:
        _fail(f"{name} has unsupported fields: {sorted(unexpected)}")


def _require_int(value: Any, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{name} must be an integer")
    if not MIN_JSON_INTEGER <= value <= (1 << 63) - 1:
        _fail(f"{name} must fit a signed 64-bit schema integer")
    if minimum is not None and value < minimum:
        _fail(f"{name} must be >= {minimum}")
    return value


def _require_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be a finite number")
    try:
        converted = float(value)
    except OverflowError:
        _fail(f"{name} must be a finite number")
    if not math.isfinite(converted):
        _fail(f"{name} must be a finite number")
    return converted


def _require_f32(value: Any, name: str) -> float:
    converted = _require_finite(value, name)
    with np.errstate(over="ignore", invalid="ignore"):
        float32_value = np.float32(converted)
    if not np.isfinite(float32_value):
        _fail(f"{name} must be representable as Float32")
    return converted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cdfs(value: Any, classes: int, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != classes:
        _fail(f"{name} must contain exactly {classes} class arrays")
    result: list[list[float]] = []
    for class_index, cdf in enumerate(value):
        if not isinstance(cdf, list):
            _fail(f"{name}[{class_index}] must be an array")
        converted = [_require_f32(item, f"{name}[{class_index}]") for item in cdf]
        if any(item < 0.0 for item in converted):
            _fail(f"{name}[{class_index}] contains a negative value")
        if any(a > b for a, b in zip(converted, converted[1:])):
            _fail(f"{name}[{class_index}] must be sorted ascending")
        result.append(converted)
    return result


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a detached mutable copy of a v1 manifest."""

    required = {
        "schemaVersion",
        "modelID",
        "createdAt",
        "producer",
        "configuration",
        "normalization",
        "representation",
        "weights",
        "support",
        "distanceCDFs",
        "rescaledSimilarityCDFs",
        "regions",
    }
    _check_keys(manifest, "manifest", required, {"checksums", "metadata"})
    if _require_int(manifest["schemaVersion"], "schemaVersion") != 1:
        _fail("only schemaVersion 1 is supported")
    if not isinstance(manifest["modelID"], str) or not manifest["modelID"]:
        _fail("modelID must be a non-empty string")
    _require_unicode(manifest["modelID"], "modelID")
    if not isinstance(manifest["createdAt"], str) or not manifest["createdAt"]:
        _fail("createdAt must be a non-empty RFC-3339 string")
    _require_unicode(manifest["createdAt"], "createdAt")
    try:
        parsed_created_at = datetime.fromisoformat(manifest["createdAt"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ArtifactValidationError("createdAt must be an RFC-3339 timestamp") from error
    if "T" not in manifest["createdAt"] or parsed_created_at.tzinfo is None:
        _fail("createdAt must include a time and UTC offset")
    producer = _require_mapping(manifest["producer"], "producer")
    _check_keys(producer, "producer", {"name", "version"})
    for key in ("name", "version"):
        if not isinstance(producer.get(key), str) or not producer[key]:
            _fail(f"producer.{key} must be a non-empty string")
        _require_unicode(producer[key], f"producer.{key}")

    configuration = _require_mapping(manifest["configuration"], "configuration")
    _check_keys(
        configuration,
        "configuration",
        {
            "numberOfClasses",
            "classNames",
            "embeddingDimension",
            "exemplarDimension",
            "maxNeighbors",
            "qOffset",
            "oodLimit",
            "alphaResolution",
            "distanceMetric",
            "neighborTieBreak",
        },
    )
    classes = _require_int(configuration.get("numberOfClasses"), "numberOfClasses", 2)
    embedding_dimension = _require_int(
        configuration.get("embeddingDimension"), "embeddingDimension", 1
    )
    exemplar_dimension = _require_int(
        configuration.get("exemplarDimension"), "exemplarDimension", 1
    )
    _require_int(configuration.get("maxNeighbors"), "maxNeighbors", 1)
    q_offset = _require_f32(configuration.get("qOffset"), "qOffset")
    if np.float32(q_offset) <= np.float32(1.0):
        _fail("qOffset must be greater than one")
    _require_int(configuration.get("oodLimit"), "oodLimit", 0)
    resolution = _require_finite(configuration.get("alphaResolution"), "alphaResolution")
    try:
        _validate_alpha_resolution(resolution)
    except ValueError as error:
        raise ArtifactValidationError(str(error).replace("alpha_resolution", "alphaResolution")) from error
    if configuration.get("distanceMetric") != "squaredL2":
        _fail("distanceMetric must be 'squaredL2'")
    if configuration.get("neighborTieBreak") != "supportIndexAscending":
        _fail("neighborTieBreak must be 'supportIndexAscending'")
    class_names = configuration.get("classNames")
    if (
        not isinstance(class_names, list)
        or len(class_names) != classes
        or any(not isinstance(name, str) or not name for name in class_names)
        or len(set(class_names)) != classes
    ):
        _fail("classNames must contain one unique string per class")
    if len({_identity_key(name, "classNames") for name in class_names}) != classes:
        _fail("classNames must be unique under Unicode canonical equivalence")

    normalization = _require_mapping(manifest["normalization"], "normalization")
    _check_keys(normalization, "normalization", {"mean", "standardDeviation"})
    _require_f32(normalization.get("mean"), "normalization.mean")
    standard_deviation = _require_f32(
        normalization.get("standardDeviation"), "normalization.standardDeviation"
    )
    if np.float32(standard_deviation) <= np.float32(0.0):
        _fail("normalization.standardDeviation must be positive")

    representation = _require_mapping(manifest["representation"], "representation")
    _check_keys(
        representation,
        "representation",
        {"provider", "model", "revision", "inputTemplate", "fingerprint"},
    )
    if not isinstance(representation.get("provider"), str) or not representation["provider"]:
        _fail("representation.provider must be a non-empty string")
    if not isinstance(representation.get("fingerprint"), str) or not representation["fingerprint"]:
        _fail("representation.fingerprint must be a non-empty string")
    for key in ("provider", "fingerprint"):
        _require_unicode(representation[key], f"representation.{key}")
    for key in ("model", "revision", "inputTemplate"):
        if key not in representation or representation[key] is not None and not isinstance(
            representation[key], str
        ):
            _fail(f"representation.{key} must be a string or null")
        if representation[key] is not None:
            _require_unicode(representation[key], f"representation.{key}")

    expected_weight_count = (
        exemplar_dimension * embedding_dimension
        + exemplar_dimension
        + classes * exemplar_dimension
        + classes
    )
    if expected_weight_count > MAX_TENSOR_ELEMENTS:
        _fail("weight tensor exceeds the loader safety limit")
    weights = _require_mapping(manifest["weights"], "weights")
    _check_keys(weights, "weights", {"file", "elementCount"})
    if weights.get("file") != WEIGHTS_FILE:
        _fail(f"weights.file must be '{WEIGHTS_FILE}'")
    if _require_int(weights.get("elementCount"), "weights.elementCount", 1) != expected_weight_count:
        _fail("weights.elementCount does not match the configured dimensions")

    support = _require_mapping(manifest["support"], "support")
    _check_keys(support, "support", {"vectorsFile", "recordsFile", "count"})
    if support.get("vectorsFile") != SUPPORT_VECTORS_FILE:
        _fail(f"support.vectorsFile must be '{SUPPORT_VECTORS_FILE}'")
    if support.get("recordsFile") != SUPPORT_RECORDS_FILE:
        _fail(f"support.recordsFile must be '{SUPPORT_RECORDS_FILE}'")
    support_count = _require_int(support.get("count"), "support.count", 1)
    if support_count * exemplar_dimension > MAX_TENSOR_ELEMENTS:
        _fail("support tensor exceeds the loader safety limit")

    _validate_cdfs(manifest["distanceCDFs"], classes, "distanceCDFs")
    similarity_cdfs = _validate_cdfs(manifest["rescaledSimilarityCDFs"], classes, "rescaledSimilarityCDFs")
    neighbor_limit = np.float32(min(configuration["maxNeighbors"], support_count))
    if any(np.float32(item) > neighbor_limit for cdf in similarity_cdfs for item in cdf):
        _fail("rescaledSimilarityCDFs exceed the available neighbor count")

    regions = manifest["regions"]
    if not isinstance(regions, list):
        _fail("regions must be an array")
    prior_alpha = math.inf
    seen_alphas: set[float] = set()
    configured_alphas = ladder_alphas(resolution)
    for index, region_value in enumerate(regions):
        region = _require_mapping(region_value, f"regions[{index}]")
        _check_keys(
            region,
            f"regions[{index}]",
            {"alpha", "minimumRescaledSimilarity", "outputThresholds"},
        )
        alpha = _require_finite(region.get("alpha"), f"regions[{index}].alpha")
        if not 0.5 < alpha < 1.0:
            _fail(f"regions[{index}].alpha must be in (0.5, 1)")
        if not any(abs(alpha - configured) <= 1.0e-9 for configured in configured_alphas):
            _fail(f"regions[{index}].alpha is not on the configured alpha ladder")
        if alpha >= prior_alpha or alpha in seen_alphas:
            _fail("regions must have unique alpha values sorted descending")
        prior_alpha = alpha
        seen_alphas.add(alpha)
        minimum = _require_f32(
            region.get("minimumRescaledSimilarity"),
            f"regions[{index}].minimumRescaledSimilarity",
        )
        if minimum < 0.0:
            _fail("minimumRescaledSimilarity cannot be negative")
        if np.float32(minimum) > neighbor_limit:
            _fail("minimumRescaledSimilarity exceeds the available neighbor count")
        thresholds = region.get("outputThresholds")
        if not isinstance(thresholds, list) or len(thresholds) != classes:
            _fail(f"regions[{index}].outputThresholds must have {classes} values")
        converted = [
            _require_f32(value, f"regions[{index}].outputThresholds")
            for value in thresholds
        ]
        alpha_gate = np.float32(alpha)
        threshold_gates = np.asarray(converted, dtype=np.float32)
        if np.any(threshold_gates < alpha_gate) or np.any(
            threshold_gates > np.float32(1.0)
        ):
            _fail("region output thresholds must be in [alpha, 1]")

    checksums = manifest.get("checksums", {})
    if not isinstance(checksums, Mapping):
        _fail("checksums must be an object")
    if "checksums" in manifest:
        required_checksums = {WEIGHTS_FILE, SUPPORT_VECTORS_FILE, SUPPORT_RECORDS_FILE}
        checksum_names = set(checksums)
        if checksum_names not in (required_checksums, required_checksums | {CALIBRATION_FILE}):
            _fail("checksums must cover every required payload and optionally calibration.jsonl")
    for filename, digest in checksums.items():
        if filename not in {
            WEIGHTS_FILE,
            SUPPORT_VECTORS_FILE,
            SUPPORT_RECORDS_FILE,
            CALIBRATION_FILE,
        }:
            _fail(f"unsupported checksum path: {filename}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            _fail(f"invalid SHA-256 checksum for {filename}")
    if "metadata" in manifest:
        if not isinstance(manifest["metadata"], Mapping):
            _fail("metadata must be an object")
        _validate_json_value(manifest["metadata"], "metadata")
        if "trainingRun" in manifest["metadata"]:
            from .training_metadata import validate_training_run
            try:
                validate_training_run(manifest["metadata"]["trainingRun"])
            except ValueError as error:
                _fail(str(error))
    result = copy.deepcopy(dict(manifest))
    if "metadata" in manifest:
        result["metadata"] = _copy_json_value(manifest["metadata"])
    return result


def _decode_model_json(source: str) -> Any:
    """Decode the shared model JSON domain before object keys can collapse."""
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        seen: set[str] = set()
        for key, value in items:
            canonical = _identity_key(key, "model JSON object key")
            if canonical in seen:
                _fail("model JSON contains duplicate or canonically equivalent object keys")
            seen.add(canonical)
            result[key] = value
        return result

    def integer(token: str) -> int:
        value = int(token)
        if not MIN_JSON_INTEGER <= value <= MAX_JSON_INTEGER:
            _fail("model JSON integer is outside the shared Int64/UInt64 range")
        return value

    def constant(token: str) -> None:
        _fail(f"model JSON contains a non-finite number: {token}")

    return json.loads(source, object_pairs_hook=pairs, parse_int=integer, parse_constant=constant)


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        _fail(f"missing or invalid file: {path.name}")
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        _fail("manifest exceeds the loader safety limit")
    try:
        value = _decode_model_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"unable to read manifest: {error}") from error
    return _require_mapping(value, "manifest")


def _read_exact_f32(path: Path, count: int) -> np.ndarray:
    if not path.is_file() or path.is_symlink():
        _fail(f"missing or invalid file: {path.name}")
    expected_bytes = count * 4
    if path.stat().st_size != expected_bytes:
        _fail(f"{path.name} has {path.stat().st_size} bytes; expected {expected_bytes}")
    values = np.fromfile(path, dtype="<f4", count=count)
    if values.size != count or not np.all(np.isfinite(values)):
        _fail(f"{path.name} is truncated or contains a non-finite value")
    return values.astype(np.float32, copy=False)


def _parse_calibration_row(
    value: Any,
    *,
    line_number: int,
    classes: int,
    seen_ids: set[str],
    neighbor_limit: int,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"calibration record {line_number} must be an object")
    required = {"id", "label", "prediction", "sdm", "qPrime"}
    optional = {"q", "d0", "d", "zPrime"}
    _check_keys(value, f"calibration record {line_number}", required, optional)
    identifier = value["id"]
    if not isinstance(identifier, str) or not identifier:
        _fail(f"calibration record {line_number} has a missing or duplicate id")
    identity = _identity_key(identifier, f"calibration record {line_number} id")
    if identity in seen_ids:
        _fail(f"calibration record {line_number} has a duplicate or canonically equivalent id")
    seen_ids.add(identity)
    label = _require_int(value["label"], f"calibration line {line_number} label")
    prediction = _require_int(
        value["prediction"], f"calibration line {line_number} prediction"
    )
    if not 0 <= label < classes or not 0 <= prediction < classes:
        _fail(f"calibration record {line_number} has an out-of-range class")
    probabilities = value["sdm"]
    if not isinstance(probabilities, list) or len(probabilities) != classes:
        _fail(f"calibration record {line_number} sdm must have {classes} values")
    probability_values = [
        _require_f32(item, f"calibration line {line_number} sdm")
        for item in probabilities
    ]
    probability_values_f32 = np.asarray(probability_values, dtype=np.float32)
    if any(item < 0.0 or item > 1.0 for item in probability_values) or abs(
        float(np.sum(probability_values_f32, dtype=np.float32) - np.float32(1.0))
    ) > 1.0e-5:
        _fail(f"calibration record {line_number} sdm is not categorical")
    q_prime = float(np.float32(_require_f32(value["qPrime"], f"calibration line {line_number} qPrime")))
    if not 0.0 <= q_prime <= float(np.float32(neighbor_limit)):
        _fail(f"calibration record {line_number} qPrime is outside the available neighbor count")
    if "q" in value:
        q = float(np.float32(_require_f32(value["q"], f"calibration line {line_number} q")))
        if not 0.0 <= q <= float(np.float32(neighbor_limit)) or not q.is_integer():
            _fail(f"calibration record {line_number} q must be a count within the available neighbors")
        if q_prime > q:
            _fail(f"calibration record {line_number} qPrime exceeds q")
    if "d0" in value:
        d0 = _require_f32(value["d0"], f"calibration line {line_number} d0")
        if d0 < 0.0:
            _fail(f"calibration record {line_number} d0 cannot be negative")
    if "d" in value:
        distance = _require_f32(value["d"], f"calibration line {line_number} d")
        if not 0.0 <= distance <= 1.0:
            _fail(f"calibration record {line_number} d must be in [0, 1]")
    if "zPrime" in value:
        logits = value["zPrime"]
        if not isinstance(logits, list) or len(logits) != classes:
            _fail(f"calibration record {line_number} zPrime must have {classes} values")
        with np.errstate(over="ignore", invalid="ignore"):
            logits_array = np.asarray(
                [
                    _require_f32(item, f"calibration line {line_number} zPrime")
                    for item in logits
                ],
                dtype=np.float32,
            )
        if not np.all(np.isfinite(logits_array)):
            _fail(f"calibration record {line_number} zPrime exceeds Float32 range")
        if int(np.argmax(logits_array)) != prediction:
            _fail(
                f"calibration record {line_number} prediction must match zPrime argmax"
            )
    return copy.deepcopy(dict(value))


def _validate_json_value(
    value: Any,
    name: str,
    *,
    ancestors: set[int] | None = None,
    depth: int = 0,
) -> None:
    """Validate the JSON value subset shared with Foundation Codable."""

    if depth > 64:
        _fail(f"{name} exceeds the maximum metadata nesting depth")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        _require_unicode(value, name)
        return
    if isinstance(value, int):
        if not MIN_JSON_INTEGER <= value <= MAX_JSON_INTEGER:
            _fail(f"{name} contains an integer outside the shared Int64/UInt64 range")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{name} contains a non-finite number")
        return
    if not isinstance(value, (list, Mapping)):
        _fail(f"{name} contains a value that is not JSON-safe")

    ancestors = ancestors if ancestors is not None else set()
    identity = id(value)
    if identity in ancestors:
        _fail(f"{name} contains a circular value")
    ancestors.add(identity)
    try:
        if isinstance(value, list):
            for index, item in enumerate(value):
                _validate_json_value(
                    item,
                    f"{name}[{index}]",
                    ancestors=ancestors,
                    depth=depth + 1,
                )
        else:
            seen_keys: set[str] = set()
            for key, item in value.items():
                if not isinstance(key, str):
                    _fail(f"{name} contains a non-string object key")
                canonical = _identity_key(key, f"{name} object key")
                if canonical in seen_keys:
                    _fail(f"{name} contains canonically equivalent object keys")
                seen_keys.add(canonical)
                _validate_json_value(
                    item,
                    f"{name}.{key}",
                    ancestors=ancestors,
                    depth=depth + 1,
                )
    finally:
        ancestors.remove(identity)


def _copy_json_value(value: Any) -> Any:
    """Detach a value already accepted by :func:`_validate_json_value`."""

    if isinstance(value, list):
        return [_copy_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _copy_json_value(item) for key, item in value.items()}
    return value


def _validate_support_records(
    records: Any,
    *,
    classes: int,
    expected_count: int,
) -> tuple[SupportRecord, ...]:
    if not isinstance(records, (list, tuple)):
        _fail("support_records must be a list or tuple")
    if len(records) != expected_count:
        _fail(f"support_records has {len(records)} rows; expected {expected_count}")
    validated: list[SupportRecord] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        name = f"support_records[{index}]"
        if not isinstance(record, SupportRecord):
            _fail(f"{name} must be a SupportRecord")
        if not isinstance(record.id, str) or not record.id:
            _fail(f"{name}.id must be nonempty and unique")
        identity = _identity_key(record.id, f"{name}.id")
        if identity in seen_ids:
            _fail(f"{name}.id must be unique under Unicode canonical equivalence")
        seen_ids.add(identity)
        label = _require_int(record.label, f"{name}.label")
        predicted = _require_int(record.predicted_label, f"{name}.predicted_label")
        if label != -99 and not 0 <= label < classes:
            _fail(f"{name}.label is outside the allowed classes")
        if not 0 <= predicted < classes:
            _fail(f"{name}.predicted_label is outside the configured classes")
        if record.document is not None and not isinstance(record.document, str):
            _fail(f"{name}.document must be a string or null")
        if record.document is not None:
            _require_unicode(record.document, f"{name}.document")
        if record.metadata is not None:
            if not isinstance(record.metadata, Mapping):
                _fail(f"{name}.metadata must be an object or null")
            _validate_json_value(record.metadata, f"{name}.metadata")
        metadata = (
            _copy_json_value(record.metadata) if record.metadata is not None else None
        )
        validated.append(
            SupportRecord(
                id=record.id,
                label=label,
                predicted_label=predicted,
                document=record.document,
                metadata=metadata,
            )
        )
    return tuple(validated)


def _validated_in_memory_tensor(
    value: Any, shape: tuple[int, ...], name: str
) -> np.ndarray:
    try:
        raw_value = np.asarray(value)
    except ValueError as error:
        raise ArtifactValidationError(f"{name} must be a rectangular tensor") from error
    if raw_value.shape != shape:
        _fail(f"{name} shape {raw_value.shape} does not match expected {shape}")
    if raw_value.dtype.kind not in "iuf":
        _fail(f"{name} must contain numeric values, excluding booleans and strings")
    with np.errstate(over="ignore", invalid="ignore"):
        float32_value = raw_value.astype(np.float32, copy=False)
    if not np.all(np.isfinite(float32_value)):
        _fail(f"{name} must contain finite Float32 values")
    owned_value = np.array(float32_value, dtype=np.float32, order="C", copy=True)
    owned_value.setflags(write=False)
    return owned_value


def validate_artifact(artifact: SDMArtifact) -> SDMArtifact:
    """Validate an in-memory artifact and return a detached validated view.

    Unlike :func:`load_artifact`, this performs no filesystem or checksum I/O.
    It is used at every public in-memory construction boundary so callers
    cannot bypass support-record or cached-calibration invariants.
    """

    if not isinstance(artifact, SDMArtifact):
        _fail("artifact must be an SDMArtifact")
    manifest = validate_manifest(artifact.manifest)
    configuration = manifest["configuration"]
    classes = int(configuration["numberOfClasses"])
    embedding_dimension = int(configuration["embeddingDimension"])
    exemplar_dimension = int(configuration["exemplarDimension"])
    support_count = int(manifest["support"]["count"])
    if not isinstance(artifact.weights, AdapterWeights):
        _fail("weights must be AdapterWeights")

    projection_weight = _validated_in_memory_tensor(
        artifact.weights.projection_weight,
        (exemplar_dimension, embedding_dimension),
        "weights.projection_weight",
    )
    projection_bias = _validated_in_memory_tensor(
        artifact.weights.projection_bias,
        (exemplar_dimension,),
        "weights.projection_bias",
    )
    classifier_weight = _validated_in_memory_tensor(
        artifact.weights.classifier_weight,
        (classes, exemplar_dimension),
        "weights.classifier_weight",
    )
    classifier_bias = _validated_in_memory_tensor(
        artifact.weights.classifier_bias,
        (classes,),
        "weights.classifier_bias",
    )
    support_vectors = _validated_in_memory_tensor(
        artifact.support_vectors,
        (support_count, exemplar_dimension),
        "support_vectors",
    )
    support_records = _validate_support_records(
        artifact.support_records,
        classes=classes,
        expected_count=support_count,
    )

    calibration_rows: tuple[Mapping[str, Any], ...] | None = None
    if artifact.calibration_rows is not None:
        if not isinstance(artifact.calibration_rows, (list, tuple)):
            _fail("calibration_rows must be a list, tuple, or null")
        parsed_rows: list[Mapping[str, Any]] = []
        seen_calibration_ids: set[str] = set()
        for line_number, row in enumerate(artifact.calibration_rows, start=1):
            parsed_rows.append(
                _parse_calibration_row(
                    row,
                    line_number=line_number,
                    classes=classes,
                    seen_ids=seen_calibration_ids,
                    neighbor_limit=min(configuration["maxNeighbors"], support_count),
                )
            )
        calibration_rows = tuple(parsed_rows)

    source_data = manifest.get("metadata", {}).get("sourceData")
    if source_data is not None:
        from .source_data import validate_source_data
        from .errors import DatasetValidationError
        try:
            validate_source_data(source_data, training_count=support_count,
                                 calibration_count=len(calibration_rows or ()))
        except DatasetValidationError as error:
            _fail(str(error))

    return SDMArtifact(
        manifest=manifest,
        weights=AdapterWeights(
            projection_weight=projection_weight,
            projection_bias=projection_bias,
            classifier_weight=classifier_weight,
            classifier_bias=classifier_bias,
        ),
        support_vectors=support_vectors,
        support_records=support_records,
        calibration_rows=calibration_rows,
    )


def load_artifact(path: str | os.PathLike[str], verify_checksums: bool = True) -> SDMArtifact:
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        _fail("artifact must be a non-symlink directory package")
    manifest = validate_manifest(_load_json(root / MANIFEST_FILE))
    configuration = manifest["configuration"]
    classes = int(configuration["numberOfClasses"])
    embedding_dimension = int(configuration["embeddingDimension"])
    exemplar_dimension = int(configuration["exemplarDimension"])
    support_count = int(manifest["support"]["count"])

    checksums = manifest.get("checksums")
    if checksums is not None:
        expected_files = {WEIGHTS_FILE, SUPPORT_VECTORS_FILE, SUPPORT_RECORDS_FILE}
        if (root / CALIBRATION_FILE).exists():
            expected_files.add(CALIBRATION_FILE)
        if set(checksums) != expected_files:
            _fail("checksums do not cover exactly the payload files present")
    for filename, expected in (checksums or {}).items():
        target = root / filename
        if not target.is_file() or target.is_symlink():
            _fail(f"checksummed file is missing or invalid: {filename}")
        if verify_checksums and _sha256(target) != expected:
            _fail(f"checksum mismatch for {filename}")

    flat_weights = _read_exact_f32(root / WEIGHTS_FILE, int(manifest["weights"]["elementCount"]))
    offset = 0
    projection_size = exemplar_dimension * embedding_dimension
    projection_weight = flat_weights[offset : offset + projection_size].reshape(
        exemplar_dimension, embedding_dimension
    )
    offset += projection_size
    projection_bias = flat_weights[offset : offset + exemplar_dimension]
    offset += exemplar_dimension
    classifier_size = classes * exemplar_dimension
    classifier_weight = flat_weights[offset : offset + classifier_size].reshape(
        classes, exemplar_dimension
    )
    offset += classifier_size
    classifier_bias = flat_weights[offset : offset + classes]

    support_vectors = _read_exact_f32(
        root / SUPPORT_VECTORS_FILE, support_count * exemplar_dimension
    ).reshape(support_count, exemplar_dimension)

    records_path = root / SUPPORT_RECORDS_FILE
    if not records_path.is_file() or records_path.is_symlink():
        _fail(f"missing or invalid file: {SUPPORT_RECORDS_FILE}")
    support_records: list[SupportRecord] = []
    seen_ids: set[str] = set()
    try:
        with records_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    _fail(f"empty support record at line {line_number}")
                value = _decode_model_json(line)
                if not isinstance(value, Mapping):
                    _fail(f"support record {line_number} must be an object")
                _check_keys(
                    value,
                    f"support record {line_number}",
                    {"id", "label", "predictedLabel"},
                    {"document", "metadata"},
                )
                record_id = value.get("id")
                if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
                    _fail(f"support record {line_number} has a missing or duplicate id")
                seen_ids.add(record_id)
                label = _require_int(value.get("label"), f"support line {line_number} label")
                predicted = _require_int(
                    value.get("predictedLabel"), f"support line {line_number} predictedLabel"
                )
                if label != -99 and not 0 <= label < classes:
                    _fail(f"support line {line_number} label is outside the allowed classes")
                if not 0 <= predicted < classes:
                    _fail(f"support line {line_number} predictedLabel is outside the classes")
                document = value.get("document")
                metadata = value.get("metadata")
                if document is not None and not isinstance(document, str):
                    _fail(f"support line {line_number} document must be a string")
                if metadata is not None and not isinstance(metadata, Mapping):
                    _fail(f"support line {line_number} metadata must be an object")
                support_records.append(
                    SupportRecord(record_id, label, predicted, document, metadata)
                )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"unable to read support records: {error}") from error
    if len(support_records) != support_count:
        _fail(f"support.jsonl has {len(support_records)} rows; expected {support_count}")

    calibration_rows: tuple[Mapping[str, Any], ...] | None = None
    calibration_path = root / CALIBRATION_FILE
    if calibration_path.exists():
        if not calibration_path.is_file() or calibration_path.is_symlink():
            _fail("calibration.jsonl is not a regular file")
        loaded_rows: list[Mapping[str, Any]] = []
        seen_calibration_ids: set[str] = set()
        try:
            with calibration_path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        _fail(f"empty calibration record at line {line_number}")
                    value = _decode_model_json(line)
                    loaded_rows.append(
                        _parse_calibration_row(
                            value,
                            line_number=line_number,
                            classes=classes,
                            seen_ids=seen_calibration_ids,
                            neighbor_limit=min(configuration["maxNeighbors"], support_count),
                        )
                    )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ArtifactValidationError(f"unable to read calibration records: {error}") from error
        calibration_rows = tuple(loaded_rows)

    weights = AdapterWeights(
        projection_weight=np.array(projection_weight, copy=True),
        projection_bias=np.array(projection_bias, copy=True),
        classifier_weight=np.array(classifier_weight, copy=True),
        classifier_bias=np.array(classifier_bias, copy=True),
    )
    return validate_artifact(
        SDMArtifact(
            manifest=manifest,
            weights=weights,
            support_vectors=support_vectors,
            support_records=tuple(support_records),
            calibration_rows=calibration_rows,
        )
    )


def _write_f32(path: Path, values: np.ndarray) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(values, dtype="<f4")
    if not np.all(np.isfinite(array)):
        _fail(f"cannot write non-finite values to {path.name}")
    array.tofile(path)


def _path_identity(path: Path) -> tuple[int, int, int] | None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino, status.st_mode


def _validate_overwrite_package(path: Path) -> None:
    """Recognize a model package without loading its discarded tensor payloads."""
    try:
        manifest = validate_manifest(_load_json(path / MANIFEST_FILE))
        expected_sizes = {
            WEIGHTS_FILE: manifest["weights"]["elementCount"] * 4,
            SUPPORT_VECTORS_FILE: (
                manifest["support"]["count"]
                * manifest["configuration"]["exemplarDimension"] * 4
            ),
            SUPPORT_RECORDS_FILE: None,
        }
        if _path_identity(path / CALIBRATION_FILE) is not None:
            expected_sizes[CALIBRATION_FILE] = None
        for filename, expected_size in expected_sizes.items():
            status = (path / filename).lstat()
            if not stat.S_ISREG(status.st_mode):
                _fail(f"missing or invalid file: {filename}")
            if expected_size is not None and status.st_size != expected_size:
                _fail(f"{filename} has {status.st_size} bytes; expected {expected_size}")
        checksums = manifest.get("checksums")
        if checksums is not None and set(checksums) != set(expected_sizes):
            _fail("checksums do not cover exactly the payload files present")
    except (ArtifactValidationError, OSError) as error:
        raise ArtifactValidationError(
            f"refusing to overwrite {path}: existing destination must have a valid "
            f"model package structure ({error})"
        ) from error


def write_artifact(
    path: str | os.PathLike[str],
    artifact: SDMArtifact,
    *,
    overwrite: bool = False,
    include_checksums: bool = True,
) -> Path:
    """Write and atomically install a v1 directory artifact.

    Overwrite requires an existing non-symlink model package with a valid
    manifest, regular payload files, and matching tensor sizes. The discarded
    payloads' contents and checksums are not reread for this structural check.
    """

    artifact = validate_artifact(artifact)
    target = Path(path)
    original_identity = _path_identity(target)
    if original_identity is not None:
        if not overwrite:
            raise FileExistsError(f"artifact already exists: {target}")
        if not stat.S_ISDIR(original_identity[2]):
            _fail(f"refusing to overwrite {target}: destination must be a non-symlink model directory")
        _validate_overwrite_package(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = copy.deepcopy(dict(artifact.manifest))
    configuration = _require_mapping(manifest.get("configuration"), "configuration")
    classes = int(configuration["numberOfClasses"])
    embedding_dimension = int(configuration["embeddingDimension"])
    exemplar_dimension = int(configuration["exemplarDimension"])
    support_count = len(artifact.support_records)

    expected_shapes = (
        (artifact.weights.projection_weight, (exemplar_dimension, embedding_dimension)),
        (artifact.weights.projection_bias, (exemplar_dimension,)),
        (artifact.weights.classifier_weight, (classes, exemplar_dimension)),
        (artifact.weights.classifier_bias, (classes,)),
        (artifact.support_vectors, (support_count, exemplar_dimension)),
    )
    for value, shape in expected_shapes:
        raw_value = np.asarray(value)
        if raw_value.shape != shape:
            _fail(f"array shape {raw_value.shape} does not match expected {shape}")
        if raw_value.dtype.kind not in "iuf":
            _fail("artifact tensors must contain numeric values, excluding booleans and strings")
        with np.errstate(over="ignore", invalid="ignore"):
            float32_value = raw_value.astype(np.float32, copy=False)
        if not np.all(np.isfinite(float32_value)):
            _fail("artifact tensors must contain finite Float32 values")

    manifest["weights"] = {
        "file": WEIGHTS_FILE,
        "elementCount": int(sum(np.asarray(value).size for value, _ in expected_shapes[:4])),
    }
    manifest["support"] = {
        "vectorsFile": SUPPORT_VECTORS_FILE,
        "recordsFile": SUPPORT_RECORDS_FILE,
        "count": support_count,
    }
    manifest = validate_manifest(manifest)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    backup: Path | None = None
    try:
        flat_weights = np.concatenate(
            [
                np.asarray(artifact.weights.projection_weight, dtype=np.float32).reshape(-1),
                np.asarray(artifact.weights.projection_bias, dtype=np.float32).reshape(-1),
                np.asarray(artifact.weights.classifier_weight, dtype=np.float32).reshape(-1),
                np.asarray(artifact.weights.classifier_bias, dtype=np.float32).reshape(-1),
            ]
        )
        _write_f32(temporary / WEIGHTS_FILE, flat_weights)
        _write_f32(temporary / SUPPORT_VECTORS_FILE, artifact.support_vectors)
        with (temporary / SUPPORT_RECORDS_FILE).open("w", encoding="utf-8", newline="\n") as stream:
            for record in artifact.support_records:
                stream.write(
                    json.dumps(
                        record.to_json_dict(),
                        ensure_ascii=True,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
                stream.write("\n")
        if artifact.calibration_rows is not None:
            with (temporary / CALIBRATION_FILE).open("w", encoding="utf-8", newline="\n") as stream:
                for row in artifact.calibration_rows:
                    stream.write(
                        json.dumps(
                            dict(row), ensure_ascii=True, allow_nan=False, separators=(",", ":")
                        )
                    )
                    stream.write("\n")

        if include_checksums:
            files = [WEIGHTS_FILE, SUPPORT_VECTORS_FILE, SUPPORT_RECORDS_FILE]
            if artifact.calibration_rows is not None:
                files.append(CALIBRATION_FILE)
            manifest["checksums"] = {filename: _sha256(temporary / filename) for filename in files}
        else:
            manifest.pop("checksums", None)
        (temporary / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_artifact(temporary, verify_checksums=True)

        if _path_identity(target) != original_identity:
            raise FileExistsError(f"artifact destination changed while writing: {target}")
        if original_identity is not None:
            backup = target.with_name(f".{target.name}.backup-{uuid.uuid4()}")
            target.rename(backup)
        temporary.rename(target)
        if backup is not None:
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
        return target
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            backup.rename(target)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
