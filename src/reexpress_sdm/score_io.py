# Copyright Reexpress AI, Inc. All rights reserved.
"""Portable scored documents, shared by Python JSONL exports and macOS imports.

A full row keeps the selected source features and content beside SDM diagnostics.
It deliberately contains no app-local dataset UUID or split assignment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .dataset import Dataset
from .bundle import _source_fingerprint
from .errors import DatasetValidationError, DimensionMismatchError
from .model import SDMModel
from .types import SDMScore

SCORE_SCHEMA_VERSION = 1
MATCHING_SEMANTICS_VERSION = 2


def _source_fields(row: Mapping[str, Any], *, composition: str) -> dict[str, Any]:
    if composition not in {"auto", "embedding", "attributes", "embedding+attributes"}:
        raise DatasetValidationError("unsupported feature composition")
    result = dict(row)
    # An explicitly deselected feature must not be reintroduced by macOS's
    # normal embedding-then-attributes composition when this row is uploaded.
    if composition == "embedding":
        result.pop("attributes", None)
    elif composition == "attributes":
        result.pop("embedding", None)
    return result


def _feature_array(value: Any, name: str) -> np.ndarray:
    """Accept JSON arrays and mapped bundle row views with the same validation."""
    if not isinstance(value, (list, np.ndarray)):
        raise DatasetValidationError(f"{name} must be a nonempty finite numeric array")
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf":
        raise DatasetValidationError(f"{name} must be a nonempty finite numeric array")
    # NumPy would coerce mixed bool/number lists to numbers; reject those before
    # conversion just as the original JSONL scoring path does.
    if isinstance(value, list) and any(isinstance(item, (bool, np.bool_)) for item in value):
        raise DatasetValidationError(f"{name} must be a nonempty finite numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        result = raw.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise DatasetValidationError(f"{name} must be a nonempty finite numeric array")
    return result


def _json_array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def score_to_document(
    score: SDMScore, source: Mapping[str, Any], *, representation_fingerprint: str,
    identity_support_index: int | None = None, excluded_support_id: str | None = None, composition: str = "auto",
) -> dict[str, Any]:
    """Attach a complete SDM score to its source document for JSONL upload."""
    identifier = source.get("id")
    if not isinstance(identifier, str) or not identifier or score.id != identifier:
        raise DatasetValidationError("a scored document needs its original nonempty string id")
    if not isinstance(representation_fingerprint, str) or not representation_fingerprint:
        raise DatasetValidationError("a scored document needs a representation fingerprint")
    label = source.get("label", -1)
    if isinstance(label, bool) or not isinstance(label, int) or label < -1 and label != -99:
        raise DatasetValidationError("a scored document needs an integer class label, -1, or -99")
    document = source.get("document")
    if document is not None and not isinstance(document, str):
        raise DatasetValidationError("document must be a string or null")
    if identity_support_index is not None and (
        isinstance(identity_support_index, bool) or not isinstance(identity_support_index, int)
        or identity_support_index < 0
    ):
        raise DatasetValidationError("identity_support_index must be a nonnegative integer or None")
    result = _source_fields(source, composition=composition)
    # Retired diagnostics still belong to scores, not source metadata. Drop
    # them when rescoring a row, along with any previous exemplar list.
    for key in ("isOOD", "floorQPrime", "floorQPrimeLower", "nearestSupportMatches"):
        result.pop(key, None)
    result.update(score.to_dict(detail="full"))
    result.update({
        "label": label,
        "scoreSchemaVersion": SCORE_SCHEMA_VERSION,
        "matchingSemanticsVersion": MATCHING_SEMANTICS_VERSION,
        "representationFingerprint": representation_fingerprint,
    })
    # Incoming score fields are replaced, never mistaken for identity evidence.
    result.pop("datasetID", None)
    result.pop("split", None)
    result.pop("excludedSupportIndex", None)
    result.pop("excludedSupportID", None)
    if identity_support_index is not None:
        if not isinstance(excluded_support_id, str) or not excluded_support_id:
            raise DatasetValidationError("an explicit support exclusion requires its support ID")
        result["excludedSupportIndex"] = identity_support_index
        result["excludedSupportID"] = excluded_support_id
    if document is not None:
        result["document"] = document
    else:
        result.pop("document", None)
    return result


def _validated_support_identity(model: SDMModel, vector: Sequence[float], label: int, index: int | None) -> str | None:
    if index is None:
        return None
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(model.artifact.support_records):
        raise DatasetValidationError("identity_support_index is outside the model support")
    support = model.artifact.support_records[index]
    if label != support.label:
        raise DatasetValidationError("the explicitly excluded support row has a different label")
    _, exemplar = model.transform([vector])
    expected = np.asarray(model.artifact.support_vectors[index], dtype=np.float32)
    tolerance = np.float32(2e-4) * np.maximum(1, np.maximum(np.abs(exemplar[0]), np.abs(expected)))
    if not np.all(np.abs(exemplar[0] - expected) <= tolerance):
        raise DatasetValidationError("the source does not project to the explicitly excluded support row")
    return support.id


def score_document(
    model: SDMModel, document: Mapping[str, Any], *,
    representation_fingerprint: str | None = None, composition: str = "auto",
    identity_support_index: int | None = None,
    nearest_exemplars: int = 25,
) -> dict[str, Any]:
    """Score one source record and return an uploadable full JSON object.

    Matching excludes a support row only when the caller explicitly supplies
    its index. A user-facing document ID never establishes support identity.
    """
    source = _source_fields(document, composition=composition)
    selected = composition
    if selected == "auto":
        selected = "embedding+attributes" if "embedding" in source and "attributes" in source else (
            "embedding" if "embedding" in source else "attributes"
        )
    keys = ("embedding", "attributes") if selected == "embedding+attributes" else (selected,)
    vector = np.concatenate([_feature_array(source.get(key), key) for key in keys])
    source_fingerprint = _source_fingerprint(source)
    fingerprint = representation_fingerprint or source_fingerprint or model.representation_fingerprint
    if source_fingerprint is not None and source_fingerprint != fingerprint:
        raise DatasetValidationError("the source and requested representation fingerprints differ")
    label = source.get("label", -1)
    if isinstance(label, bool) or not isinstance(label, int) or label >= model.number_of_classes:
        raise DatasetValidationError("the source label is outside the model's classes")
    if identity_support_index is not None and (
        isinstance(identity_support_index, bool) or not isinstance(identity_support_index, int)
        or not 0 <= identity_support_index < len(model.artifact.support_records)
    ):
        raise DatasetValidationError("identity_support_index is outside the model support")
    excluded_id = _validated_support_identity(model, vector, label, identity_support_index)
    score = model.score([vector], ids=[source.get("id")], representation_fingerprint=fingerprint,
                        identity_support_indices=[identity_support_index], nearest_exemplars=nearest_exemplars)[0]
    result = score_to_document(score, source, representation_fingerprint=fingerprint,
                               identity_support_index=identity_support_index, excluded_support_id=excluded_id, composition=composition)
    # The single-document API promises a JSON object. The batch iterator keeps
    # mapped views until its writer consumes each row to avoid eager expansion.
    for key in ("embedding", "attributes"):
        if isinstance(result.get(key), np.ndarray):
            result[key] = result[key].tolist()
    return result


def score_dataset_rows(
    model: SDMModel, dataset: Dataset, *, composition: str | None = None, batch_size: int = 256,
    identity_support_indices: Sequence[int | None] | None = None,
    nearest_exemplars: int = 25,
) -> Iterator[dict[str, Any]]:
    """Score bounded batches and yield complete, uploadable source rows."""
    dataset.validate_class_names(model.configuration["classNames"])
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    identities = list(identity_support_indices) if identity_support_indices is not None else [None] * dataset.count
    if len(identities) != dataset.count:
        raise DimensionMismatchError("identity support indices must align with the dataset")
    if any(value is not None and (isinstance(value, bool) or not isinstance(value, int)
               or not 0 <= value < len(model.artifact.support_records)) for value in identities):
        raise DatasetValidationError("identity support indices must be valid indices or None")
    composition = dataset.composition if composition is None else composition
    fingerprint = dataset.representation_fingerprint or model.representation_fingerprint
    for start in range(0, dataset.count, batch_size):
        stop = min(start + batch_size, dataset.count)
        scores = model.score(dataset.vectors[start:stop], ids=dataset.ids[start:stop],
                             representation_fingerprint=fingerprint, identity_support_indices=identities[start:stop],
                             nearest_exemplars=nearest_exemplars)
        for offset, score in enumerate(scores, start=start):
            emitted = _source_fields(dataset.rows[offset], composition=composition)
            combined = [_feature_array(emitted[key], key) for key in ("embedding", "attributes") if key in emitted]
            vector = np.concatenate(combined) if combined else np.empty(0, np.float32)
            if vector.shape != dataset.vectors[offset].shape or not np.array_equal(vector, dataset.vectors[offset]):
                raise DatasetValidationError("exported source features do not match the scored dataset; use its original composition")
            excluded_id = _validated_support_identity(model, dataset.vectors[offset], dataset.labels[offset], identities[offset])
            yield score_to_document(score, dataset.rows[offset], representation_fingerprint=fingerprint,
                                    identity_support_index=identities[offset], excluded_support_id=excluded_id, composition=composition)


def write_scored_jsonl(rows: Iterable[Mapping[str, Any]], destination: str | Path) -> None:
    """Atomically replace a file only after every score row has been encoded."""
    if str(destination) == "-":
        for row in rows:
            sys.stdout.write(json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(",", ":"), default=_json_array) + "\n")
        return
    path = Path(destination)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(",", ":"), default=_json_array) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
