# Copyright Reexpress AI, Inc. All rights reserved.
"""Optional, model-bound source datasets for the macOS analysis workflow."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .artifact import load_artifact
from .bundle import DatasetBundle, _digest, _source_fingerprint, iter_dataset_rows, validate_dataset_rows, write_dataset_bundle
from .dataset import Dataset
from .errors import DatasetValidationError
from .model import SDMModel
from .score_io import _feature_array, _source_fields, score_dataset_rows
from .source_data import feature_digest, validate_source_data
from .types import SDMArtifact

SOURCE_ROLES = ("original-training", "original-calibration", "selected-training", "selected-calibration")
_MODEL_FILES = ("manifest.json", "weights.f32", "support.f32", "support.jsonl")
_SCORE_FIELDS = frozenset((
    "scoreSchemaVersion", "matchingSemanticsVersion", "modelID", "datasetID", "split", "prediction", "zPrime",
    "sdm", "q", "d0", "d", "qPrime", "floorQPrime", "floorQPrimeLower", "isOOD", "nearestSupportIndex",
    "nearestSupportID", "nearestSupportMatches", "excludedSupportIndex", "excludedSupportID", "centroidRegionAlpha", "lowerRegionAlpha",
    "isInMostConservativeRegion", "isInMostConservativeRegionLower", "cumulativeEffectiveSampleSizes",
    "effectiveSampleSizeErrors", "dLower", "dUpper", "sdmLower", "sdmUpper", "qPrimeLower",
))


def model_file_digests(path: str | Path) -> dict[str, str]:
    root = Path(path)
    names = list(_MODEL_FILES)
    if (root / "calibration.jsonl").exists():
        names.append("calibration.jsonl")
    result = {}
    for name in names:
        member = root / name
        if member.is_symlink() or not member.is_file():
            raise DatasetValidationError(f"model source binding requires a regular {name} file")
        result[name] = _digest(member)
    return result


def _membership(artifact: SDMArtifact):
    """Derive canonical identities exclusively from the saved numerical model."""
    calibration = artifact.calibration_rows or ()
    selected = {
        "training": [(row.id, row.label) for row in artifact.support_records],
        "calibration": [(row.get("id"), row["label"]) for row in calibration],
    }
    identities = {}
    for split, rows in selected.items():
        for index, (identifier, label) in enumerate(rows):
            if not isinstance(identifier, str) or not identifier or identifier in identities:
                raise DatasetValidationError("model source records require distinct nonempty training/calibration IDs")
            identities[identifier] = {"selectedSplit": split, "selectedIndex": index, "label": label}
    metadata = artifact.manifest.get("metadata", {})
    source_data = metadata.get("sourceData")
    if source_data is not None:
        validate_source_data(source_data, training_count=len(selected["training"]), calibration_count=len(calibration))
        for identity in identities.values():
            identity["expectedDigest"] = source_data[identity["selectedSplit"] + "FeatureDigests"][identity["selectedIndex"]]
    membership = metadata.get("bestIterationSplits")
    original = None
    if membership is not None:
        if not isinstance(membership, Mapping):
            raise DatasetValidationError("bestIterationSplits must be an object")
        if membership.get("indexConvention") != "zero-based in original training followed by original calibration":
            raise DatasetValidationError("bestIterationSplits uses an unsupported indexConvention")
        counts = [membership.get("originalTrainingCount"), membership.get("originalCalibrationCount")]
        if any(type(count) is not int or count < 1 for count in counts):
            raise DatasetValidationError("bestIterationSplits requires positive original split counts")
        total = sum(counts)
        if total != len(identities):
            raise DatasetValidationError("bestIterationSplits original counts must equal the saved source record count")
        pool = [None] * total
        for split, rows in selected.items():
            indices = membership.get(split + "PoolIndices")
            ids = membership.get(split + "IDs")
            if not isinstance(indices, list) or len(indices) != len(rows) or ids != [row[0] for row in rows]:
                raise DatasetValidationError(f"bestIterationSplits {split} rows do not align with the saved model")
            for pool_index, (identifier, _) in zip(indices, rows):
                if type(pool_index) is not int or not 0 <= pool_index < total or pool[pool_index] is not None:
                    raise DatasetValidationError("bestIterationSplits indexes must partition the original pool")
                pool[pool_index] = identifier
                identities[identifier]["poolIndex"] = pool_index
        if any(identifier is None for identifier in pool):
            raise DatasetValidationError("bestIterationSplits indexes must cover the complete original pool")
        original = {"training": pool[:counts[0]], "calibration": pool[counts[0]:]}
    return identities, selected, original


def _raw_vector(row: Mapping[str, Any], composition: str) -> np.ndarray | None:
    keys = tuple(key for key in ("embedding", "attributes") if key in row) if composition == "auto" else (
        ("embedding", "attributes") if composition == "embedding+attributes" else (composition,)
    )
    if not any(key in row for key in ("embedding", "attributes")):
        return None
    if not keys or any(key not in row for key in keys):
        raise DatasetValidationError(f"source row {row['id']!r} lacks features required by {composition}")
    return np.concatenate([_feature_array(row[key], key) for key in keys])


def _legacy_feature_check(model: SDMModel, vector: np.ndarray, identity: Mapping[str, Any]) -> None:
    logits, projected = model.transform([vector])
    index = identity["selectedIndex"]
    if identity["selectedSplit"] == "training":
        observed, expected = projected[0], model.artifact.support_vectors[index]
    else:
        saved = model.artifact.calibration_rows[index]
        if "zPrime" not in saved:
            raise DatasetValidationError("legacy calibration rows lack logits; export --text_only or use a newly trained model")
        observed, expected = logits[0], np.asarray(saved["zPrime"], dtype=np.float32)
    tolerance = np.float32(2e-4) * np.maximum(1, np.maximum(np.abs(observed), np.abs(expected)))
    if observed.shape != expected.shape or not np.all(np.abs(observed - expected) <= tolerance):
        raise DatasetValidationError("source features do not reproduce the saved model's projected source values")


def export_source_dataset(
    model_path: str | Path,
    output_path: str | Path,
    *,
    training: str | Path | None = None,
    calibration: str | Path | None = None,
    role: str = "selected-training",
    composition: str = "auto",
    text_only: bool = False,
    with_scores: bool = False,
    device: str = "auto",
    query_batch_size: int = 256,
    support_tile_size: int = 16_384,
    overwrite: bool = False,
) -> Path:
    """Export one independently optional original or winning source split.

    Rows are ordered by saved membership, regardless of source-file order. The
    original files may be supplied independently; a selected split requires all
    its IDs among supplied inputs. Text-only data never supplies self-match
    evidence. With scores, selected training rows use their validated explicit
    support indices, including rows originally supplied in calibration.
    """
    if role not in SOURCE_ROLES:
        raise DatasetValidationError(f"role must be one of {SOURCE_ROLES}")
    if composition not in {"auto", "embedding", "attributes", "embedding+attributes"}:
        raise DatasetValidationError("unsupported feature composition")
    if text_only and with_scores:
        raise DatasetValidationError("--text_only and --with_scores cannot be combined")
    sources = [Path(path) for path in (training, calibration) if path is not None]
    if not sources:
        raise DatasetValidationError("provide at least one original training or calibration source")
    destination = Path(output_path)
    if destination.suffix.lower() != ".sdmdataset":
        raise DatasetValidationError("source attachment output must end in .sdmdataset")
    if destination.resolve() in {Path(model_path).resolve(), *(path.resolve() for path in sources)}:
        raise DatasetValidationError("source attachment output must differ from its inputs and model")
    file_digests = model_file_digests(model_path)
    artifact = load_artifact(model_path)
    identities, selected, original = _membership(artifact)
    prefix, target_split = role.split("-", 1)
    if prefix == "original" and original is None:
        raise DatasetValidationError("original-role exports require complete bestIterationSplits metadata; use a selected role")
    requested = original[target_split] if prefix == "original" else [row[0] for row in selected[target_split]]
    if not requested:
        raise DatasetValidationError(f"the model has no saved {target_split} source records")
    fingerprint = artifact.manifest["representation"]["fingerprint"]
    names = artifact.configuration["classNames"]
    loaded = {}
    for path in sources:
        bundle = DatasetBundle.open(path) if path.is_dir() else None
        try:
            if bundle is not None:
                if bundle.class_names is not None and tuple(names) != bundle.class_names:
                    raise DatasetValidationError("source classNames differ from the model's ordered classNames")
                if bundle.fingerprint not in (None, fingerprint):
                    raise DatasetValidationError("source representation fingerprint differs from the model")
            rows = bundle.iter_rows() if bundle is not None else iter_dataset_rows(path)
            for row in validate_dataset_rows(rows):
                identifier = row["id"]
                if identifier in loaded:
                    raise DatasetValidationError(f"source ID {identifier!r} appears more than once across inputs")
                if identifier not in identities:
                    raise DatasetValidationError(f"source ID {identifier!r} does not belong to the saved model")
                if row["label"] != identities[identifier]["label"]:
                    raise DatasetValidationError(f"source label for {identifier!r} differs from the saved model")
                if _source_fingerprint(row) not in (None, fingerprint):
                    raise DatasetValidationError("source representation fingerprint differs from the model")
                loaded[identifier] = row
        finally:
            if bundle is not None:
                bundle.close()
    missing = [identifier for identifier in requested if identifier not in loaded]
    if missing:
        raise DatasetValidationError(f"{role} is missing {len(missing)} source rows, including {missing[:5]!r}; supply the other original split")
    model = None
    output_rows, vectors, support_indices = [], [], []
    for identifier in requested:
        identity = identities[identifier]
        row = _source_fields(loaded[identifier], composition=composition)
        for key in _SCORE_FIELDS:
            row.pop(key, None)
        row["representationFingerprint"] = fingerprint
        # Explicit text-only export intentionally discards unneeded feature
        # columns before composing; text can be attached to older models too.
        if text_only:
            row.pop("embedding", None)
            row.pop("attributes", None)
        vector = _raw_vector(row, composition)
        digest = identity.get("expectedDigest")
        if vector is not None:
            if len(vector) != artifact.configuration["embeddingDimension"]:
                raise DatasetValidationError(f"source row {identifier!r} has the wrong composed feature dimension")
            actual_digest = feature_digest(vector)
            if digest is not None and actual_digest != digest:
                raise DatasetValidationError(f"source features for {identifier!r} differ from the training input digest")
            if digest is None:
                if model is None:
                    model = SDMModel(artifact, device=device, query_batch_size=query_batch_size, support_tile_size=support_tile_size)
                _legacy_feature_check(model, vector, identity)
            digest = actual_digest
        elif with_scores:
            raise DatasetValidationError("--with_scores requires complete cached features for every source row")
        descriptor = {key: identity[key] for key in ("poolIndex", "selectedSplit", "selectedIndex") if key in identity}
        if digest is not None:
            descriptor["featureSHA256"] = digest
        row["metadata"] = {**row.get("metadata", {}), "sdmSource": descriptor}
        output_rows.append(row)
        vectors.append(vector)
        support_indices.append(identity["selectedIndex"] if identity["selectedSplit"] == "training" else None)
    if with_scores:
        if model is None:
            model = SDMModel(artifact, device=device, query_batch_size=query_batch_size, support_tile_size=support_tile_size)
        dataset = Dataset(tuple(requested), np.stack(vectors), tuple(row["label"] for row in output_rows),
                          output_rows, fingerprint, composition, tuple(names))
        output_rows = score_dataset_rows(model, dataset, composition=composition, batch_size=query_batch_size,
                                         identity_support_indices=support_indices)
    # Guard against a model edited during an export; avoid publishing a bundle
    # that binds file hashes from one snapshot to identities from another.
    if file_digests != model_file_digests(model_path):
        raise DatasetValidationError("model files changed while preparing the source attachment")
    return write_dataset_bundle(destination, output_rows, class_names=names,
                                representation=artifact.manifest["representation"],
                                metadata={"modelSource": {"schemaVersion": 1, "modelID": artifact.model_id,
                                          "modelFiles": file_digests, "role": role}}, overwrite=overwrite)
