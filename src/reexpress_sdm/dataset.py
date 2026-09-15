# Copyright Reexpress AI, Inc. All rights reserved.
"""Feature-ready datasets loaded from JSON Lines or mapped SDM bundles."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bundle import _source_fingerprint, _validate_source_logits
from .errors import DatasetValidationError, DimensionMismatchError, RepresentationMismatchError


@dataclass(frozen=True)
class Dataset:
    ids: tuple[str, ...]
    vectors: np.ndarray
    labels: tuple[int, ...]
    rows: Sequence[Mapping[str, Any]]
    representation_fingerprint: str | None = None
    composition: str = "auto"
    class_names: tuple[str, ...] | None = None

    @property
    def count(self) -> int:
        return len(self.ids)

    def validate_class_names(self, expected_class_names: Sequence[str]) -> None:
        """Require declared label meanings to match a model's class order.

        JSONL and bundles without classNames retain their existing integer-label
        convention. Declared names must never be reordered or replaced without
        also remapping labels explicitly.
        """
        if self.class_names is not None and tuple(self.class_names) != tuple(expected_class_names):
            raise DatasetValidationError(
                f"dataset classNames {list(self.class_names)!r} do not match the model's "
                f"ordered classNames {list(expected_class_names)!r}; use a dataset and model "
                "with the same class names in the same order"
            )

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "Dataset":
        """Load feature-ready JSONL or an .sdmdataset directory.

        Use DatasetBundle for source-only or partially embedded data. A complete
        single-column bundle retains a read-only mapped vector matrix here.
        Combining embedding and attribute columns allocates a composed matrix.
        """
        path = Path(path)
        if path.is_dir() or path.suffix.lower() == ".sdmdataset":
            return cls.from_bundle(path, **kwargs)
        return cls.from_jsonl(path, **kwargs)

    @classmethod
    def from_bundle(
        cls, path: str | Path, *, composition: str = "auto",
        expected_dimension: int | None = None, number_of_classes: int | None = None,
        require_labels: bool = False, representation_fingerprint: str | None = None,
        verify_checksums: bool = True,
    ) -> "Dataset":
        from .bundle import DatasetBundle
        if composition not in {"auto", "embedding", "attributes", "embedding+attributes"}:
            raise DatasetValidationError("composition must be auto, embedding, attributes, or embedding+attributes")
        if expected_dimension is not None and (
            isinstance(expected_dimension, bool) or not isinstance(expected_dimension, int) or expected_dimension < 1
        ):
            raise DatasetValidationError("expected_dimension must be a positive integer")
        if number_of_classes is not None and (
            isinstance(number_of_classes, bool) or not isinstance(number_of_classes, int) or number_of_classes < 2
        ):
            raise DatasetValidationError("number_of_classes must be at least two")
        if representation_fingerprint is not None and (
            not isinstance(representation_fingerprint, str) or not representation_fingerprint
        ):
            raise DatasetValidationError("representation_fingerprint must be a nonempty string")
        bundle = DatasetBundle.open(path, verify_checksums=verify_checksums)
        try:
            if representation_fingerprint is not None and bundle.fingerprint is not None and representation_fingerprint != bundle.fingerprint:
                raise RepresentationMismatchError("requested and dataset representation fingerprints differ")
            if number_of_classes is not None:
                if bundle.class_names is not None and len(bundle.class_names) != number_of_classes:
                    raise DatasetValidationError("dataset classNames count differs from the requested number of classes")
                if any(label >= number_of_classes for label in bundle.labels):
                    raise DatasetValidationError("dataset label is outside the requested classes")
            if require_labels and any(label < 0 for label in bundle.labels):
                raise DatasetValidationError("training/calibration requires a known class label for every row")
            keys = tuple(key for key in ("embedding", "attributes") if key in bundle.matrices) if composition == "auto" else (
                ("embedding", "attributes") if composition == "embedding+attributes" else (composition,)
            )
            if not keys or any(key not in bundle.matrices or bundle.matrices[key].shape[0] != bundle.row_count for key in keys):
                raise DatasetValidationError(
                    "training/scoring requires complete cached features for every row; "
                    "this bundle can still be inspected and exported with DatasetBundle"
                )
            matrices = [bundle.matrices[key] for key in keys]
            dimension = sum(matrix.shape[1] for matrix in matrices)
            if expected_dimension is not None and dimension != expected_dimension:
                raise DimensionMismatchError(f"dataset dimension {dimension} != {expected_dimension}")
            vectors = matrices[0] if len(matrices) == 1 else np.concatenate(matrices, axis=1)
            return cls(ids=bundle.ids, vectors=vectors, labels=bundle.labels, rows=bundle,
                       representation_fingerprint=representation_fingerprint or bundle.fingerprint,
                       composition=composition, class_names=bundle.class_names)
        except BaseException:
            bundle.close()
            raise

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        composition: str = "auto",
        expected_dimension: int | None = None,
        number_of_classes: int | None = None,
        require_labels: bool = False,
        representation_fingerprint: str | None = None,
    ) -> "Dataset":
        if composition not in {"auto", "embedding", "attributes", "embedding+attributes"}:
            raise DatasetValidationError(
                "composition must be auto, embedding, attributes, or embedding+attributes"
            )
        if expected_dimension is not None and (
            isinstance(expected_dimension, bool)
            or not isinstance(expected_dimension, int)
            or expected_dimension < 1
        ):
            raise DatasetValidationError("expected_dimension must be a positive integer")
        if number_of_classes is not None and (
            isinstance(number_of_classes, bool)
            or not isinstance(number_of_classes, int)
            or number_of_classes < 2
        ):
            raise DatasetValidationError("number_of_classes must be at least two")
        if representation_fingerprint is not None and (
            not isinstance(representation_fingerprint, str) or not representation_fingerprint
        ):
            raise DatasetValidationError("representation_fingerprint must be a nonempty string")
        ids: list[str] = []
        vectors: list[np.ndarray] = []
        labels: list[int] = []
        rows: list[Mapping[str, Any]] = []
        seen_ids: set[str] = set()
        observed_fingerprint: str | None = None
        try:
            with Path(path).open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise DatasetValidationError(f"line {line_number} must be an object")
                    _validate_source_logits(value)
                    identifier = value.get("id")
                    if not isinstance(identifier, str) or not identifier or identifier in seen_ids:
                        raise DatasetValidationError(
                            f"line {line_number} has a missing or duplicate id"
                        )
                    seen_ids.add(identifier)
                    ids.append(identifier)

                    selected_composition = composition
                    if selected_composition == "auto":
                        if "embedding" in value and "attributes" in value:
                            selected_composition = "embedding+attributes"
                        elif "embedding" in value:
                            selected_composition = "embedding"
                        elif "attributes" in value:
                            selected_composition = "attributes"
                        else:
                            raise DatasetValidationError(
                                f"line {line_number} has neither embedding nor attributes"
                            )
                    if selected_composition == "embedding+attributes":
                        if "embedding" not in value or "attributes" not in value:
                            raise DatasetValidationError(
                                f"line {line_number} needs embedding and attributes"
                            )
                        if not isinstance(value["embedding"], list) or not isinstance(
                            value["attributes"], list
                        ):
                            raise DatasetValidationError(
                                f"line {line_number} embedding and attributes must be arrays"
                            )
                        if not value["embedding"] or not value["attributes"]:
                            raise DatasetValidationError(
                                f"line {line_number} embedding and attributes must be nonempty"
                            )
                        raw_vector = value["embedding"] + value["attributes"]
                    else:
                        if selected_composition not in value:
                            raise DatasetValidationError(
                                f"line {line_number} is missing {selected_composition}"
                            )
                        raw_vector = value[selected_composition]
                    if not isinstance(raw_vector, list) or any(
                        isinstance(item, bool)
                        or not isinstance(item, (int, float))
                        or not math.isfinite(item)
                        for item in raw_vector
                    ):
                        raise DatasetValidationError(
                            f"line {line_number} vector must contain only finite JSON numbers"
                        )
                    with np.errstate(over="ignore", invalid="ignore"):
                        vector = np.asarray(raw_vector, dtype=np.float32).reshape(-1)
                    if vector.size == 0 or not np.all(np.isfinite(vector)):
                        raise DatasetValidationError(
                            f"line {line_number} has an empty or non-finite vector"
                        )
                    if expected_dimension is not None and vector.size != expected_dimension:
                        raise DimensionMismatchError(
                            f"line {line_number} dimension {vector.size} != {expected_dimension}"
                        )
                    if vectors and vector.size != vectors[0].size:
                        raise DimensionMismatchError(f"line {line_number} has an inconsistent dimension")
                    vectors.append(vector)

                    if "label" not in value:
                        if require_labels:
                            raise DatasetValidationError(f"line {line_number} is missing label")
                        label = -1
                    else:
                        label_value = value["label"]
                        if isinstance(label_value, bool) or not isinstance(label_value, int):
                            raise DatasetValidationError(f"line {line_number} label must be an integer")
                        label = label_value
                    if label < 0 and label not in {-1, -99}:
                        raise DatasetValidationError(f"line {line_number} label is invalid")
                    if number_of_classes is not None and label >= number_of_classes:
                        raise DatasetValidationError(f"line {line_number} label is invalid")
                    if require_labels and (
                        label < 0
                        or number_of_classes is not None and label >= number_of_classes
                    ):
                        raise DatasetValidationError(
                            f"line {line_number} requires a known class label"
                        )
                    labels.append(label)

                    row_fingerprint = _source_fingerprint(value)
                    if row_fingerprint is not None:
                        if observed_fingerprint is None:
                            observed_fingerprint = row_fingerprint
                        elif observed_fingerprint != row_fingerprint:
                            raise RepresentationMismatchError(
                                "dataset contains multiple representation fingerprints"
                            )
                    rows.append(value)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DatasetValidationError(f"unable to read dataset: {error}") from error
        if not vectors:
            raise DatasetValidationError("dataset is empty")
        effective_fingerprint = representation_fingerprint or observed_fingerprint
        if (
            representation_fingerprint is not None
            and observed_fingerprint is not None
            and representation_fingerprint != observed_fingerprint
        ):
            raise RepresentationMismatchError(
                "requested and row-level representation fingerprints differ"
            )
        return cls(
            ids=tuple(ids),
            vectors=np.stack(vectors).astype(np.float32, copy=False),
            labels=tuple(labels),
            rows=tuple(rows),
            representation_fingerprint=effective_fingerprint,
            composition=composition,
        )
