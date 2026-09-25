# Copyright Reexpress AI, Inc. All rights reserved.
"""Readable metadata plus mapped NPY matrices: the portable .sdmdataset format.

The on-disk layout is framework independent. No Python object arrays or pickle
are accepted. Metadata stays mapped and rows are decoded only when requested;
feature matrices remain read-only NumPy memory maps until an operation needs a
copy (for example, concatenating two feature sources or moving them to a GPU).
"""
from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator, Mapping, Sequence
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
from typing import Any
import uuid

import numpy as np

from .errors import DatasetValidationError, DimensionMismatchError, RepresentationMismatchError

FORMAT = "reexpress-dataset"
SCHEMA_VERSION = 1
FEATURE_FILES = {"embedding": "embeddings.npy", "attributes": "attributes.npy"}
FEATURE_REFS = {"embedding": "embeddingRow", "attributes": "attributesRow"}
MAX_HEADER_SIZE = 65_536


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise DatasetValidationError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _encode(value: Any) -> bytes:
    try:
        return json.dumps(_json_value(value), ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeError) as error:
        raise DatasetValidationError(f"metadata must contain finite JSON values: {error}") from error


def _decode(value: bytes | str, name: str) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = item
        return result

    def invalid_constant(value):
        raise ValueError(f"non-finite JSON number {value}")

    try:
        result = json.loads(value, object_pairs_hook=pairs, parse_constant=invalid_constant)
        if not isinstance(result, dict):
            raise ValueError("expected a JSON object")
        # JSON exponents such as 1e999 can overflow without parse_constant.
        _encode(result)
        return result
    except (ValueError, TypeError, UnicodeError) as error:
        raise DatasetValidationError(f"invalid {name}: {error}") from error


def _class_names(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) < 2 or any(
        not isinstance(item, str) or not item.strip() or item != item.strip() for item in value
    ) or len(set(value)) != len(value):
        raise DatasetValidationError("classNames must contain at least two distinct nonempty trimmed names")
    return tuple(value)


def _representation(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not isinstance(value.get("fingerprint"), str) or not value["fingerprint"].strip():
        raise DatasetValidationError("representation must be an object with a nonempty fingerprint")
    return _decode(_encode(value), "representation")


def _feature(value: Any, name: str) -> np.ndarray:
    try:
        if isinstance(value, (list, tuple)) and any(isinstance(item, (bool, np.bool_)) for item in value):
            raise ValueError("booleans are not numeric features")
        array = np.asarray(value)
        if array.dtype.kind not in "iuf" or array.ndim != 1 or array.size == 0:
            raise ValueError("expected a nonempty one-dimensional numeric array, excluding booleans")
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.asarray(array, dtype="<f4")
        if not np.isfinite(result).all():
            raise ValueError("values must remain finite in Float32")
        return np.ascontiguousarray(result)
    except (TypeError, ValueError) as error:
        raise DatasetValidationError(f"invalid {name}: {error}") from error


def _source_fingerprint(row: Mapping[str, Any]) -> str | None:
    """Resolve explicit row identity, then the legacy nested metadata identity."""
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise DatasetValidationError("metadata must be an object")
    if "representationFingerprint" in row:
        fingerprint = row["representationFingerprint"]
    elif "representationFingerprint" in metadata:
        fingerprint = metadata["representationFingerprint"]
    else:
        return None
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise DatasetValidationError("representationFingerprint must be a nonempty string")
    return fingerprint


def _validate_source_logits(row: Mapping[str, Any]) -> None:
    """Check reserved source scores without rounding their stored JSON numbers."""
    if "logits" in row and "sourceLogits" in row:
        raise DatasetValidationError("a row cannot contain both logits and legacy sourceLogits")
    key = "logits" if "logits" in row else "sourceLogits"
    if key not in row:
        return
    values = _json_value(row[key])
    if not isinstance(values, list) or len(values) < 2 or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in values
    ):
        raise DatasetValidationError(f"{key} must be an array of at least two finite JSON numbers, excluding booleans")
    # Reexpress two consumes these scores as Float32. Validate that portable range,
    # but retain the original values rather than replacing them with this array.
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            finite = np.isfinite(np.asarray(values, dtype=np.float32)).all()
    except (OverflowError, TypeError, ValueError) as error:
        raise DatasetValidationError(f"{key} values must remain finite in Float32") from error
    if not finite:
        raise DatasetValidationError(f"{key} values must remain finite in Float32")


class _RowValidator:
    def __init__(self, *, class_names=None, fingerprint=None):
        self.seen: set[str] = set()
        self.dimensions: dict[str, int] = {}
        self.fingerprint = fingerprint
        self.class_names = class_names

    def validate(self, value: Mapping[str, Any], *, features=True) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise DatasetValidationError("dataset rows must be objects")
        row = dict(value)
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in self.seen:
            raise DatasetValidationError("dataset contains a missing or duplicate id")
        self.seen.add(identifier)
        label = row.get("label")
        if isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)):
            raise DatasetValidationError(f"row {identifier!r} needs an integer label (-1 for unlabeled)")
        label = int(label)
        if label < 0 and label not in (-1, -99) or self.class_names is not None and label >= len(self.class_names):
            raise DatasetValidationError(f"row {identifier!r} has an invalid label")
        row["label"] = label
        for key in ("document", "prompt"):
            if key in row and row[key] is not None and not isinstance(row[key], str):
                raise DatasetValidationError(f"row {identifier!r} {key} must be a string or null")
        if "metadata" in row and not isinstance(row["metadata"], Mapping):
            raise DatasetValidationError(f"row {identifier!r} metadata must be an object")
        _validate_source_logits(row)
        fingerprint = _source_fingerprint(row)
        if fingerprint is not None:
            if self.fingerprint is not None and fingerprint != self.fingerprint:
                raise RepresentationMismatchError("dataset contains inconsistent representation fingerprints")
            self.fingerprint = fingerprint
        arrays = {}
        if features:
            for key in FEATURE_FILES:
                if FEATURE_REFS[key] in row:
                    raise DatasetValidationError(f"{FEATURE_REFS[key]} is reserved for bundle storage")
                if key not in row:
                    continue
                array = _feature(row.pop(key), key)
                dimension = int(array.size)
                if key in self.dimensions and self.dimensions[key] != dimension:
                    raise DimensionMismatchError(f"dataset has inconsistent {key} dimensions")
                self.dimensions[key] = dimension
                arrays[key] = array
        _encode(row)
        row.update(arrays)
        return row


def validate_dataset_rows(rows: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """Validate raw/source-only or featured canonical rows without collecting them."""
    validator = _RowValidator()
    for row in rows:
        yield validator.validate(row)
    if not validator.seen:
        raise DatasetValidationError("dataset is empty")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _member(root: Path, name: str) -> Path:
    result = root / name
    if result.is_symlink() or not result.is_file():
        raise DatasetValidationError(f"{name} must be a regular file inside the dataset bundle")
    return result


def _checked_file(root: Path, descriptor: Any, expected_name: str, verify: bool) -> Path:
    if not isinstance(descriptor, dict) or descriptor.get("path") != expected_name:
        raise DatasetValidationError(f"expected dataset member {expected_name}")
    checksum = descriptor.get("sha256")
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise DatasetValidationError(f"{expected_name} needs a lowercase SHA-256 digest")
    path = _member(root, expected_name)
    if verify and _digest(path) != checksum:
        raise DatasetValidationError(f"checksum mismatch for {expected_name}")
    return path


def _mapped_matrix(path: Path, descriptor: Mapping[str, Any]) -> np.memmap:
    shape = descriptor.get("shape")
    if descriptor.get("dtype") != "<f4" or not isinstance(shape, list) or len(shape) != 2:
        raise DatasetValidationError(f"{path.name} needs a two-dimensional <f4 matrix descriptor")
    shape = tuple(_integer(value, "matrix shape", 1) for value in shape)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY" or prefix[6:] not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
            raise DatasetValidationError(f"{path.name} is not supported NPY 1.0, 2.0, or 3.0")
        length_size = 2 if prefix[6] == 1 else 4
        raw_length = stream.read(length_size)
        if len(raw_length) != length_size:
            raise DatasetValidationError(f"truncated NPY header in {path.name}")
        length = int.from_bytes(raw_length, "little")
        if not 0 < length <= MAX_HEADER_SIZE:
            raise DatasetValidationError(f"NPY header exceeds {MAX_HEADER_SIZE} bytes")
        header = stream.read(length)
        if len(header) != length or not header.endswith(b"\n"):
            raise DatasetValidationError(f"truncated or invalid NPY header in {path.name}")
        try:
            expression = ast.parse(header.decode("utf-8" if prefix[6] == 3 else "latin1").strip(), mode="eval")
            if not isinstance(expression.body, ast.Dict) or len(expression.body.keys) != 3:
                raise ValueError("header must contain exactly three fields")
            keys = [ast.literal_eval(key) for key in expression.body.keys]
            if set(keys) != {"descr", "fortran_order", "shape"}:
                raise ValueError("unexpected header fields")
            decoded = ast.literal_eval(expression)
            actual_shape = decoded["shape"]
            if decoded["descr"] != "<f4" or decoded["fortran_order"] is not False or not isinstance(actual_shape, tuple) or len(actual_shape) != 2:
                raise ValueError("only C-order <f4 matrices are supported")
            actual_shape = tuple(_integer(value, "NPY shape", 1) for value in actual_shape)
            if actual_shape != shape:
                raise ValueError("NPY and manifest shapes differ")
        except (ValueError, TypeError, SyntaxError, UnicodeError) as error:
            raise DatasetValidationError(f"invalid NPY header in {path.name}: {error}") from error
        offset = stream.tell()
    if path.stat().st_size != offset + shape[0] * shape[1] * 4:
        raise DatasetValidationError(f"{path.name} payload length does not match its shape")
    result = np.memmap(path, mode="r", dtype="<f4", offset=offset, shape=shape, order="C")
    batch_size = max(1, (8 << 20) // (4 * shape[1]))
    for start in range(0, shape[0], batch_size):
        if not np.isfinite(result[start:start + batch_size]).all():
            raise DatasetValidationError(f"{path.name} contains non-finite Float32 values")
    return result


class DatasetBundle(Sequence[Mapping[str, Any]]):
    """Validated, read-only mapped dataset, including datasets without embeddings.

    ``matrices`` exposes compact memory-mapped columns. Indexing/``iter_rows``
    returns one metadata object with NumPy views for present features. Keep the
    bundle open while accessing rows. Exporters convert those views only as needed.
    """
    def __init__(self, path: str | Path, *, verify_checksums: bool = True):
        self.path = Path(path)
        self._metadata_map = None
        self._metadata_stream = None
        self._closed = False
        try:
            if self.path.is_symlink() or not self.path.is_dir():
                raise DatasetValidationError("dataset bundle must be a directory, not a symbolic link")
            manifest_path = _member(self.path, "manifest.json")
            if manifest_path.stat().st_size > 8 << 20:
                raise DatasetValidationError("dataset manifest exceeds 8 MiB")
            self.manifest = _decode(manifest_path.read_bytes(), "dataset manifest")
            manifest = self.manifest
            if manifest.get("format") != FORMAT or _integer(manifest.get("schemaVersion"), "schemaVersion", 1) != SCHEMA_VERSION:
                raise DatasetValidationError("unsupported dataset bundle format or schema version")
            self.row_count = _integer(manifest.get("rowCount"), "rowCount", 1)
            for key in ("classNames", "representation", "metadata"):
                if key in manifest and manifest[key] is None:
                    raise DatasetValidationError(f"omit absent {key} instead of using null")
            self.class_names = _class_names(manifest.get("classNames"))
            self.representation = _representation(manifest.get("representation"))
            self.fingerprint = self.representation["fingerprint"] if self.representation else None
            self.metadata = manifest.get("metadata", {})
            if not isinstance(self.metadata, dict):
                raise DatasetValidationError("dataset metadata must be an object")
            descriptors = manifest.get("matrices")
            if not isinstance(descriptors, dict) or not set(descriptors) <= set(FEATURE_FILES):
                raise DatasetValidationError("matrices must contain only embedding and/or attributes")
            rows_path = _checked_file(self.path, manifest.get("rows"), "rows.jsonl", verify_checksums)
            self.matrices = {}
            for key, descriptor in descriptors.items():
                matrix_path = _checked_file(self.path, descriptor, FEATURE_FILES[key], verify_checksums)
                self.matrices[key] = _mapped_matrix(matrix_path, descriptor)
                if self.matrices[key].shape[0] > self.row_count:
                    raise DatasetValidationError("matrix has more rows than the dataset")
            validator = _RowValidator(class_names=self.class_names, fingerprint=self.fingerprint)
            self._offsets: list[tuple[int, int]] = []
            counts = {key: 0 for key in self.matrices}
            identifiers, labels = [], []
            with rows_path.open("rb") as stream:
                while True:
                    start = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    row = _decode(line, f"metadata row {len(self._offsets) + 1}")
                    if any(key in row for key in FEATURE_FILES):
                        raise DatasetValidationError("bundle metadata must not contain inline embedding/attributes")
                    row = validator.validate(row, features=False)
                    for key, ref in FEATURE_REFS.items():
                        if ref not in row:
                            continue
                        index = _integer(row[ref], ref)
                        if key not in counts or index != counts[key] or index >= self.matrices[key].shape[0]:
                            raise DatasetValidationError(f"{ref} references must be sequential and match a declared matrix")
                        counts[key] += 1
                    self._offsets.append((start, len(line)))
                    identifiers.append(row["id"])
                    labels.append(row["label"])
            if len(self._offsets) != self.row_count or any(counts[key] != matrix.shape[0] for key, matrix in self.matrices.items()):
                raise DatasetValidationError("dataset row count or matrix reference count does not match manifest")
            self.fingerprint = validator.fingerprint
            self.ids, self.labels = tuple(identifiers), tuple(labels)
            self._metadata_stream = rows_path.open("rb")
            self._metadata_map = mmap.mmap(self._metadata_stream.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException as error:
            self.close()
            if isinstance(error, (OSError, ValueError, TypeError, OverflowError)):
                raise DatasetValidationError(f"unable to open dataset bundle: {error}") from error
            raise

    @classmethod
    def open(cls, path: str | Path, *, verify_checksums: bool = True) -> "DatasetBundle":
        return cls(path, verify_checksums=verify_checksums)

    @property
    def rows(self) -> "DatasetBundle":
        return self

    def __len__(self) -> int:
        return self.row_count

    def __getitem__(self, index):
        if self._closed:
            raise ValueError("dataset bundle is closed")
        if isinstance(index, slice):
            return tuple(self[row] for row in range(*index.indices(self.row_count)))
        if not isinstance(index, int):
            raise TypeError("dataset row indices must be integers")
        if index < 0:
            index += self.row_count
        if not 0 <= index < self.row_count:
            raise IndexError(index)
        start, length = self._offsets[index]
        row = _decode(self._metadata_map[start:start + length], f"metadata row {index + 1}")
        for key, ref in FEATURE_REFS.items():
            if ref in row:
                row[key] = self.matrices[key][row.pop(ref)]
        if self.fingerprint is not None:
            row.setdefault("representationFingerprint", self.fingerprint)
        return row

    def iter_rows(self) -> Iterator[Mapping[str, Any]]:
        for index in range(self.row_count):
            yield self[index]

    def close(self) -> None:
        self._closed = True
        if self._metadata_map is not None:
            self._metadata_map.close()
            self._metadata_map = None
        if self._metadata_stream is not None:
            self._metadata_stream.close()
            self._metadata_stream = None
        # Do not forcibly close np.memmap backing objects: callers may retain a
        # matrix/view. NumPy releases mappings when their last owner is released.
        if hasattr(self, "matrices"):
            self.matrices = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


def iter_dataset_rows(path: str | Path) -> Iterator[Mapping[str, Any]]:
    """Stream JSON objects, or hydrate mapped bundle rows.

    JSONL here intentionally permits cached-score rows without source IDs. Use
    ``validate_dataset_rows`` when requiring canonical source-dataset semantics.
    """
    path = Path(path)
    if path.is_dir() or path.suffix.lower() == ".sdmdataset":
        with DatasetBundle.open(path) as bundle:
            yield from bundle.iter_rows()
    else:
        try:
            with path.open("rb") as stream:
                for number, line in enumerate(stream, 1):
                    if line.strip():
                        yield _decode(line, f"JSONL line {number}")
        except OSError as error:
            raise DatasetValidationError(f"unable to read dataset: {error}") from error


def _npy_header(rows: int, dimension: int) -> bytes:
    # Reserve a fixed, aligned header so streamed arrays need no second payload
    # copy when the final row count becomes known. This is ordinary NPY 1.0.
    literal = f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {dimension}), }}".encode("ascii")
    header_length = 256 - 10
    if len(literal) + 1 > header_length:
        raise DatasetValidationError("matrix shape exceeds the supported NPY header")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", header_length) + literal + b" " * (header_length - len(literal) - 1) + b"\n"


def _publish_directory(temporary: Path, destination: Path, overwrite: bool) -> None:
    if destination.is_symlink():
        raise DatasetValidationError("refusing to replace a symbolic link")
    if not destination.exists():
        os.rename(temporary, destination)
        return
    if not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    if not destination.is_dir():
        raise DatasetValidationError("dataset bundle output must be a directory")
    manifest = destination / "manifest.json"
    if manifest.is_symlink() or not manifest.is_file() or _decode(manifest.read_bytes(), "existing manifest").get("format") != FORMAT:
        raise DatasetValidationError("refusing to replace a directory that is not an SDM dataset bundle")
    backup = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.backup")
    os.rename(destination, backup)
    try:
        os.rename(temporary, destination)
    except BaseException:
        os.rename(backup, destination)
        raise
    # The new package is committed. Cleanup failure must not report a failed
    # write after publication or roll back a complete successfully saved bundle.
    shutil.rmtree(backup, ignore_errors=True)


def write_dataset_bundle(
    path: str | Path, rows: Iterable[Mapping[str, Any]], *,
    class_names: Sequence[str] | None = None, representation: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None, overwrite: bool = False,
) -> Path:
    """Atomically stream canonical rows to metadata and compact NPY matrices."""
    destination = Path(path)
    names = _class_names(class_names)
    representation_value = _representation(representation)
    if metadata is not None and not isinstance(metadata, Mapping):
        raise DatasetValidationError("dataset metadata must be an object")
    metadata_value = _decode(_encode(metadata), "metadata") if metadata is not None else None
    if destination.is_symlink() or destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    streams = {}
    counts = {}
    validator = _RowValidator(class_names=names, fingerprint=representation_value["fingerprint"] if representation_value else None)
    total = 0
    try:
        with (temporary / "rows.jsonl").open("wb") as metadata_stream:
            for value in rows:
                row = validator.validate(value)
                for key, filename in FEATURE_FILES.items():
                    if key not in row:
                        continue
                    vector = row.pop(key)
                    if key not in streams:
                        streams[key] = (temporary / filename).open("w+b")
                        streams[key].write(_npy_header(0, len(vector)))
                        counts[key] = 0
                    streams[key].write(vector.tobytes(order="C"))
                    row[FEATURE_REFS[key]] = counts[key]
                    counts[key] += 1
                metadata_stream.write(_encode(row) + b"\n")
                total += 1
            metadata_stream.flush()
            os.fsync(metadata_stream.fileno())
        if not total:
            raise DatasetValidationError("dataset is empty")
        matrices = {}
        for key, stream in streams.items():
            stream.seek(0)
            stream.write(_npy_header(counts[key], validator.dimensions[key]))
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            filename = FEATURE_FILES[key]
            matrices[key] = {"path": filename, "dtype": "<f4", "shape": [counts[key], validator.dimensions[key]],
                             "sha256": _digest(temporary / filename)}
        manifest = {"format": FORMAT, "schemaVersion": SCHEMA_VERSION, "rowCount": total,
                    "rows": {"path": "rows.jsonl", "sha256": _digest(temporary / "rows.jsonl")}, "matrices": matrices}
        if names is not None:
            manifest["classNames"] = names
        if representation_value is not None:
            manifest["representation"] = representation_value
        elif validator.fingerprint is not None:
            manifest["representation"] = {"fingerprint": validator.fingerprint}
        if metadata_value is not None:
            manifest["metadata"] = metadata_value
        encoded_manifest = _encode(manifest) + b"\n"
        if len(encoded_manifest) > 8 << 20:
            raise DatasetValidationError("dataset manifest exceeds 8 MiB")
        with (temporary / "manifest.json").open("wb") as stream:
            stream.write(encoded_manifest)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_directory(temporary, destination, overwrite)
        return destination
    finally:
        for stream in streams.values():
            if not stream.closed:
                stream.close()
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def write_dataset_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool = False) -> Path:
    """Atomically write canonical dataset rows, converting only current-row views."""
    destination = Path(path)
    if destination.is_symlink() or destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            for row in validate_dataset_rows(rows):
                stream.write(_encode(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if destination.is_symlink() or destination.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {destination}")
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)
