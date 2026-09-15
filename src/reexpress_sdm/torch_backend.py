# Copyright Reexpress AI, Inc. All rights reserved.
"""PyTorch training and exact dense matching on CPU, MPS, or CUDA.

Framework imports remain lazy until a runtime is constructed. Projection,
optimization, and matching use the selected device; portable Float32 arrays,
CDFs, balanced metrics, and nested region fitting share the host contract.
"""

from __future__ import annotations

import importlib
import math
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .calibration import NestedCalibrator
from .errors import DimensionMismatchError
from .math import rescaled_similarities
from .training import (
    TrainingConfig,
    TrainingResult,
    TrainingControl,
    _EpochEvaluator,
    _run_epochs,
    _starting_model,
    _representation_defaults,
    _weight_arrays,
    _check_control,
    _finish_training_result,
    _balanced_loss,
    _distance_cdfs,
    _distances_from_cdfs,
    _probabilities,
    _validate_training_arrays,
    build_artifact,
)
from .types import AdapterWeights, SupportRecord, SDMArtifact
from .source_data import build_source_data


def _load_torch():
    """Import the required Torch dependency with actionable installation guidance."""

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError(
            "PyTorch is required; install the current package dependencies with "
            "`python -m pip install reexpress_sdm`"
        ) from error


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _resolve_device(torch: Any, requested: str) -> Any:
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("device must be a nonempty PyTorch device string or 'auto'")
    requested = requested.strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    try:
        device = torch.device(requested)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid PyTorch device: {requested!r}") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but PyTorch reports that CUDA is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {device.index} is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    elif device.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("MPS was requested, but PyTorch reports that MPS is unavailable")
    elif device.type != "cpu":
        raise ValueError("the PyTorch backend supports cpu, mps, and cuda devices")
    elif device.index is not None:
        device = torch.device("cpu")
    return device


def _numeric_float32_matrix(value: Any, name: str, dimension: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        result = raw.astype(np.float32, copy=False)
    if result.ndim != 2 or (dimension is not None and result.shape[1] != dimension):
        expected = "a matrix" if dimension is None else f"a matrix with dimension {dimension}"
        raise DimensionMismatchError(f"{name} must be {expected}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains a non-finite value")
    return np.ascontiguousarray(result, dtype=np.float32)


def _lexicographic_smallest(
    torch: Any,
    distances: Any,
    indices: Any,
    count: int,
) -> tuple[Any, Any]:
    """Keep ``count`` rows ordered by distance, then support index."""

    index_order = torch.argsort(indices, dim=1, stable=True)
    indices = torch.gather(indices, 1, index_order)
    distances = torch.gather(distances, 1, index_order)
    distance_order = torch.argsort(distances, dim=1, stable=True)[:, :count]
    return (
        torch.gather(distances, 1, distance_order),
        torch.gather(indices, 1, distance_order),
    )


def _deterministic_partial_topk(
    torch: Any,
    distances: Any,
    indices: Any,
    count: int,
    support_count: int,
) -> tuple[Any, Any]:
    """Select top-k without relying on ``torch.topk`` tie ordering.

    ``topk`` is used only to discover the kth distance.  Every strictly closer
    item is retained and the remaining boundary slots are filled using the
    smallest support indices.  Only that at-most-2k candidate set is stably
    sorted into the artifact contract's lexicographic order.
    """

    count = min(count, distances.shape[1])
    if distances.shape[1] <= count:
        return _lexicographic_smallest(torch, distances, indices, count)

    threshold = torch.topk(
        distances, count, dim=1, largest=False, sorted=False
    ).values.max(dim=1).values
    infinity = torch.tensor(float("inf"), dtype=distances.dtype, device=distances.device)
    sentinel = torch.tensor(support_count, dtype=indices.dtype, device=indices.device)

    strict_candidates = torch.where(distances < threshold[:, None], distances, infinity)
    strict_distances, strict_positions = torch.topk(
        strict_candidates, count, dim=1, largest=False, sorted=False
    )
    strict_indices = torch.gather(indices, 1, strict_positions)
    strict_valid = torch.isfinite(strict_distances)
    strict_indices = torch.where(strict_valid, strict_indices, sentinel)

    boundary_keys = torch.where(distances == threshold[:, None], indices, sentinel)
    selected_keys, boundary_positions = torch.topk(
        boundary_keys, count, dim=1, largest=False, sorted=False
    )
    boundary_valid = selected_keys != sentinel
    boundary_distances = torch.gather(distances, 1, boundary_positions)
    boundary_indices = torch.gather(indices, 1, boundary_positions)
    boundary_distances = torch.where(boundary_valid, boundary_distances, infinity)
    boundary_indices = torch.where(boundary_valid, boundary_indices, sentinel)

    return _lexicographic_smallest(
        torch,
        torch.cat((strict_distances, boundary_distances), dim=1),
        torch.cat((strict_indices, boundary_indices), dim=1),
        count,
    )


class TorchExactL2Index:
    """Batched, tiled, exact squared-L2 matching on a PyTorch device.

    The ordering contract is Float32 distance ascending, followed by original
    support index ascending for exact ties.  Stable sorts are used at every
    tile merge so results do not inherit the unspecified tie order of
    ``torch.topk``.
    """

    def __init__(
        self,
        support_vectors: np.ndarray,
        *,
        device: str = "auto",
        query_batch_size: int = 256,
        support_tile_size: int = 16_384,
        training_control: TrainingControl | None = None,
    ):
        torch = _load_torch()
        vectors = _numeric_float32_matrix(support_vectors, "support vectors")
        if vectors.shape[0] == 0 or vectors.shape[1] == 0:
            raise DimensionMismatchError("support vectors must have shape [N, M] with N,M > 0")
        self._torch = torch
        self._training_control = training_control
        self._device = _resolve_device(torch, device)
        self._query_batch_size = _positive_integer(query_batch_size, "query_batch_size")
        self._support_tile_size = _positive_integer(support_tile_size, "support_tile_size")
        self._count = int(vectors.shape[0])
        self._dimension = int(vectors.shape[1])

        # As in ExactL2Index, translate by a fixed support origin before using
        # the norm expansion identity to avoid cancellation for large offsets.
        origin = np.array(vectors[0], dtype=np.float32, copy=True)
        with np.errstate(over="ignore", invalid="ignore"):
            centered = np.ascontiguousarray(vectors - origin, dtype=np.float32)
            squared_norms = np.sum(centered * centered, axis=1, dtype=np.float32)
        if not np.all(np.isfinite(centered)) or not np.all(np.isfinite(squared_norms)):
            raise ValueError("centered support vectors or their squared norms are non-finite")
        self._origin = torch.as_tensor(origin, dtype=torch.float32, device=self._device)
        self._vectors = torch.as_tensor(centered, dtype=torch.float32, device=self._device)
        self._squared_norms = torch.as_tensor(
            squared_norms, dtype=torch.float32, device=self._device
        )

    @property
    def count(self) -> int:
        return self._count

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def query_batch_size(self) -> int:
        return self._query_batch_size

    @property
    def support_tile_size(self) -> int:
        return self._support_tile_size

    def search_one(
        self, query: np.ndarray, k: int, exclude_index: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(query, self._torch.Tensor):
            exclusions = None if exclude_index is None else [exclude_index]
            distances, indices = self.search_many(query.reshape(1, -1), k, exclusions)
            return distances[0], indices[0]
        raw_query = np.asarray(query)
        if raw_query.dtype.kind not in "iuf":
            raise ValueError("query must contain numeric values")
        with np.errstate(over="ignore", invalid="ignore"):
            query_array = raw_query.astype(np.float32, copy=False).reshape(-1)
        if query_array.shape[0] != self.dimension:
            raise DimensionMismatchError(
                f"query dimension {query_array.shape[0]} != support dimension {self.dimension}"
            )
        exclusions = None if exclude_index is None else [exclude_index]
        distances, indices = self.search_many(query_array[None, :], k, exclusions)
        return distances[0], indices[0]

    def search_many(
        self,
        queries: np.ndarray,
        k: int,
        exclude_indices: Sequence[int] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exact neighbors for a query matrix without materializing QxN.

        ``exclude_indices`` supplies one support index per query and is useful
        for identity exclusion while matching the training split against
        itself.  All rows consequently return the same number of neighbors.
        """

        if isinstance(queries, self._torch.Tensor):
            queries = queries.detach().to(device=self._device)
            if queries.dtype == self._torch.bool or queries.is_complex() or queries.is_quantized:
                raise ValueError("queries must contain real numeric values")
            return self.search_many_device(queries.to(dtype=self._torch.float32), k, exclude_indices)
        return self._search_many(queries, k, exclude_indices, resident=False)

    def search_many_device(self, queries, k, exclude_indices=None):
        """Match resident Float32 query tensors without an exemplar readback."""
        if not isinstance(queries, self._torch.Tensor) or queries.dtype != self._torch.float32:
            raise ValueError("resident queries must be Float32 Torch tensors")
        same_device = (queries.device.type == self._device.type
                       and (self._device.index is None or queries.device.index == self._device.index))
        if queries.ndim != 2 or queries.shape[1] != self.dimension or not same_device:
            raise DimensionMismatchError("resident queries must match the index dimension and device")
        return self._search_many(queries, k, exclude_indices, resident=True)

    def _search_many(self, queries, k, exclude_indices=None, *, resident):
        query_array = queries if resident else _numeric_float32_matrix(queries, "queries", self.dimension)
        requested = _positive_integer(k, "k")
        exclusion_array: np.ndarray | None = None
        if exclude_indices is not None:
            raw_exclusions = np.asarray(exclude_indices)
            if raw_exclusions.size == 0 and query_array.shape[0] == 0 and raw_exclusions.ndim == 1:
                raw_exclusions = raw_exclusions.astype(np.int64)
            if raw_exclusions.dtype.kind not in "iu" or raw_exclusions.ndim != 1:
                raise ValueError("exclude_indices must be a one-dimensional integer array")
            exclusion_array = raw_exclusions.astype(np.int64, copy=False)
            if exclusion_array.shape[0] != query_array.shape[0]:
                raise DimensionMismatchError("exclude_indices must contain one value per query")
            if np.any(exclusion_array < 0) or np.any(exclusion_array >= self.count):
                raise ValueError("an exclude_indices value is outside the support")

        available = self.count - (1 if exclusion_array is not None else 0)
        result_count = min(requested, available)
        result_distances = np.empty((query_array.shape[0], result_count), dtype=np.float32)
        result_indices = np.empty((query_array.shape[0], result_count), dtype=np.int64)
        if query_array.shape[0] == 0 or result_count == 0:
            return result_distances, result_indices
        torch = self._torch

        with torch.no_grad(), torch.autocast(device_type=self._device.type, enabled=False):
            for query_start in range(0, query_array.shape[0], self._query_batch_size):
                _check_control(self._training_control)
                query_end = min(query_start + self._query_batch_size, query_array.shape[0])
                query_tensor = query_array[query_start:query_end] if resident else torch.as_tensor(
                    query_array[query_start:query_end], dtype=torch.float32, device=self._device
                )
                centered_queries = query_tensor - self._origin
                query_norms = torch.sum(centered_queries * centered_queries, dim=1)
                best_distances = torch.empty(
                    (query_end - query_start, 0), dtype=torch.float32, device=self._device
                )
                best_indices = torch.empty(
                    (query_end - query_start, 0), dtype=torch.int64, device=self._device
                )
                batch_exclusions = (
                    None
                    if exclusion_array is None
                    else torch.as_tensor(
                        exclusion_array[query_start:query_end],
                        dtype=torch.int64,
                        device=self._device,
                    )
                )
                all_finite = torch.tensor(True, dtype=torch.bool, device=self._device)

                for support_start in range(0, self.count, self._support_tile_size):
                    _check_control(self._training_control)
                    support_end = min(support_start + self._support_tile_size, self.count)
                    support = self._vectors[support_start:support_end]
                    distances = (
                        query_norms[:, None]
                        + self._squared_norms[support_start:support_end][None, :]
                        - 2.0 * (centered_queries @ support.transpose(0, 1))
                    ).clamp_min_(0.0)
                    all_finite = all_finite & torch.isfinite(distances).all()
                    indices = torch.arange(
                        support_start,
                        support_end,
                        dtype=torch.int64,
                        device=self._device,
                    ).expand(query_end - query_start, -1)
                    if batch_exclusions is not None:
                        distances = distances.masked_fill(
                            indices == batch_exclusions[:, None], float("inf")
                        )

                    candidate_distances = torch.cat((best_distances, distances), dim=1)
                    candidate_indices = torch.cat((best_indices, indices), dim=1)
                    best_distances, best_indices = _deterministic_partial_topk(
                        torch,
                        candidate_distances,
                        candidate_indices,
                        result_count,
                        self.count,
                    )

                if not bool(all_finite.item()):
                    raise ValueError("squared-L2 matching produced a non-finite distance")
                batch_distances = best_distances.cpu().numpy().astype(np.float32, copy=False)
                batch_indices = best_indices.cpu().numpy().astype(np.int64, copy=False)
                if not np.all(np.isfinite(batch_distances)):
                    raise ValueError("squared-L2 matching produced a non-finite distance")
                result_distances[query_start:query_end] = batch_distances
                result_indices[query_start:query_end] = batch_indices
        return result_distances, result_indices


def _torch_forward_numpy(
    torch: Any,
    values: np.ndarray,
    projection_weight: Any,
    projection_bias: Any,
    classifier_weight: Any,
    classifier_bias: Any,
    device: Any,
    batch_size: int,
    *,
    include_exemplars: bool = True,
    control: TrainingControl | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    logits: list[np.ndarray] = []
    exemplars: list[np.ndarray] = []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, values.shape[0], batch_size):
            _check_control(control)
            batch = torch.as_tensor(
                values[start : start + batch_size], dtype=torch.float32, device=device
            )
            batch_exemplars = batch @ projection_weight.transpose(0, 1) + projection_bias
            batch_logits = batch_exemplars @ classifier_weight.transpose(0, 1) + classifier_bias
            if include_exemplars:
                exemplars.append(batch_exemplars.cpu().numpy().astype(np.float32, copy=False))
            logits.append(batch_logits.cpu().numpy().astype(np.float32, copy=False))
    result_logits = np.ascontiguousarray(np.concatenate(logits, axis=0), dtype=np.float32)
    result_exemplars = np.ascontiguousarray(np.concatenate(exemplars, axis=0), dtype=np.float32) if include_exemplars else None
    if not np.all(np.isfinite(result_logits)) or (result_exemplars is not None and not np.all(np.isfinite(result_exemplars))):
        raise ValueError("adaptor produced a non-finite exemplar or logit")
    return result_logits, result_exemplars


def _q_d0_many(
    index: TorchExactL2Index,
    exemplars: np.ndarray,
    predictions: np.ndarray,
    support_labels: np.ndarray,
    support_predictions: np.ndarray,
    max_neighbors: int,
    identity: bool,
) -> tuple[np.ndarray, np.ndarray]:
    requested_neighbors = max_neighbors - 1 if identity else max_neighbors
    if requested_neighbors < 1:
        raise ValueError("training matching requires max_neighbors >= 2")
    if identity and index.count < 2:
        raise ValueError("a training split needs at least two support rows")
    count = exemplars.shape[0]
    batch_size = getattr(index, "query_batch_size", 256)
    q_values = np.empty(count, dtype=np.float32)
    d0_values = np.empty(count, dtype=np.float32)
    for start in range(0, count, batch_size):
        _check_control(getattr(index, "_training_control", None))
        stop = min(start + batch_size, count)
        # Identity indices address the original support, not the query chunk.
        exclusions = np.arange(start, stop, dtype=np.int64) if identity else None
        distances, neighbors = index.search_many(
            exemplars[start:stop], requested_neighbors, exclude_indices=exclusions
        )
        if neighbors.shape[1] == 0:
            raise ValueError("a training split needs at least two support rows")
        good = ((support_labels[neighbors] == support_predictions[neighbors])
                & (support_predictions[neighbors] == predictions[start:stop, None]))
        # Reduce while the bounded query×k result is live. An Int64 cumulative
        # product over the full dataset was both larger and unnecessary.
        q_values[start:stop] = np.sum(np.logical_and.accumulate(good, axis=1), axis=1)
        d0_values[start:stop] = distances[:, 0]
    return q_values, d0_values


class TorchTrainer:
    """Accelerated SDM trainer with portable Float32 artifact export."""

    def __init__(
        self,
        configuration: TrainingConfig,
        *,
        device: str = "auto",
        matching_query_batch_size: int = 256,
        matching_support_tile_size: int = 16_384,
    ):
        self.configuration = configuration
        self._torch = _load_torch()
        self._device = _resolve_device(self._torch, device)
        self._matching_query_batch_size = _positive_integer(
            matching_query_batch_size, "matching_query_batch_size"
        )
        self._matching_support_tile_size = _positive_integer(
            matching_support_tile_size, "matching_support_tile_size"
        )

    @property
    def device(self) -> str:
        return str(self._device)

    def fit(
        self,
        train_vectors: np.ndarray,
        train_labels: Sequence[int],
        calibration_vectors: np.ndarray,
        calibration_labels: Sequence[int],
        *,
        representation_fingerprint: str | None = None,
        train_ids: Sequence[str] | None = None,
        calibration_ids: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        representation_provider: str | None = None,
        representation_model: str | None = None,
        initial_artifact: SDMArtifact | None = None,
        control: TrainingControl | None = None,
        progress: Callable[[Mapping[str, float | int | bool | None]], None] | None = None,
    ) -> TrainingResult:
        """Fit a model, recording wall time through final calibration, excluding export.

        Unspecified representation metadata comes from the initial model, or
        uses embedding_v1/precomputed with no representation model name.
        """
        started = perf_counter()
        config = self.configuration
        train_vectors, train_labels_array, calibration_vectors, calibration_labels_array = (
            _validate_training_arrays(
                train_vectors,
                train_labels,
                calibration_vectors,
                calibration_labels,
                config.number_of_classes,
            )
        )
        representation_fingerprint, representation_provider, representation_model = _representation_defaults(
            representation_fingerprint, representation_provider, representation_model, initial_artifact)
        if not isinstance(representation_fingerprint, str) or not representation_fingerprint:
            raise ValueError("representation_fingerprint cannot be empty")
        if not isinstance(representation_provider, str) or not representation_provider:
            raise ValueError("representation_provider cannot be empty")
        if class_names is not None and (
            len(class_names) != config.number_of_classes
            or any(not isinstance(value, str) or not value for value in class_names)
            or len(set(class_names)) != len(class_names)
        ):
            raise ValueError("class_names must contain one unique nonempty string per class")
        train_ids = tuple(train_ids) if train_ids is not None else tuple(
            f"train-{index}" for index in range(len(train_labels_array))
        )
        calibration_ids = tuple(calibration_ids) if calibration_ids is not None else tuple(
            f"calibration-{index}" for index in range(len(calibration_labels_array))
        )
        if len(train_ids) != len(train_labels_array) or len(calibration_ids) != len(
            calibration_labels_array
        ):
            raise DimensionMismatchError("ids must align with their split")
        if len(set(train_ids)) != len(train_ids) or len(set(calibration_ids)) != len(
            calibration_ids
        ):
            raise ValueError("ids must be unique within each split")
        if any(not isinstance(value, str) or not value for value in train_ids + calibration_ids):
            raise ValueError("ids must be nonempty strings")
        if not set(train_ids).isdisjoint(calibration_ids):
            raise ValueError("training and calibration ids must not overlap")

        _check_control(control)
        config, class_names, saved_normalization = _starting_model(
            config, initial_artifact, train_vectors.shape[1], class_names, representation_fingerprint)
        mean = float(saved_normalization["mean"]) if saved_normalization else float(np.mean(train_vectors, dtype=np.float32))
        standard_deviation = float(saved_normalization["standardDeviation"]) if saved_normalization else float(np.std(train_vectors, ddof=1, dtype=np.float32))
        if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
            raise ValueError("training vectors must have a positive sample standard deviation")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            normalized_train = np.ascontiguousarray(
                (train_vectors - np.float32(mean)) / np.float32(standard_deviation),
                dtype=np.float32,
            )
            normalized_calibration = np.ascontiguousarray(
                (calibration_vectors - np.float32(mean)) / np.float32(standard_deviation),
                dtype=np.float32,
            )
        if not np.all(np.isfinite(normalized_train)) or not np.all(
            np.isfinite(normalized_calibration)
        ):
            raise ValueError("normalization produced a non-finite value")

        torch = self._torch
        rng = np.random.default_rng(config.seed)
        input_dimension = train_vectors.shape[1]
        projection_bound = 1.0 / math.sqrt(input_dimension)
        classifier_bound = 1.0 / math.sqrt(config.exemplar_dimension)

        # Host Float32 initialization intentionally preserves the established
        # seeded initialization before tensors move to the selected device.
        if initial_artifact is not None:
            initial_arrays = _weight_arrays(initial_artifact.weights)
        else:
            initial_arrays = (
                rng.uniform(
                    -projection_bound,
                    projection_bound,
                    (config.exemplar_dimension, input_dimension),
                ).astype(np.float32),
                rng.uniform(
                    -projection_bound, projection_bound, config.exemplar_dimension
                ).astype(np.float32),
                rng.uniform(
                    -classifier_bound,
                    classifier_bound,
                    (config.number_of_classes, config.exemplar_dimension),
                ).astype(np.float32),
                rng.uniform(
                    -classifier_bound, classifier_bound, config.number_of_classes
                ).astype(np.float32),
            )
        parameters = [
            torch.nn.Parameter(torch.as_tensor(value, dtype=torch.float32, device=self._device))
            for value in initial_arrays
        ]
        projection_weight, projection_bias, classifier_weight, classifier_bias = parameters
        optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
        def train_epoch(train_q, train_d):
            marginal_loss_sum = 0.0
            order = rng.permutation(len(train_labels_array))
            for start in range(0, len(order), config.batch_size):
                _check_control(control)
                rows = order[start : start + config.batch_size]
                x = torch.as_tensor(
                    normalized_train[rows], dtype=torch.float32, device=self._device
                )
                y = torch.as_tensor(
                    train_labels_array[rows], dtype=torch.int64, device=self._device
                )
                q = torch.as_tensor(train_q[rows], dtype=torch.float32, device=self._device)
                d = torch.as_tensor(train_d[rows], dtype=torch.float32, device=self._device)
                optimizer.zero_grad(set_to_none=True)
                exemplars = x @ projection_weight.transpose(0, 1) + projection_bias
                logits = exemplars @ classifier_weight.transpose(0, 1) + classifier_bias
                log_base = torch.log(q + float(config.q_offset))
                scaled = logits * d[:, None] * log_base[:, None]
                losses = torch.nn.functional.cross_entropy(scaled, y, reduction="none") / log_base
                loss = torch.mean(losses)
                batch_loss = float(loss.detach().item())
                if not math.isfinite(batch_loss):
                    raise ValueError("training loss is non-finite")
                marginal_loss_sum += batch_loss * len(rows)
                loss.backward()
                optimizer.step()

            return marginal_loss_sum / len(train_labels_array)

        def snapshot():
            return tuple(np.ascontiguousarray(parameter.detach().cpu().numpy(), dtype=np.float32).copy()
                         for parameter in parameters)

        # Historical CE candidates get separate tensors; no load_state_dict or
        # optimizer reset can disturb the latest weights/Adam transition state.
        frozen_parameters = None
        frozen_tensors = None
        def forward(calibration, saved_parameters=None, *, logits_only=False):
            nonlocal frozen_parameters, frozen_tensors
            _check_control(control)
            if saved_parameters is None:
                tensors = parameters
            else:
                if frozen_parameters is not saved_parameters:
                    frozen_parameters = saved_parameters
                    frozen_tensors = [torch.as_tensor(value, dtype=torch.float32, device=self._device)
                                      for value in saved_parameters]
                tensors = frozen_tensors
            return _torch_forward_numpy(
                torch, normalized_calibration if calibration else normalized_train,
                *tensors, self._device, self._matching_query_batch_size,
                include_exemplars=not logits_only, control=control)

        def index_factory(exemplars):
            return TorchExactL2Index(exemplars, device=str(self._device),
                query_batch_size=self._matching_query_batch_size,
                support_tile_size=self._matching_support_tile_size, training_control=control)

        evaluator = _EpochEvaluator(config, train_labels_array, calibration_labels_array,
                                    forward, index_factory, _q_d0_many, control)
        best_parameters, best_epoch, best_loss, history = _run_epochs(
            config, len(train_labels_array), train_epoch, snapshot, evaluator, progress, control)
        assert best_parameters is not None
        result = self._canonical_result(
            best_parameters,
            normalized_train,
            train_labels_array,
            normalized_calibration,
            calibration_labels_array,
            train_ids,
            calibration_ids,
            input_dimension=input_dimension,
            normalization_mean=mean,
            normalization_standard_deviation=standard_deviation,
            representation_fingerprint=representation_fingerprint,
            class_names=class_names,
            representation_provider=representation_provider,
            representation_model=representation_model,
            best_epoch=best_epoch,
            accelerated_selection_loss=best_loss,
            history=history,
            configuration=config,
            control=control,
        )
        source_data = build_source_data(train_vectors, calibration_vectors)
        return _finish_training_result(result, config, initial_artifact, control,
                                       duration_seconds=perf_counter() - started, source_data=source_data)

    def _canonical_result(
        self,
        parameters: tuple[np.ndarray, ...],
        normalized_train: np.ndarray,
        train_labels: np.ndarray,
        normalized_calibration: np.ndarray,
        calibration_labels: np.ndarray,
        train_ids: Sequence[str],
        calibration_ids: Sequence[str],
        *,
        input_dimension: int,
        normalization_mean: float,
        normalization_standard_deviation: float,
        representation_fingerprint: str,
        class_names: Sequence[str] | None,
        representation_provider: str,
        representation_model: str | None,
        best_epoch: int,
        accelerated_selection_loss: float,
        history: Sequence[Mapping[str, float | int | bool | None]],
        configuration: TrainingConfig | None = None,
        control: TrainingControl | None = None,
    ) -> TrainingResult:
        """Project and match on the selected Torch device; fit shared host CDFs and regions."""

        _check_control(control)
        config = configuration or self.configuration
        projection_weight, projection_bias, classifier_weight, classifier_bias = (
            np.ascontiguousarray(value, dtype=np.float32) for value in parameters
        )
        tensors = tuple(self._torch.tensor(value, dtype=self._torch.float32, device=self._device)
                        for value in parameters)
        train_logits, train_exemplars = _torch_forward_numpy(
            self._torch, normalized_train, *tensors, self._device, self._matching_query_batch_size,
            control=control,
        )
        calibration_logits, calibration_exemplars = _torch_forward_numpy(
            self._torch, normalized_calibration, *tensors, self._device, self._matching_query_batch_size,
            control=control,
        )
        train_predictions = np.argmax(train_logits, axis=1)
        calibration_predictions = np.argmax(calibration_logits, axis=1)
        support_index = TorchExactL2Index(train_exemplars, device=str(self._device),
            query_batch_size=self._matching_query_batch_size,
            support_tile_size=self._matching_support_tile_size, training_control=control)
        calibration_q, calibration_d0 = _q_d0_many(support_index, calibration_exemplars,
            calibration_predictions, train_labels, train_predictions, config.max_neighbors, False)
        calibration_cdfs = _distance_cdfs(
            calibration_labels,
            calibration_q,
            calibration_d0,
            config.number_of_classes,
            config.ood_limit,
        )
        calibration_d = _distances_from_cdfs(calibration_d0, calibration_cdfs)
        canonical_loss = _balanced_loss(
            calibration_logits,
            calibration_labels,
            calibration_q,
            calibration_d,
            config.q_offset,
            config.number_of_classes,
        )
        calibration_probabilities = _probabilities(
            calibration_logits, calibration_q, calibration_d, config.q_offset
        )
        calibration_q_prime = rescaled_similarities(
            calibration_q,
            calibration_probabilities[np.arange(len(calibration_q)), calibration_predictions],
            config.q_offset,
        )
        q_prime_cdfs = tuple(
            tuple(
                float(value)
                for value in np.sort(calibration_q_prime[calibration_labels == label])
            )
            for label in range(config.number_of_classes)
        )
        _check_control(control)
        regions = NestedCalibrator(
            config.number_of_classes,
            config.alpha_resolution,
            config.ood_limit,
        ).fit(
            calibration_probabilities,
            calibration_q_prime,
            calibration_labels,
            calibration_predictions,
        )
        support_records = tuple(
            SupportRecord(
                id=train_ids[row],
                label=int(train_labels[row]),
                predicted_label=int(train_predictions[row]),
            )
            for row in range(len(train_ids))
        )
        calibration_rows = tuple(
            {
                "id": calibration_ids[row],
                "label": int(calibration_labels[row]),
                "prediction": int(calibration_predictions[row]),
                "sdm": [float(value) for value in calibration_probabilities[row]],
                "qPrime": float(calibration_q_prime[row]),
                "q": int(calibration_q[row]),
                "d0": float(calibration_d0[row]),
                "d": float(calibration_d[row]),
                "zPrime": [float(value) for value in calibration_logits[row]],
            }
            for row in range(len(calibration_ids))
        )
        artifact = build_artifact(
            weights=AdapterWeights(
                projection_weight,
                projection_bias,
                classifier_weight,
                classifier_bias,
            ),
            support_vectors=train_exemplars,
            support_records=support_records,
            distance_cdfs=calibration_cdfs,
            rescaled_similarity_cdfs=q_prime_cdfs,
            regions=regions,
            embedding_dimension=input_dimension,
            exemplar_dimension=config.exemplar_dimension,
            number_of_classes=config.number_of_classes,
            class_names=class_names,
            max_neighbors=config.max_neighbors,
            q_offset=config.q_offset,
            ood_limit=config.ood_limit,
            alpha_resolution=config.alpha_resolution,
            normalization_mean=normalization_mean,
            normalization_standard_deviation=normalization_standard_deviation,
            representation_provider=representation_provider,
            representation_model=representation_model,
            representation_fingerprint=representation_fingerprint,
            calibration_rows=calibration_rows,
            metadata={
                "trainingBackend": "pytorch",
                "trainingDevice": str(self._device),
                "artifactFinalizationBackend": f"torch:{self._device}",
                "bestEpoch": best_epoch,
                "bestBalancedCalibrationSDMLoss": accelerated_selection_loss,
                "canonicalFinalBalancedCalibrationSDMLoss": canonical_loss,
                "acceleratedSelectionBalancedCalibrationSDMLoss": accelerated_selection_loss,
            },
        )
        _check_control(control)
        return TrainingResult(
            artifact=artifact,
            best_epoch=best_epoch,
            best_balanced_calibration_loss=accelerated_selection_loss,
            history=tuple(history),
        )
