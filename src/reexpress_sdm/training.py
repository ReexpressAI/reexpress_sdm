# Copyright Reexpress AI, Inc. All rights reserved.
"""Shared training lifecycle, host calibration math, and portable artifacts.

PyTorch owns adaptor optimization, forward passes, and dense matching. These
helpers keep checkpoint selection and the portable calibration contract shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import math
import threading
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence
import uuid

import numpy as np

from .types import AdapterWeights, Region, SDMArtifact, SupportRecord
from .errors import DimensionMismatchError
from .math import _validate_alpha_resolution, distance_quantiles, stable_softmax


@dataclass(frozen=True)
class TrainingConfig:
    number_of_classes: int
    exemplar_dimension: int = 1000
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1.0e-5
    seed: int = 0
    max_neighbors: int = 2048
    q_offset: float = 2.0
    ood_limit: int = 0
    alpha_resolution: float = 0.05
    cross_entropy_epochs: int = 1

    def __post_init__(self) -> None:
        for name in (
            "number_of_classes",
            "exemplar_dimension",
            "epochs",
            "batch_size",
            "seed",
            "max_neighbors",
            "ood_limit",
            "cross_entropy_epochs",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.number_of_classes < 2:
            raise ValueError("number_of_classes must be at least two")
        if self.exemplar_dimension < 1 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("dimensions, epochs, and batch_size must be positive")
        if not 1 <= self.cross_entropy_epochs <= self.epochs:
            raise ValueError("cross_entropy_epochs must be in 1...epochs")
        if not 0 <= self.seed < 2**64:
            raise ValueError("seed must be a nonnegative UInt64")
        if self.max_neighbors < 2:
            raise ValueError("max_neighbors must be at least two for identity exclusion")
        if self.ood_limit < 0:
            raise ValueError("ood_limit must be nonnegative")
        if (
            isinstance(self.learning_rate, (bool, np.bool_))
            or not isinstance(self.learning_rate, (int, float, np.integer, np.floating))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError("learning_rate must be a finite positive number")
        with np.errstate(over="ignore", under="ignore"):
            float32_rate = np.float32(self.learning_rate)
        if not np.isfinite(float32_rate) or float32_rate <= 0:
            raise ValueError("learning_rate must remain finite and positive as Float32")
        if (
            isinstance(self.q_offset, (bool, np.bool_))
            or not isinstance(self.q_offset, (int, float, np.integer, np.floating))
            or not math.isfinite(self.q_offset)
            or self.q_offset <= 1.0
        ):
            raise ValueError("q_offset must be greater than one")
        if self.q_offset > math.e:
            raise ValueError("training requires q_offset <= e for its first CE-equivalent epoch")
        _validate_alpha_resolution(self.alpha_resolution)


@dataclass(frozen=True)
class TrainingResult:
    artifact: SDMArtifact
    best_epoch: int
    best_balanced_calibration_loss: float
    history: tuple[Mapping[str, float | int | bool | None], ...]
    stopped_early: bool = field(default=False, kw_only=True)
    duration_seconds: float | None = field(default=None, kw_only=True)


class TrainingCancelled(RuntimeError):
    """The attempt was cancelled; no selected artifact is returned."""


class TrainingControl:
    """Thread-safe control shared by all J iterations of one training attempt.

    request_stop finishes the current epoch and final scoring. Before any
    complete epoch it requests cancellation instead and returns False. cancel
    always interrupts at the next batch/scoring boundary, including finalization.
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._completed = 0
        self._stop = False
        self._cancelled = False

    def request_stop(self) -> bool:
        with self._lock:
            if not self._completed:
                self._cancelled = True
                return False
            self._stop = True
            return True

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True

    @property
    def is_stop_requested(self) -> bool:
        with self._lock:
            return self._stop

    @property
    def completed_epoch_count(self) -> int:
        with self._lock:
            return self._completed

    def check_cancelled(self) -> None:
        with self._lock:
            if self._cancelled:
                raise TrainingCancelled("training was cancelled")

    def _complete_epoch(self) -> None:
        with self._lock:
            if self._cancelled:
                raise TrainingCancelled("training was cancelled")
            self._completed += 1


def _check_control(control: TrainingControl | None) -> None:
    if control is not None:
        control.check_cancelled()


def _balanced_mean(values: np.ndarray, labels: np.ndarray, classes: int) -> float:
    """Mean of per-true-class means. Training validation requires every class."""
    counts = np.bincount(labels, minlength=classes)
    if np.any(counts == 0):
        raise ValueError("balanced training metrics require every class")
    return float(np.mean(np.bincount(labels, weights=values, minlength=classes) / counts))


def _representation_defaults(fingerprint, provider, model, initial_artifact):
    """Resolve unspecified provenance without replacing a saved model's identity."""
    saved = {}
    if initial_artifact is not None:
        from .artifact import validate_artifact
        saved = validate_artifact(initial_artifact).manifest["representation"]
    return (
        fingerprint if fingerprint is not None else saved.get("fingerprint", "embedding_v1"),
        provider if provider is not None else saved.get("provider", "precomputed"),
        model if model is not None else saved.get("model"),
    )


def _starting_model(config, artifact, dimension, class_names, fingerprint):
    if artifact is None:
        return config, class_names, None
    from .artifact import validate_artifact
    artifact = validate_artifact(artifact)
    saved = artifact.manifest["configuration"]
    if (saved["numberOfClasses"] != config.number_of_classes
            or saved["embeddingDimension"] != dimension
            or artifact.manifest["representation"]["fingerprint"] != fingerprint
            or (class_names is not None and list(class_names) != saved["classNames"])):
        raise ValueError("continuation requires the saved class order, feature dimension, and representation fingerprint")
    config = replace(config, exemplar_dimension=saved["exemplarDimension"],
                     q_offset=saved["qOffset"], ood_limit=saved["oodLimit"])
    return config, tuple(saved["classNames"]), artifact.manifest["normalization"]


def _weight_arrays(weights: AdapterWeights) -> tuple[np.ndarray, ...]:
    return tuple(np.array(value, dtype=np.float32, copy=True) for value in (
        weights.projection_weight, weights.projection_bias, weights.classifier_weight, weights.classifier_bias
    ))


class _EpochEvaluator:
    """Shared measurements; forwards and matching remain batched by each backend."""
    def __init__(self, config, train_labels, calibration_labels, forward, index_factory, match, control):
        self.config, self.train_labels, self.calibration_labels = config, train_labels, calibration_labels
        self.forward, self.index_factory, self.match, self.control = forward, index_factory, match, control

    def measure(self, parameters=None, *, ce=False, training_only=False):
        _check_control(self.control)
        config = self.config
        train_logits, train_exemplars = self.forward(False, parameters, logits_only=ce)
        train_predictions = np.argmax(train_logits, axis=1)
        if ce:
            train_q = np.full(len(self.train_labels), np.float32(math.e - config.q_offset))
            train_d = np.ones(len(self.train_labels), dtype=np.float32)
            index = None
        else:
            index = self.index_factory(train_exemplars)
            train_q, train_d0 = self.match(index, train_exemplars, train_predictions,
                                         self.train_labels, train_predictions, config.max_neighbors, True)
            train_d = _distances_from_cdfs(train_d0, _distance_cdfs(
                self.train_labels, train_q, train_d0, config.number_of_classes, config.ood_limit))
        _check_control(self.control)
        if training_only:
            return {}, train_q, train_d
        calibration_logits, calibration_exemplars = self.forward(True, parameters, logits_only=ce)
        calibration_predictions = np.argmax(calibration_logits, axis=1)
        if ce:
            calibration_q = np.full(len(self.calibration_labels), np.float32(math.e - config.q_offset))
            calibration_d = np.ones(len(self.calibration_labels), dtype=np.float32)
        else:
            calibration_q, calibration_d0 = self.match(index, calibration_exemplars, calibration_predictions,
                                                      self.train_labels, train_predictions, config.max_neighbors, False)
            calibration_d = _distances_from_cdfs(calibration_d0, _distance_cdfs(
                self.calibration_labels, calibration_q, calibration_d0, config.number_of_classes, config.ood_limit))
        _check_control(self.control)
        values = {
            "balancedTrainingAccuracy": _balanced_mean(train_predictions == self.train_labels, self.train_labels, config.number_of_classes),
            "balancedCalibrationAccuracy": _balanced_mean(calibration_predictions == self.calibration_labels, self.calibration_labels, config.number_of_classes),
            "balancedTrainingSDMLoss": None, "balancedCalibrationSDMLoss": None,
            "balancedTrainingCELoss": None, "balancedCalibrationCELoss": None,
            "balancedMeanTrainingQ": None, "balancedMeanCalibrationQ": None,
        }
        suffix = "CE" if ce else "SDM"
        values[f"balancedTraining{suffix}Loss"] = _balanced_loss(train_logits, self.train_labels, train_q, train_d, config.q_offset, config.number_of_classes)
        values[f"balancedCalibration{suffix}Loss"] = _balanced_loss(calibration_logits, self.calibration_labels, calibration_q, calibration_d, config.q_offset, config.number_of_classes)
        if not ce:
            values["balancedMeanTrainingQ"] = _balanced_mean(train_q, self.train_labels, config.number_of_classes)
            values["balancedMeanCalibrationQ"] = _balanced_mean(calibration_q, self.calibration_labels, config.number_of_classes)
        _check_control(self.control)
        return values, train_q, train_d


def _run_epochs(config, train_count, train_epoch, snapshot, evaluator, progress, control):
    """Select complete checkpoints without comparing CE and SDM loss scales.

    Epoch duration measures optimization, evaluation and checkpoint selection.
    Deferred SDM scoring is charged to the CE winner whose metrics it fills in;
    callbacks and transition-only q/d preparation are covered by total fit time.
    """
    ce_q = np.full(train_count, np.float32(math.e - config.q_offset))
    ce_d = np.ones(train_count, dtype=np.float32)
    train_q, train_d = ce_q, ce_d
    history = []
    best_loss, best_epoch, best_parameters = math.inf, 0, None
    ce_loss, ce_epoch, ce_parameters = math.inf, 0, None

    def score_ce_winner():
        nonlocal best_loss, best_epoch, best_parameters
        scoring_started = perf_counter()
        values, q, d = evaluator.measure(None if ce_epoch == len(history) else ce_parameters)
        best_loss, best_epoch, best_parameters = values["balancedCalibrationSDMLoss"], ce_epoch, ce_parameters
        updated = dict(history[ce_epoch - 1])
        updated.update({key: value for key, value in values.items() if value is not None})
        updated["isBest"] = True
        updated["durationSeconds"] += perf_counter() - scoring_started
        history[ce_epoch - 1] = updated
        if progress is not None:
            progress(dict(updated))
        _check_control(control)
        return q, d

    for epoch in range(1, config.epochs + 1):
        _check_control(control)
        if history and control is not None and control.is_stop_requested:
            break
        epoch_started = perf_counter()
        uses_ce = epoch <= config.cross_entropy_epochs
        marginal_loss = train_epoch(ce_q if uses_ce else train_q, ce_d if uses_ce else train_d)
        _check_control(control)
        deferred = uses_ce and config.cross_entropy_epochs > 1
        values, train_q, train_d = evaluator.measure(ce=deferred)
        _check_control(control)
        if deferred:
            loss = values["balancedCalibrationCELoss"]
            is_best = loss <= ce_loss
            if is_best:
                ce_loss, ce_epoch, ce_parameters = loss, epoch, snapshot()
        else:
            loss = values["balancedCalibrationSDMLoss"]
            is_best = loss <= best_loss
            if is_best:
                best_loss, best_epoch, best_parameters = loss, epoch, snapshot()
        values.update(epoch=epoch, isBest=is_best, trainingLoss=marginal_loss,
                      durationSeconds=perf_counter() - epoch_started)
        history.append(values)
        if control is not None:
            control._complete_epoch()
        if progress is not None:
            progress(dict(values))
        _check_control(control)
        if deferred and (epoch == config.cross_entropy_epochs or (control is not None and control.is_stop_requested)):
            train_q, train_d = score_ce_winner()
            if epoch < config.epochs and not (control is not None and control.is_stop_requested) and ce_epoch != epoch:
                # Prepare only training q/d at the last CE weights. Its calibration
                # loss is not another candidate, and the live optimizer is untouched.
                _, train_q, train_d = evaluator.measure(training_only=True)
    if best_parameters is None and ce_parameters is not None:
        score_ce_winner()
    _check_control(control)
    assert best_parameters is not None
    return best_parameters, best_epoch, best_loss, history


def _finish_training_result(result, config, initial_artifact, control, *, duration_seconds=None, source_data=None):
    _check_control(control)
    metadata = dict(result.artifact.manifest.get("metadata", {}))
    stopped = bool(control and control.is_stop_requested and len(result.history) < config.epochs)
    from .training_metadata import build_training_run
    backend = "torch:" + metadata.get("trainingDevice", "cpu")
    metadata.update(crossEntropyEpochs=config.cross_entropy_epochs, stoppedEarly=stopped)
    if source_data is not None:
        metadata["sourceData"] = source_data
    metadata["trainingRun"] = build_training_run(
        config, result.history, best_epoch=result.best_epoch,
        best_balanced_calibration_loss=result.best_balanced_calibration_loss,
        backend=backend, source_model_id=initial_artifact.model_id if initial_artifact is not None else None,
        stopped_early=stopped,
        duration_seconds=duration_seconds,
    )
    manifest = dict(result.artifact.manifest)
    if initial_artifact is not None:
        manifest["representation"] = dict(initial_artifact.manifest["representation"])
        metadata.update(continuedFromModelID=initial_artifact.model_id, optimizerRestarted=True, normalizationPreserved=True)
    manifest["metadata"] = metadata
    return replace(result, artifact=replace(result.artifact, manifest=manifest),
                   stopped_early=stopped, duration_seconds=duration_seconds)


class TrainingBackend(Protocol):
    def fit(self, *args: Any, **kwargs: Any) -> TrainingResult: ...


def _validate_training_arrays(
    train_vectors: np.ndarray,
    train_labels: np.ndarray,
    calibration_vectors: np.ndarray,
    calibration_labels: np.ndarray,
    classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        raw_train_vectors = np.asarray(train_vectors)
        raw_calibration_vectors = np.asarray(calibration_vectors)
        raw_train_labels = np.asarray(train_labels)
        raw_calibration_labels = np.asarray(calibration_labels)
    except ValueError as error:
        raise DimensionMismatchError("training inputs must be rectangular arrays") from error
    if raw_train_vectors.dtype.kind not in "iuf" or raw_calibration_vectors.dtype.kind not in "iuf":
        raise ValueError("training and calibration vectors must contain numeric values")
    if raw_train_labels.dtype.kind not in "iu" or raw_calibration_labels.dtype.kind not in "iu":
        raise ValueError("training and calibration labels must contain actual integers")
    with np.errstate(over="ignore", invalid="ignore"):
        train_vectors = raw_train_vectors.astype(np.float32, copy=False)
        calibration_vectors = raw_calibration_vectors.astype(np.float32, copy=False)
    train_labels = raw_train_labels.astype(np.int64, copy=False).reshape(-1)
    calibration_labels = raw_calibration_labels.astype(np.int64, copy=False).reshape(-1)
    if train_vectors.ndim != 2 or calibration_vectors.ndim != 2:
        raise DimensionMismatchError("training and calibration vectors must be matrices")
    if train_vectors.shape[1] != calibration_vectors.shape[1]:
        raise DimensionMismatchError("training and calibration dimensions differ")
    if train_vectors.shape[0] != train_labels.size or calibration_vectors.shape[0] != calibration_labels.size:
        raise DimensionMismatchError("vectors and labels do not align")
    if train_labels.size == 0 or calibration_labels.size == 0:
        raise ValueError("training and calibration splits cannot be empty")
    if not np.all(np.isfinite(train_vectors)) or not np.all(np.isfinite(calibration_vectors)):
        raise ValueError("training data contains non-finite values")
    for name, labels in (("training", train_labels), ("calibration", calibration_labels)):
        if np.any(labels < 0) or np.any(labels >= classes):
            raise ValueError(f"{name} labels must be known classes")
        if any(np.sum(labels == label) == 0 for label in range(classes)):
            raise ValueError(f"{name} split must contain every class")
    return train_vectors, train_labels, calibration_vectors, calibration_labels


def _distance_cdfs(
    labels: np.ndarray,
    q_values: np.ndarray,
    d0_values: np.ndarray,
    classes: int,
    ood_limit: int,
) -> tuple[tuple[float, ...], ...]:
    result: list[tuple[float, ...]] = []
    for label in range(classes):
        selected = d0_values[(labels == label) & (q_values > ood_limit)]
        result.append(tuple(float(value) for value in np.sort(selected)))
    return tuple(result)


def _distances_from_cdfs(d0_values: np.ndarray, cdfs: Sequence[Sequence[float]]) -> np.ndarray:
    return distance_quantiles(np.asarray(d0_values, dtype=np.float32), cdfs)


def _probabilities(logits: np.ndarray, q: np.ndarray, d: np.ndarray, q_offset: float) -> np.ndarray:
    scale = d[:, None] * np.log(q[:, None] + np.float32(q_offset))
    return stable_softmax(logits * scale).astype(np.float32)


def _balanced_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    q: np.ndarray,
    d: np.ndarray,
    q_offset: float,
    classes: int,
) -> float:
    log_base = np.log(q + np.float32(q_offset)).astype(np.float32)
    scaled = logits * d[:, None] * log_base[:, None]
    maxima = np.max(scaled, axis=1, keepdims=True)
    shifted = scaled - maxima
    log_sum_exp_shifted = np.log(np.sum(np.exp(shifted), axis=1, dtype=np.float32))
    losses = (log_sum_exp_shifted - shifted[np.arange(labels.size), labels]) / log_base
    result = float(np.mean([np.mean(losses[labels == label]) for label in range(classes)]))
    if not math.isfinite(result):
        raise ValueError("balanced calibration loss is non-finite")
    return result


def build_artifact(
    *,
    weights: AdapterWeights,
    support_vectors: np.ndarray,
    support_records: Sequence[SupportRecord],
    distance_cdfs: Sequence[Sequence[float]],
    rescaled_similarity_cdfs: Sequence[Sequence[float]],
    regions: Sequence[Region],
    embedding_dimension: int,
    exemplar_dimension: int,
    number_of_classes: int,
    class_names: Sequence[str] | None = None,
    max_neighbors: int = 2048,
    q_offset: float = 2.0,
    ood_limit: int = 0,
    alpha_resolution: float = 0.05,
    normalization_mean: float = 0.0,
    normalization_standard_deviation: float = 1.0,
    representation_provider: str = "precomputed",
    representation_model: str | None = None,
    representation_revision: str | None = None,
    representation_input_template: str | None = None,
    representation_fingerprint: str = "embedding_v1",
    model_id: str | None = None,
    producer_name: str = "reexpress_sdm",
    producer_version: str = "0.4.5",
    metadata: Mapping[str, Any] | None = None,
    calibration_rows: Sequence[Mapping[str, Any]] | None = None,
) -> SDMArtifact:
    names = list(class_names) if class_names is not None else [
        f"Class{index}" for index in range(number_of_classes)
    ]
    weight_count = (
        exemplar_dimension * embedding_dimension
        + exemplar_dimension
        + number_of_classes * exemplar_dimension
        + number_of_classes
    )
    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "modelID": model_id or str(uuid.uuid4()),
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "producer": {"name": producer_name, "version": producer_version},
        "configuration": {
            "numberOfClasses": number_of_classes,
            "classNames": names,
            "embeddingDimension": embedding_dimension,
            "exemplarDimension": exemplar_dimension,
            "maxNeighbors": max_neighbors,
            "qOffset": q_offset,
            "oodLimit": ood_limit,
            "alphaResolution": alpha_resolution,
            "distanceMetric": "squaredL2",
            "neighborTieBreak": "supportIndexAscending",
        },
        "normalization": {
            "mean": normalization_mean,
            "standardDeviation": normalization_standard_deviation,
        },
        "representation": {
            "provider": representation_provider,
            "model": representation_model,
            "revision": representation_revision,
            "inputTemplate": representation_input_template,
            "fingerprint": representation_fingerprint,
        },
        "weights": {"file": "weights.f32", "elementCount": weight_count},
        "support": {
            "vectorsFile": "support.f32",
            "recordsFile": "support.jsonl",
            "count": len(support_records),
        },
        "distanceCDFs": [[float(x) for x in cdf] for cdf in distance_cdfs],
        "rescaledSimilarityCDFs": [
            [float(x) for x in cdf] for cdf in rescaled_similarity_cdfs
        ],
        "regions": [region.to_manifest_dict() for region in regions],
    }
    if metadata is not None:
        manifest["metadata"] = dict(metadata)
    from .artifact import validate_artifact, validate_manifest

    manifest = validate_manifest(manifest)
    def validated_tensor(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
        raw_value = np.asarray(value)
        if raw_value.shape != shape:
            raise DimensionMismatchError(f"{name} shape {raw_value.shape} != {shape}")
        if raw_value.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain numeric values")
        with np.errstate(over="ignore", invalid="ignore"):
            result = raw_value.astype(np.float32, copy=False)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain finite Float32 values")
        return np.array(result, copy=True)

    validated_weights = AdapterWeights(
        validated_tensor(
            weights.projection_weight,
            (exemplar_dimension, embedding_dimension),
            "projection_weight",
        ),
        validated_tensor(weights.projection_bias, (exemplar_dimension,), "projection_bias"),
        validated_tensor(
            weights.classifier_weight,
            (number_of_classes, exemplar_dimension),
            "classifier_weight",
        ),
        validated_tensor(weights.classifier_bias, (number_of_classes,), "classifier_bias"),
    )
    validated_support_vectors = validated_tensor(
        support_vectors,
        (len(support_records), exemplar_dimension),
        "support_vectors",
    )
    return validate_artifact(
        SDMArtifact(
            manifest=manifest,
            weights=validated_weights,
            support_vectors=validated_support_vectors,
            support_records=tuple(support_records),
            calibration_rows=tuple(calibration_rows) if calibration_rows is not None else None,
        )
    )
