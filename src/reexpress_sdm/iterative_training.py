# Copyright Reexpress AI, Inc. All rights reserved.
"""Backend-neutral independent training over J optional split shuffles.

This orchestration follows the
research/app lifecycle: optionally pool and uniformly split in half, train a
fresh adaptor, then retain the lowest balanced calibration SDM loss (last tie).
Each iteration uses fresh Adam state and either fresh weights or the same
provided initial artifact, including its saved normalization.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .backends import create_training_backend
from .errors import DimensionMismatchError
from .training import TrainingConfig, TrainingResult, TrainingControl, _check_control, _representation_defaults, _starting_model, _validate_training_arrays
from .types import SDMArtifact


@dataclass(frozen=True)
class IterativeTrainingResult(TrainingResult):
    """Selected model plus all-iteration history and exact source membership.

    Pool indices are zero-based in ``[original training; original calibration]``.
    Iteration and epoch numbers are one-based. IDs survive reordering and are
    also written to artifact metadata for non-Python consumers.
    """

    best_iteration: int
    number_of_iterations: int
    training_pool_indices: tuple[int, ...]
    calibration_pool_indices: tuple[int, ...]


def train_iterations(
    configuration: TrainingConfig,
    train_vectors: np.ndarray,
    train_labels: Sequence[int],
    calibration_vectors: np.ndarray,
    calibration_labels: Sequence[int],
    *,
    representation_fingerprint: str | None = None,
    number_of_random_shuffles: int = 1,
    shuffle_training_and_calibration: bool = True,
    backend: str = "torch",
    backend_options: Mapping[str, Any] | None = None,
    train_ids: Sequence[str] | None = None,
    calibration_ids: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    representation_provider: str | None = None,
    representation_model: str | None = None,
    initial_artifact: SDMArtifact | None = None,
    control: TrainingControl | None = None,
    progress: Callable[[Mapping[str, float | int | bool | None]], None] | None = None,
) -> IterativeTrainingResult:
    """Train J independent adaptors, returning the selected portable artifact.

    By default, pool and shuffle the supplied training/calibration rows before
    every iteration, including iteration 1, then split the pool in half.
    Set ``shuffle_training_and_calibration=False`` to retain the supplied
    memberships and sizes. Odd pools put the extra row in calibration.
    Iteration ``i`` (zero-based) uses ``(configuration.seed + i) % 2**64`` both for its
    independent NumPy permutation and its fresh backend training seed. Seeds
    are deterministic within a backend, not bit-identical to Swift/PyTorch's
    research RNG stream. If either shuffled half omits a class, fail explicitly.
    ``duration_seconds`` includes all completed iterations, split preparation,
    and their final calibration, but excludes artifact export.
    Unspecified representation metadata comes from the initial model, or uses
    fingerprint ``embedding_v1``, provider ``precomputed``, and no model name.
    """

    started = perf_counter()
    if (
        isinstance(number_of_random_shuffles, (bool, np.bool_))
        or not isinstance(number_of_random_shuffles, (int, np.integer))
        or number_of_random_shuffles < 1
    ):
        raise ValueError("number_of_random_shuffles must be a positive integer")
    if not isinstance(shuffle_training_and_calibration, bool):
        raise ValueError("shuffle_training_and_calibration must be a bool")
    train_vectors, train_labels, calibration_vectors, calibration_labels = _validate_training_arrays(
        train_vectors, train_labels, calibration_vectors, calibration_labels,
        configuration.number_of_classes,
    )
    _check_control(control)
    representation_fingerprint, representation_provider, representation_model = _representation_defaults(
        representation_fingerprint, representation_provider, representation_model, initial_artifact)
    configuration, class_names, _ = _starting_model(
        configuration, initial_artifact, train_vectors.shape[1], class_names, representation_fingerprint)
    train_ids = tuple(train_ids) if train_ids is not None else tuple(
        f"train-{index}" for index in range(len(train_labels))
    )
    calibration_ids = tuple(calibration_ids) if calibration_ids is not None else tuple(
        f"calibration-{index}" for index in range(len(calibration_labels))
    )
    if len(train_ids) != len(train_labels) or len(calibration_ids) != len(calibration_labels):
        raise DimensionMismatchError("ids must align with their split")
    ids = train_ids + calibration_ids
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("ids must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("training and calibration ids must be unique and must not overlap")

    # Do not allocate a second complete feature matrix for the ordinary
    # unshuffled path. For J shuffles the one pooled matrix is reused.
    pooled_vectors = np.concatenate((train_vectors, calibration_vectors)) if shuffle_training_and_calibration else None
    pooled_labels = np.concatenate((train_labels, calibration_labels)) if shuffle_training_and_calibration else None
    total = len(ids)
    selected: TrainingResult | None = None
    selected_iteration = 1
    selected_train_indices: tuple[int, ...] = ()
    selected_calibration_indices: tuple[int, ...] = ()
    history: list[Mapping[str, float | int | bool | None]] = []
    iteration_summaries: list[dict[str, Any]] = []

    for iteration_index in range(number_of_random_shuffles):
        _check_control(control)
        if selected is not None and control is not None and control.is_stop_requested:
            break
        iteration = iteration_index + 1
        seed = (int(configuration.seed) + iteration_index) % 2**64
        if shuffle_training_and_calibration:
            order = np.random.default_rng(seed).permutation(total)
            train_indices = order[:total // 2]
            calibration_indices = order[total // 2:]
            iteration_train_vectors = pooled_vectors[train_indices]
            iteration_train_labels = pooled_labels[train_indices]
            iteration_calibration_vectors = pooled_vectors[calibration_indices]
            iteration_calibration_labels = pooled_labels[calibration_indices]
            if any(
                set(labels.tolist()) != set(range(configuration.number_of_classes))
                for labels in (iteration_train_labels, iteration_calibration_labels)
            ):
                raise ValueError(
                    f"shuffle iteration {iteration} omitted a class from a split; "
                    "increase data per class, choose a different seed, or disable split shuffling"
                )
        else:
            train_indices = np.arange(len(train_ids))
            calibration_indices = np.arange(len(train_ids), total)
            iteration_train_vectors, iteration_train_labels = train_vectors, train_labels
            iteration_calibration_vectors, iteration_calibration_labels = calibration_vectors, calibration_labels

        if selected is not None and control is not None and control.is_stop_requested:
            break

        def report(values: Mapping[str, float | int | bool | None]) -> None:
            if progress is not None:
                progress({**values, "iteration": iteration, "iterations": int(number_of_random_shuffles)})

        trainer = create_training_backend(backend, replace(configuration, seed=seed), **dict(backend_options or {}))
        result = trainer.fit(
            iteration_train_vectors, iteration_train_labels,
            iteration_calibration_vectors, iteration_calibration_labels,
            representation_fingerprint=representation_fingerprint,
            train_ids=tuple(ids[int(index)] for index in train_indices),
            calibration_ids=tuple(ids[int(index)] for index in calibration_indices),
            class_names=class_names,
            representation_provider=representation_provider,
            representation_model=representation_model,
            progress=report,
            initial_artifact=initial_artifact,
            control=control,
        )
        history.extend({**row, "iteration": iteration, "iterations": int(number_of_random_shuffles)} for row in result.history)
        iteration_summaries.append({
            "iteration": iteration,
            "seed": seed,
            "bestEpoch": result.best_epoch,
            "bestBalancedCalibrationSDMLoss": result.best_balanced_calibration_loss,
            "durationSeconds": result.duration_seconds,
        })
        if selected is None or result.best_balanced_calibration_loss <= selected.best_balanced_calibration_loss:
            selected = result
            selected_iteration = iteration
            selected_train_indices = tuple(int(index) for index in train_indices)
            selected_calibration_indices = tuple(int(index) for index in calibration_indices)

    _check_control(control)
    assert selected is not None
    completed_iterations = len(iteration_summaries)
    stopped_early = bool(control and control.is_stop_requested and (
        completed_iterations < number_of_random_shuffles or len(result.history) < configuration.epochs))
    metadata = dict(selected.artifact.manifest.get("metadata", {}))
    metadata.update({
        "selectedTrainingIteration": selected_iteration,
        "trainingIterations": completed_iterations,
        "requestedTrainingIterations": int(number_of_random_shuffles),
        "stoppedEarly": stopped_early,
        "shuffledTrainingAndCalibration": shuffle_training_and_calibration,
        "iterationSelectionCriterion": "lowestBalancedCalibrationSDMLossLastTie",
        "iterationSeeds": [item["seed"] for item in iteration_summaries],
        "iterationSummaries": iteration_summaries,
        "bestIterationSplits": {
            "indexConvention": "zero-based in original training followed by original calibration",
            "originalTrainingCount": len(train_ids),
            "originalCalibrationCount": len(calibration_ids),
            "trainingPoolIndices": list(selected_train_indices),
            "calibrationPoolIndices": list(selected_calibration_indices),
            "trainingIDs": [ids[index] for index in selected_train_indices],
            "calibrationIDs": [ids[index] for index in selected_calibration_indices],
        },
    })
    from .training_metadata import build_training_run
    duration_seconds = perf_counter() - started
    metadata["trainingRun"] = build_training_run(
        configuration, history, best_epoch=selected.best_epoch,
        best_balanced_calibration_loss=selected.best_balanced_calibration_loss,
        backend=metadata.get("trainingRun", {}).get("backend", backend),
        best_iteration=selected_iteration, completed_iterations=completed_iterations,
        requested_iterations=int(number_of_random_shuffles),
        shuffle_training_and_calibration=shuffle_training_and_calibration,
        source_model_id=initial_artifact.model_id if initial_artifact is not None else None,
        stopped_early=stopped_early,
        duration_seconds=duration_seconds,
    )
    artifact = replace(selected.artifact, manifest={**selected.artifact.manifest, "metadata": metadata})
    return IterativeTrainingResult(
        artifact=artifact,
        best_epoch=selected.best_epoch,
        best_balanced_calibration_loss=selected.best_balanced_calibration_loss,
        history=tuple(history),
        best_iteration=selected_iteration,
        number_of_iterations=completed_iterations,
        stopped_early=stopped_early,
        duration_seconds=duration_seconds,
        training_pool_indices=selected_train_indices,
        calibration_pool_indices=selected_calibration_indices,
    )
