# Copyright Reexpress AI, Inc. All rights reserved.
"""Construct PyTorch training, inference, and matching components."""

from __future__ import annotations

from typing import Any

from .index import DenseIndex
from .training import TrainingBackend, TrainingConfig


def create_training_backend(
    name: str,
    configuration: TrainingConfig,
    **backend_options: Any,
) -> TrainingBackend:
    """Construct the PyTorch trainer for CPU, MPS, or CUDA execution."""

    if not isinstance(name, str):
        raise ValueError("training backend name must be a string")
    normalized = name.strip().lower()
    if normalized in {"torch", "pytorch"}:
        from .torch_backend import TorchTrainer

        return TorchTrainer(configuration, **backend_options)
    raise ValueError("training backend must be 'torch'")


def create_dense_index(
    name: str,
    support_vectors: Any,
    **backend_options: Any,
) -> DenseIndex:
    """Construct an exact dense index without eagerly importing accelerators."""

    if not isinstance(name, str):
        raise ValueError("matching backend name must be a string")
    normalized = name.strip().lower()
    if normalized in {"torch", "pytorch"}:
        from .torch_backend import TorchExactL2Index

        return TorchExactL2Index(support_vectors, **backend_options)
    raise ValueError("matching backend must be 'torch'")
