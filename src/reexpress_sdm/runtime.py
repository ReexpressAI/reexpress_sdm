# Copyright Reexpress AI, Inc. All rights reserved.
"""Device-resident adaptor execution shared by portable model operations.

Only host interchange and final score statistics use NumPy. Weights stay on
the selected device for the lifetime of a model, and projected query tensors
flow directly into a compatible device index during scoring.
"""
from __future__ import annotations

import numpy as np

from .errors import DimensionMismatchError
from .index import _positive_integer


def input_matrix(values, dimension: int) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except ValueError as error:
        raise DimensionMismatchError("embeddings must form a rectangular numeric matrix") from error
    if raw.dtype.kind not in "iuf":
        raise ValueError("embeddings must contain numeric values, excluding booleans and strings")
    with np.errstate(over="ignore", invalid="ignore"):
        result = raw.astype(np.float32, copy=False)
    if result.ndim == 1:
        result = result.reshape(1, -1)
    if result.ndim != 2 or result.shape[1] != dimension:
        raise DimensionMismatchError(f"embeddings must have shape [N, {dimension}]")
    if not np.all(np.isfinite(result)):
        raise ValueError("embeddings contain a non-finite value")
    return np.ascontiguousarray(result)


class TorchRuntime:
    backend = "torch"

    def __init__(self, artifact, *, device="auto", query_batch_size=256):
        from .torch_backend import _load_torch, _resolve_device
        self._torch = _load_torch()
        self._device = _resolve_device(self._torch, device)
        self.device = str(self._device)
        self.query_batch_size = _positive_integer(query_batch_size, "query_batch_size")
        self.dimension = int(artifact.configuration["embeddingDimension"])
        self.exemplar_dimension = int(artifact.configuration["exemplarDimension"])
        self.classes = int(artifact.configuration["numberOfClasses"])
        normalization = artifact.manifest["normalization"]
        self._mean = np.float32(normalization["mean"])
        self._std = np.float32(normalization["standardDeviation"])
        weights = artifact.weights
        # tensor copies avoid aliasing mutable host artifact arrays on CPU.
        self.weights = tuple(self._torch.tensor(np.asarray(value), dtype=self._torch.float32, device=self._device)
                             for value in (weights.projection_weight, weights.projection_bias,
                                           weights.classifier_weight, weights.classifier_bias))

    def inputs(self, values):
        if not isinstance(values, self._torch.Tensor):
            return input_matrix(values, self.dimension)
        torch = self._torch
        if values.dtype == torch.bool or values.is_complex() or values.is_quantized:
            raise ValueError("embeddings must contain real numeric values, excluding booleans")
        values = values.detach().to(device=self._device, dtype=torch.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise DimensionMismatchError(f"embeddings must have shape [N, {self.dimension}]")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("embeddings contain a non-finite value")
        return values

    def project_batches(self, values: np.ndarray):
        torch = self._torch
        projection_weight, projection_bias, classifier_weight, classifier_bias = self.weights
        with torch.inference_mode(), torch.autocast(device_type=self._device.type, enabled=False):
            for start in range(0, len(values), self.query_batch_size):
                batch_values = values[start:start + self.query_batch_size]
                if isinstance(batch_values, np.ndarray) and not batch_values.flags.writeable:
                    # Preserve the mapped dataset; give Torch writable storage for this batch only.
                    batch_values = batch_values.copy()
                batch = torch.as_tensor(batch_values, dtype=torch.float32, device=self._device)
                normalized = (batch - float(self._mean)) / float(self._std)
                exemplars = normalized @ projection_weight.T + projection_bias
                logits = exemplars @ classifier_weight.T + classifier_bias
                # Scoring queues matching before reading either output. Reading
                # logits here would force an extra device synchronization between
                # projection and matching, especially costly for one query.
                yield logits, exemplars

    @staticmethod
    def host(values):
        return values.detach().cpu().numpy().astype(np.float32, copy=False)

    def transform(self, values):
        logits, exemplars = [], []
        for batch_logits, batch_exemplars in self.project_batches(values):
            host_logits, host_exemplars = self.host(batch_logits), self.host(batch_exemplars)
            if not np.all(np.isfinite(host_logits)) or not np.all(np.isfinite(host_exemplars)):
                raise ValueError("adaptor produced a non-finite exemplar or logit")
            logits.append(host_logits)
            exemplars.append(host_exemplars)
        return (np.concatenate(logits) if logits else np.empty((0, self.classes), np.float32),
                np.concatenate(exemplars) if exemplars else np.empty((0, self.exemplar_dimension), np.float32))


def create_runtime(artifact, *, backend="torch", device="auto", query_batch_size=256):
    if not isinstance(backend, str):
        raise ValueError("inference backend must be 'torch'")
    normalized = backend.strip().lower()
    runtime = {"torch": TorchRuntime}.get(normalized)
    if runtime is None:
        raise ValueError("inference backend must be 'torch'")
    return runtime(artifact, device=device, query_batch_size=query_batch_size)
