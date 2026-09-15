# Copyright Reexpress AI, Inc. All rights reserved.
"""Dense matching protocol and the default PyTorch exact index."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class DenseIndex(Protocol):
    @property
    def count(self) -> int: ...

    @property
    def dimension(self) -> int: ...

    def search_one(
        self, query: np.ndarray, k: int, exclude_index: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class ExactL2Index:
    """Default Torch exact index, retaining the convenient generic index API.

    NumPy remains the host-array interchange type; it is not a matching backend.
    Distance calculation and deterministic selection use Torch on ``device``.
    ``batch_invariant=True`` uses one query per device batch when strict
    independence from surrounding query rows is required.
    """

    def __init__(self, support_vectors: np.ndarray, *, device: str = "auto",
                 query_tile_size: int = 512, support_tile_size: int = 16_384,
                 batch_invariant: bool = False):
        if not isinstance(batch_invariant, (bool, np.bool_)):
            raise ValueError("batch_invariant must be a bool")
        self._query_tile_size = _positive_integer(query_tile_size, "query_tile_size")
        self._support_tile_size = _positive_integer(support_tile_size, "support_tile_size")
        self._batch_invariant = bool(batch_invariant)
        from .torch_backend import TorchExactL2Index
        self._implementation = TorchExactL2Index(
            support_vectors, device=device,
            query_batch_size=1 if self._batch_invariant else self._query_tile_size,
            support_tile_size=self._support_tile_size,
        )

    @property
    def count(self): return self._implementation.count

    @property
    def dimension(self): return self._implementation.dimension

    @property
    def device(self): return self._implementation.device

    @property
    def query_tile_size(self): return self._query_tile_size

    @property
    def query_batch_size(self): return self._implementation.query_batch_size

    @property
    def support_tile_size(self): return self._support_tile_size

    @property
    def batch_invariant(self): return self._batch_invariant

    def search_one(self, query, k, exclude_index=None):
        return self._implementation.search_one(query, k, exclude_index)

    def search_many(self, queries, k, exclude_indices=None):
        return self._implementation.search_many(queries, k, exclude_indices)
