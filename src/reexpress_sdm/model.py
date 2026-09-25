# Copyright Reexpress AI, Inc. All rights reserved.
"""Immutable SDM inference runtime."""

from __future__ import annotations

import math  # noqa: F401 - retained for callers importing through this module
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .artifact import load_artifact, validate_artifact
from .errors import DimensionMismatchError, RepresentationMismatchError
from .index import DenseIndex, ExactL2Index
from .math import (
    distance_bands,
    distance_quantiles,
    dkw_error_matrix,
    effective_sample_size_matrix,
    region_accepts_many,
    rescaled_similarities,
    sdm_probabilities,
)
from .types import SDMArtifact, SDMScore, SupportMatch, floats


class SDMModel:
    """Portable artifact with reusable PyTorch device execution."""

    def __init__(self, artifact: SDMArtifact, index: DenseIndex | None = None, *,
                 backend: str = "torch", device: str = "auto", query_batch_size: int = 256,
                 support_tile_size: int = 16_384):
        artifact = validate_artifact(artifact)
        self.artifact = artifact
        self.configuration = artifact.configuration
        self.number_of_classes = int(self.configuration["numberOfClasses"])
        self.embedding_dimension = int(self.configuration["embeddingDimension"])
        self.exemplar_dimension = int(self.configuration["exemplarDimension"])
        self.max_neighbors = int(self.configuration["maxNeighbors"])
        self.q_offset = float(self.configuration["qOffset"])
        self.ood_limit = int(self.configuration["oodLimit"])
        self.normalization_mean = float(artifact.manifest["normalization"]["mean"])
        self.normalization_standard_deviation = float(
            artifact.manifest["normalization"]["standardDeviation"]
        )
        self.representation_fingerprint = str(
            artifact.manifest["representation"]["fingerprint"]
        )
        self.distance_cdfs = tuple(
            tuple(float(x) for x in cdf) for cdf in artifact.manifest["distanceCDFs"]
        )
        self.q_prime_cdfs = tuple(
            tuple(float(x) for x in cdf)
            for cdf in artifact.manifest["rescaledSimilarityCDFs"]
        )
        self.regions = artifact.regions
        from .runtime import create_runtime
        from .backends import create_dense_index
        self._runtime = create_runtime(artifact, backend=backend, device=device, query_batch_size=query_batch_size)
        self.backend, self.device = self._runtime.backend, self._runtime.device
        self.query_batch_size = self._runtime.query_batch_size
        self.support_tile_size = support_tile_size
        self._native_index = index is None
        self._index = index if index is not None else create_dense_index(
            self.backend, artifact.support_vectors, device=self.device,
            query_batch_size=self.query_batch_size, support_tile_size=support_tile_size,
        )
        if self._index.dimension != self.exemplar_dimension:
            raise DimensionMismatchError("dense index dimension does not match artifact")
        if self._index.count != len(artifact.support_records):
            raise DimensionMismatchError("dense index count does not match support records")
        self._support_labels = np.asarray(
            [record.label for record in artifact.support_records], dtype=np.int64
        )
        self._support_predictions = np.asarray(
            [record.predicted_label for record in artifact.support_records], dtype=np.int64
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        verify_checksums: bool = True,
        index: DenseIndex | None = None,
        backend: str = "torch", device: str = "auto", query_batch_size: int = 256,
        support_tile_size: int = 16_384,
    ) -> "SDMModel":
        return cls(load_artifact(path, verify_checksums=verify_checksums), index=index,
                   backend=backend, device=device, query_batch_size=query_batch_size,
                   support_tile_size=support_tile_size)

    def _validate_fingerprint(self, fingerprint: str | None) -> None:
        if fingerprint is not None and fingerprint != self.representation_fingerprint:
            raise RepresentationMismatchError(
                f"input representation '{fingerprint}' does not match "
                f"artifact representation '{self.representation_fingerprint}'"
            )

    def transform(self, embeddings: Sequence[Sequence[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._runtime.transform(self._runtime.inputs(embeddings))

    def _neighbor_matches(
        self,
        exemplars: np.ndarray,
        exclusions: Sequence[int | None],
        *, resident: bool = False,
        display_count: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Exact neighbors per row, batched when the index and exclusions allow it."""

        count = exemplars.shape[0]
        # Batch the common all-evaluation and all-training cases when a backend
        # exposes search_many. Preserve search_one overrides on ExactL2Index
        # subclasses unless the subclass also deliberately overrides the batch
        # method (useful for instrumented or user-defined indexes).
        batch_search = getattr(self._index, "search_many_device" if resident else "search_many", None)
        index_type = type(self._index)
        if (
            isinstance(self._index, ExactL2Index)
            and index_type is not ExactL2Index
            and index_type.search_one is not ExactL2Index.search_one
            and index_type.search_many is ExactL2Index.search_many
        ):
            batch_search = None
        all_unexcluded = all(value is None for value in exclusions)
        all_excluded = count > 0 and all(value is not None for value in exclusions)
        if callable(batch_search) and (all_unexcluded or all_excluded):
            requested_neighbors = (self.max_neighbors - (1 if all_excluded else 0)
                                   if display_count is None else min(display_count, self._index.count - all_excluded))
            if requested_neighbors < 1:
                raise ValueError("no support neighbor remains after excluding the identity")
            batch_distances, batch_indices = batch_search(
                exemplars,
                requested_neighbors,
                exclusions if all_excluded else None,
            )
            return [(batch_distances[index], batch_indices[index]) for index in range(count)]
        if resident:
            exemplars = self._runtime.host(exemplars)
        matches: list[tuple[np.ndarray, np.ndarray]] = []
        for index in range(count):
            identity = exclusions[index]
            requested_neighbors = min(self.max_neighbors if display_count is None else display_count,
                                      self._index.count)
            if identity is not None and display_count is None:
                # The reference searches K rows including the identity, then drops
                # that first row. ``ExactL2Index`` applies exclusion before its K
                # cap, so request K-1 remaining rows here.
                requested_neighbors -= 1
            elif display_count is not None:
                requested_neighbors = min(requested_neighbors, self._index.count - (identity is not None))
            if requested_neighbors < 1:
                raise ValueError("no support neighbor remains after excluding the identity")
            matches.append(
                self._index.search_one(exemplars[index], requested_neighbors, exclude_index=identity)
            )
        return matches

    def _scores_from_matches(
        self,
        logits: np.ndarray,
        matches: Sequence[tuple[np.ndarray, np.ndarray]],
        identifiers: Sequence[str | None],
    ) -> tuple[SDMScore, ...]:
        """Row-batched SDM scoring; every quantity follows the scalar contract."""

        count = logits.shape[0]
        if count == 0:
            return ()
        classes = self.number_of_classes
        rows = np.arange(count)
        predictions = np.argmax(logits, axis=1).astype(np.int64)

        widths = [int(len(neighbor_indices)) for _, neighbor_indices in matches]
        if min(widths) == 0:
            raise ValueError("no support neighbor remains after excluding the identity")
        width = max(widths)
        neighbor_matrix = np.full((count, width), -1, dtype=np.int64)
        nearest_distances = np.empty(count, dtype=np.float32)
        for row, (distances, neighbor_indices) in enumerate(matches):
            neighbor_matrix[row, : widths[row]] = neighbor_indices
            nearest_distances[row] = distances[0]
        valid = neighbor_matrix >= 0
        safe_indices = np.where(valid, neighbor_matrix, 0)
        neighbor_predictions = self._support_predictions[safe_indices]
        agreement = (
            valid
            & (self._support_labels[safe_indices] == neighbor_predictions)
            & (neighbor_predictions == predictions[:, None])
        )
        q = np.sum(np.logical_and.accumulate(agreement, axis=1), axis=1).astype(np.int64)
        q_float = q.astype(np.float32)
        d0 = np.maximum(nearest_distances, np.float32(0.0))
        d = distance_quantiles(d0, self.distance_cdfs)
        probabilities = np.asarray(sdm_probabilities(logits, q_float, d, self.q_offset), dtype=np.float32)
        q_prime = rescaled_similarities(q_float, probabilities[rows, predictions], self.q_offset)
        sample_sizes = effective_sample_size_matrix(q_prime, self.q_prime_cdfs)

        centroid_alpha = np.zeros(count, dtype=np.float64)
        for region in self.regions:
            accepted = region_accepts_many(q_prime, probabilities, predictions, region, self.ood_limit)
            newly = accepted & (centroid_alpha == 0.0)
            centroid_alpha[newly] = region.alpha

        # Default values for a row rejected by every DKW-lower rung represent
        # maximum uncertainty, matching the PyTorch implementation.
        selected_errors = np.ones((count, classes), dtype=np.float32)
        d_lower, d_upper = distance_bands(d, selected_errors)
        sdm_lower = np.asarray(sdm_probabilities(logits, q_float, d_lower, self.q_offset), dtype=np.float32)
        sdm_upper = np.asarray(sdm_probabilities(logits, q_float, d_upper, self.q_offset), dtype=np.float32)
        q_prime_lower = rescaled_similarities(q_float, sdm_lower[rows, predictions], self.q_offset)
        lower_alpha = np.zeros(count, dtype=np.float64)
        for region in self.regions:
            unassigned = lower_alpha == 0.0
            if not np.any(unassigned):
                break
            errors = dkw_error_matrix(sample_sizes, region.alpha)
            region_lower, region_upper = distance_bands(d, errors)
            region_probabilities = np.asarray(
                sdm_probabilities(logits, q_float, region_lower, self.q_offset), dtype=np.float32
            )
            region_q_prime = rescaled_similarities(
                q_float, region_probabilities[rows, predictions], self.q_offset
            )
            accepted = region_accepts_many(
                region_q_prime, region_probabilities, predictions, region, self.ood_limit
            )
            newly = accepted & unassigned
            if not np.any(newly):
                continue
            lower_alpha[newly] = region.alpha
            selected_errors[newly] = errors[newly]
            d_lower[newly] = region_lower[newly]
            d_upper[newly] = region_upper[newly]
            sdm_lower[newly] = region_probabilities[newly]
            sdm_upper[newly] = np.asarray(
                sdm_probabilities(logits[newly], q_float[newly], region_upper[newly], self.q_offset),
                dtype=np.float32,
            )
            q_prime_lower[newly] = region_q_prime[newly]

        floor_q_prime = np.floor(q_prime).astype(np.int64)
        floor_q_prime_lower = np.floor(q_prime_lower).astype(np.int64)
        most_conservative = self.regions[0].alpha if self.regions else 0.0
        nearest = neighbor_matrix[:, 0]
        logits_list = logits.tolist()
        probabilities_list = probabilities.tolist()
        lower_list = sdm_lower.tolist()
        upper_list = sdm_upper.tolist()
        errors_list = selected_errors.tolist()
        sizes_list = sample_sizes.tolist()
        model_id = self.artifact.model_id
        support_records = self.artifact.support_records
        return tuple(
            SDMScore(
                model_id=model_id,
                id=identifiers[row],
                z_prime=tuple(logits_list[row]),
                prediction=int(predictions[row]),
                q=int(q[row]),
                d0=float(d0[row]),
                d=float(d[row]),
                sdm=tuple(probabilities_list[row]),
                q_prime=float(q_prime[row]),
                floor_q_prime=int(floor_q_prime[row]),
                is_ood=bool(floor_q_prime[row] <= self.ood_limit),
                nearest_support_index=int(nearest[row]),
                nearest_support_id=support_records[int(nearest[row])].id,
                effective_sample_sizes=tuple(sizes_list[row]),
                effective_sample_size_errors=tuple(errors_list[row]),
                d_lower=float(d_lower[row]),
                d_upper=float(d_upper[row]),
                sdm_lower=tuple(lower_list[row]),
                sdm_upper=tuple(upper_list[row]),
                q_prime_lower=float(q_prime_lower[row]),
                floor_q_prime_lower=int(floor_q_prime_lower[row]),
                centroid_region_alpha=float(centroid_alpha[row]),
                lower_region_alpha=float(lower_alpha[row]),
                most_conservative_region_alpha=most_conservative,
            )
            for row in range(count)
        )

    def score(
        self,
        embeddings: Sequence[Sequence[float]] | np.ndarray,
        *,
        ids: Sequence[str | None] | None = None,
        representation_fingerprint: str | None = None,
        identity_support_indices: Sequence[int | None] | None = None,
        nearest_exemplars: int = 25,
    ) -> tuple[SDMScore, ...]:
        """Score rows and retain up to ``nearest_exemplars`` exact matches.

        The display count never changes the neighbors used for SDM calibration.
        Zero keeps only the existing singular nearest-support fields.
        """
        if isinstance(nearest_exemplars, (bool, np.bool_)) or not isinstance(nearest_exemplars, (int, np.integer)) or nearest_exemplars < 0:
            raise ValueError("nearest_exemplars must be a nonnegative integer")
        nearest_exemplars = min(int(nearest_exemplars), self._index.count)
        self._validate_fingerprint(representation_fingerprint)
        values = self._runtime.inputs(embeddings)
        count = values.shape[0]
        identifiers = list(ids) if ids is not None else [None] * count
        exclusions = (
            list(identity_support_indices)
            if identity_support_indices is not None
            else [None] * count
        )
        if len(identifiers) != count or len(exclusions) != count:
            raise DimensionMismatchError("ids and identity support indices must align with embeddings")
        logits_batches, matches, display_matches = [], [], []
        offset = 0
        for logits, exemplars in self._runtime.project_batches(values):
            batch_count = logits.shape[0]
            if not self._native_index:
                exemplars = self._runtime.host(exemplars)
                if not np.all(np.isfinite(exemplars)):
                    raise ValueError("adaptor produced a non-finite exemplar or logit")
            batch_exclusions = exclusions[offset:offset + batch_count]
            batch_matches = self._neighbor_matches(exemplars, batch_exclusions, resident=self._native_index)
            if nearest_exemplars:
                # Usually the calibration search already selected enough rows.
                # A larger display request must not enlarge q's neighbor prefix.
                if any(len(indices) < min(nearest_exemplars, self._index.count - (identity is not None))
                       for (_, indices), identity in zip(batch_matches, batch_exclusions)):
                    if not self._native_index:
                        # Custom indexes may lend reusable result buffers. Keep
                        # q/d0's original rows before the display search writes them.
                        batch_matches = [(distances.copy(), indices.copy()) for distances, indices in batch_matches]
                    expanded = self._neighbor_matches(exemplars, batch_exclusions, resident=self._native_index,
                                                      display_count=nearest_exemplars)
                else:
                    expanded = batch_matches
                display_matches.extend((distances[:nearest_exemplars].copy(), indices[:nearest_exemplars].copy())
                                       if not self._native_index else (distances[:nearest_exemplars], indices[:nearest_exemplars])
                                       for distances, indices in expanded)
            matches.extend(batch_matches)
            host_logits = self._runtime.host(logits)
            if not np.all(np.isfinite(host_logits)):
                raise ValueError("adaptor produced a non-finite exemplar or logit")
            logits_batches.append(host_logits)
            offset += batch_count
        logits = np.concatenate(logits_batches) if logits_batches else np.empty((0, self.number_of_classes), np.float32)
        scores = self._scores_from_matches(logits, matches, identifiers)
        if not nearest_exemplars:
            return scores
        records = self.artifact.support_records
        return tuple(replace(score, nearest_support_matches=tuple(
            SupportMatch(support_index=int(index), id=records[int(index)].id,
                         label=records[int(index)].label, predicted_label=records[int(index)].predicted_label,
                         squared_distance=float(distance), document=records[int(index)].document)
            for distance, index in zip(distances, indices)
        )) for score, (distances, indices) in zip(scores, display_matches))
