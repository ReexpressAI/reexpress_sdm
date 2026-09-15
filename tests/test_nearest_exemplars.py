# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reexpress_sdm import (
    AdapterWeights, Dataset, ExactL2Index, Region, SDMController, SDMModel,
    SupportMatch, SupportRecord, build_artifact, iter_dataset_rows,
    score_dataset_rows, score_document, write_artifact, write_scored_jsonl,
)
from reexpress_sdm.cli import build_parser, main


DOCUMENT = '<script>"quoted" & \\ path\nnext line\u2028Unicode café</script>'


def _artifact(*, count=32, max_neighbors=32, tied=False):
    # Exact binary fractions keep the expected distance ordering unambiguous.
    vectors = np.asarray([[1 + index / 8, 0] for index in range(count)], dtype=np.float32)
    if tied:
        vectors[:] = [1, 0]
    records = tuple(SupportRecord(f"s{index}", 0, 0,
                                  document=DOCUMENT if index == 0 else None,
                                  metadata={"privateSourceMetadata": "not a display field"})
                    for index in range(count))
    return build_artifact(
        weights=AdapterWeights(np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32),
                               np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32)),
        support_vectors=vectors, support_records=records,
        distance_cdfs=((0, 1, 16), (0, 1, 16)),
        rescaled_similarity_cdfs=((0, 1), (0, 1)),
        regions=(Region(0.9, 1, (0.9, 0.9)),),
        embedding_dimension=2, exemplar_dimension=2, number_of_classes=2,
        class_names=("zero", "one"), max_neighbors=max_neighbors,
        alpha_resolution=0.1, representation_fingerprint="nearest-fixture-v1",
        model_id="nearest-fixture-model",
    )


def _legacy_fields(score):
    result = score.to_dict()
    result.pop("nearestSupportMatches", None)
    return result


class _IndexSpy:
    def __init__(self, support):
        self.index = ExactL2Index(support, device="cpu")
        self.calls = []

    @property
    def count(self):
        return self.index.count

    @property
    def dimension(self):
        return self.index.dimension

    def search_many(self, queries, k, exclude_indices=None):
        self.calls.append(("many", k, exclude_indices))
        return self.index.search_many(queries, k, exclude_indices)

    def search_one(self, query, k, exclude_index=None):
        self.calls.append(("one", k, exclude_index))
        return self.index.search_one(query, k, exclude_index)


class NearestExemplarTests(unittest.TestCase):
    def test_custom_indexes_receive_display_counts_capped_to_eligible_support(self):
        class StrictScalarIndex:
            def __init__(self, support):
                self.index = ExactL2Index(support, device="cpu")
                self.count, self.dimension = self.index.count, self.index.dimension
                self.calls = []

            def search_one(self, query, k, exclude_index=None):
                if k > self.count - (exclude_index is not None):
                    raise ValueError("k exceeds eligible support")
                self.calls.append(("one", k, exclude_index is not None))
                return self.index.search_one(query, k, exclude_index)

        class StrictBatchIndex(StrictScalarIndex):
            def search_many(self, queries, k, exclude_indices=None):
                if k > self.count - (exclude_indices is not None):
                    raise ValueError("k exceeds eligible support")
                self.calls.append(("many", k, exclude_indices is not None))
                return self.index.search_many(queries, k, exclude_indices)

        for index_type in (StrictScalarIndex, StrictBatchIndex):
            for exclusions in ([None, None], [0, 1], [0, None]):
                with self.subTest(index=index_type.__name__, exclusions=exclusions):
                    artifact = _artifact(count=4, max_neighbors=3)
                    index = index_type(artifact.support_vectors)
                    model = SDMModel(artifact, index=index, device="cpu")
                    baseline = model.score([[1, 0], [1.125, 0]], identity_support_indices=exclusions,
                                           nearest_exemplars=0)
                    index.calls.clear()
                    scores = model.score([[1, 0], [1.125, 0]], identity_support_indices=exclusions,
                                         nearest_exemplars=25)
                    for score, prior, excluded in zip(scores, baseline, exclusions):
                        self.assertEqual(_legacy_fields(score), _legacy_fields(prior))
                        self.assertEqual(len(score.nearest_support_matches), 4 - (excluded is not None))
                        self.assertNotIn(excluded, [match.support_index for match in score.nearest_support_matches])
                    if index_type is StrictBatchIndex and (all(i is None for i in exclusions) or all(i is not None for i in exclusions)):
                        self.assertEqual(index.calls[-1], ("many", 3 if exclusions[0] is not None else 4,
                                                         exclusions[0] is not None))
                    else:
                        self.assertEqual(index.calls[-2:], [("one", 3 if i is not None else 4, i is not None)
                                                          for i in exclusions])

    def test_extra_display_search_preserves_single_batch_borrowed_calibration_arrays(self):
        class BufferedBatchIndex(_IndexSpy):
            def __init__(self, support):
                super().__init__(support)
                self.distances = np.empty(2 * self.count, dtype=np.float32)
                self.indices = np.empty(2 * self.count, dtype=np.int64)

            def search_many(self, queries, k, exclude_indices=None):
                distances, indices = super().search_many(queries, k, exclude_indices)
                # A valid custom implementation may reuse packed output buffers.
                # Expanding k changes each row's offset in this shared storage.
                self.distances[:distances.size] = distances.ravel()
                self.indices[:indices.size] = indices.ravel()
                return (self.distances[:distances.size].reshape(distances.shape),
                        self.indices[:indices.size].reshape(indices.shape))

        artifact = _artifact(count=4, max_neighbors=2)
        model = SDMModel(artifact, index=BufferedBatchIndex(artifact.support_vectors), device="cpu")
        queries = [[1, 0], [1.125, 0]]
        baseline = model.score(queries, nearest_exemplars=0)
        scores = model.score(queries, nearest_exemplars=25)
        self.assertEqual([score.d0 for score in baseline], [0, 0])
        for score, prior in zip(scores, baseline):
            self.assertEqual(_legacy_fields(score), _legacy_fields(prior))
            self.assertEqual(len(score.nearest_support_matches), 4)
            self.assertEqual(score.nearest_support_matches[0].support_index, score.nearest_support_index)
            self.assertEqual(score.nearest_support_matches[0].squared_distance, score.d0)

    def test_default_and_explicit_counts_preserve_all_existing_diagnostics(self):
        model = SDMModel(_artifact(), device="cpu")
        baseline = model.score([[1, 0]], ids=["query"], nearest_exemplars=0)[0]
        self.assertEqual(baseline.nearest_support_matches, ())
        self.assertNotIn("nearestSupportMatches", baseline.to_dict())
        self.assertEqual(baseline.nearest_support_index, 0)
        self.assertEqual(baseline.nearest_support_id, "s0")
        default = model.score([[1, 0]], ids=["query"])[0]
        self.assertEqual(len(default.nearest_support_matches), 25)
        self.assertEqual([match.support_index for match in default.nearest_support_matches], list(range(25)))
        self.assertEqual([match.squared_distance for match in default.nearest_support_matches],
                         [(index / 8) ** 2 for index in range(25)])
        self.assertEqual(_legacy_fields(default), _legacy_fields(baseline))
        for requested, expected in ((0, 0), (1, 1), (7, 7), (25, 25), (10**30, 32)):
            with self.subTest(requested=requested):
                score = model.score([[1, 0]], ids=["query"], nearest_exemplars=requested)[0]
                self.assertEqual(len(score.nearest_support_matches), expected)
                self.assertEqual(_legacy_fields(score), _legacy_fields(baseline))
                if expected:
                    first = score.nearest_support_matches[0]
                    self.assertEqual((first.support_index, first.id, first.squared_distance),
                                     (score.nearest_support_index, score.nearest_support_id, score.d0))

    def test_controller_forwards_custom_count_and_small_support_is_capped(self):
        controller = SDMController(SDMModel(_artifact(count=3), device="cpu"))
        self.assertEqual(len(controller.score([[1, 0]])[0].nearest_support_matches), 3)
        self.assertEqual(len(controller.score([[1, 0]], nearest_exemplars=1)[0].nearest_support_matches), 1)
        self.assertEqual(controller.score([[1, 0]], nearest_exemplars=0)[0].nearest_support_matches, ())

    def test_ties_use_support_index_order_and_respect_all_exclusion_patterns(self):
        model = SDMModel(_artifact(tied=True), device="cpu")
        for exclusions in ([None, None], [0, 7], [None, 7]):
            with self.subTest(exclusions=exclusions):
                scores = model.score([[1, 0], [1, 0]], identity_support_indices=exclusions,
                                     nearest_exemplars=10**6)
                baseline = model.score([[1, 0], [1, 0]], identity_support_indices=exclusions,
                                       nearest_exemplars=0)
                for score, prior, excluded in zip(scores, baseline, exclusions):
                    expected = [index for index in range(32) if index != excluded]
                    self.assertEqual([match.support_index for match in score.nearest_support_matches], expected)
                    self.assertTrue(all(match.squared_distance == 0 for match in score.nearest_support_matches))
                    self.assertEqual(_legacy_fields(score), _legacy_fields(prior))

    def test_display_search_larger_than_calibration_prefix_does_not_change_q(self):
        model = SDMModel(_artifact(max_neighbors=3), device="cpu")
        for exclusions in ([None, None], [0, 1], [None, 1]):
            with self.subTest(exclusions=exclusions):
                baseline = model.score([[1, 0], [1.125, 0]], identity_support_indices=exclusions,
                                       nearest_exemplars=0)
                expanded = model.score([[1, 0], [1.125, 0]], identity_support_indices=exclusions,
                                       nearest_exemplars=25)
                for score, prior, excluded in zip(expanded, baseline, exclusions):
                    self.assertEqual(len(score.nearest_support_matches), 25)
                    self.assertEqual(score.q, 3 if excluded is None else 2)
                    self.assertEqual(_legacy_fields(score), _legacy_fields(prior))
                    self.assertNotIn(excluded, [match.support_index for match in score.nearest_support_matches])

    def test_existing_calibration_search_is_reused_when_it_already_has_enough_matches(self):
        for exclusions in ([None, None], [0, 1], [None, 1]):
            with self.subTest(exclusions=exclusions):
                artifact = _artifact()
                index = _IndexSpy(artifact.support_vectors)
                model = SDMModel(artifact, index=index, device="cpu")
                model.score([[1, 0], [1.125, 0]], identity_support_indices=exclusions, nearest_exemplars=0)
                prior_calls = list(index.calls)
                index.calls.clear()
                scores = model.score([[1, 0], [1.125, 0]], identity_support_indices=exclusions)
                self.assertEqual(index.calls, prior_calls)
                self.assertTrue(all(len(score.nearest_support_matches) == 25 for score in scores))

    def test_separate_display_search_keeps_original_calibration_request(self):
        artifact = _artifact(max_neighbors=3)
        index = _IndexSpy(artifact.support_vectors)
        model = SDMModel(artifact, index=index, device="cpu")
        baseline = model.score([[1, 0]], nearest_exemplars=0)[0]
        index.calls.clear()
        expanded = model.score([[1, 0]])[0]
        self.assertEqual([(kind, count) for kind, count, _ in index.calls], [("many", 3), ("many", 25)])
        self.assertEqual(_legacy_fields(expanded), _legacy_fields(baseline))

    def test_invalid_counts_fail_and_empty_valid_input_remains_empty(self):
        model = SDMModel(_artifact(), device="cpu")
        for invalid in (-1, True, False, np.bool_(True), 1.0, "25", None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                model.score([[1, 0]], nearest_exemplars=invalid)
        self.assertEqual(model.score(np.empty((0, 2), dtype=np.float32)), ())

    def test_match_payload_preserves_document_text_and_omits_absent_content(self):
        artifact = _artifact()
        artifact = replace(artifact, support_records=tuple(
            replace(record, metadata={"prompt": "Independent prompt"})
            for record in artifact.support_records
        ))
        score = SDMModel(artifact, device="cpu").score([[1, 0]], nearest_exemplars=2)[0]
        self.assertIsInstance(score.nearest_support_matches[0], SupportMatch)
        for detail in ("full", "compact"):
            encoded = json.loads(json.dumps(score.to_dict(detail=detail), ensure_ascii=True, allow_nan=False))
            first, second = encoded["nearestSupportMatches"]
            self.assertEqual(first, {"supportIndex": 0, "id": "s0", "label": 0, "predictedLabel": 0,
                                     "squaredDistance": 0.0, "document": DOCUMENT})
            self.assertNotIn("document", second)
            self.assertNotIn("metadata", first)
            self.assertNotIn("prompt", first)
            self.assertNotIn("prompt", second)

    def test_support_only_ood_label_is_retained_in_nearest_exemplar(self):
        artifact = _artifact()
        artifact = replace(artifact, support_records=(replace(artifact.support_records[0], label=-99),
                                                       *artifact.support_records[1:]))
        model = SDMModel(artifact, device="cpu")
        score = model.score([[1, 0]], nearest_exemplars=1)[0]
        self.assertEqual(score.q, 0)
        self.assertEqual(score.nearest_support_matches[0].label, -99)
        self.assertEqual(score.nearest_support_matches[0].predicted_label, 0)
        row = score_document(model, {"id": "query", "label": -99, "embedding": [1, 0]}, nearest_exemplars=1)
        self.assertEqual(row["nearestSupportMatches"][0]["label"], -99)

    def test_scored_document_and_dataset_rows_propagate_counts_and_replace_stale_matches(self):
        model = SDMModel(_artifact(), device="cpu")
        source = {"id": "query", "label": 0, "embedding": [1, 0], "document": "source content"}
        old = score_document(model, source, nearest_exemplars=2)
        self.assertEqual(len(old["nearestSupportMatches"]), 2)
        changed = score_document(model, old, nearest_exemplars=1)
        self.assertEqual(len(changed["nearestSupportMatches"]), 1)
        cleared = score_document(model, old, nearest_exemplars=0)
        self.assertNotIn("nearestSupportMatches", cleared)
        self.assertEqual(cleared["nearestSupportID"], "s0")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            write_scored_jsonl([old], path)
            dataset = Dataset.load(path)
            self.assertEqual(len(list(score_dataset_rows(model, dataset, nearest_exemplars=3))[0]["nearestSupportMatches"]), 3)
            self.assertNotIn("nearestSupportMatches", list(score_dataset_rows(model, dataset, nearest_exemplars=0))[0])

    def test_cli_default_and_both_option_spellings(self):
        parser = build_parser()
        arguments = ["score", "--model", "model.sdmkitmodel", "--input", "input.jsonl"]
        self.assertEqual(parser.parse_args(arguments).nearest_exemplars, 25)
        self.assertEqual(parser.parse_args(arguments + ["--nearest_exemplars", "7"]).nearest_exemplars, 7)

    def test_cli_jsonl_full_compact_and_bundle_include_requested_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.sdmkitmodel"
            source = root / "input.jsonl"
            write_artifact(model, _artifact())
            write_scored_jsonl([{"id": "query", "label": 0, "embedding": [1, 0]}], source)
            for detail, suffix in (("full", ".jsonl"), ("compact", ".jsonl"), ("full", ".sdmdataset")):
                with self.subTest(detail=detail, suffix=suffix):
                    output = root / (detail + suffix)
                    stderr = io.StringIO()
                    with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                        status = main(["score", "--model", str(model), "--input", str(source),
                                       "--output", str(output), "--matching_device", "cpu", "--detail", detail,
                                       "--nearest_exemplars", "2"])
                    self.assertEqual(status, 0, stderr.getvalue())
                    row = next(iter_dataset_rows(output))
                    self.assertEqual(len(row["nearestSupportMatches"]), 2)
                    self.assertEqual(row["nearestSupportMatches"][0]["document"], DOCUMENT)
                    self.assertEqual(row["nearestSupportID"], row["nearestSupportMatches"][0]["id"])
                    if detail == "full":
                        self.assertEqual(row["scoreSchemaVersion"], 1)
                        self.assertEqual(list(row["embedding"]), [1, 0])
                    else:
                        self.assertNotIn("scoreSchemaVersion", row)

    def test_invalid_cli_count_does_not_replace_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.sdmkitmodel"
            source = root / "input.jsonl"
            write_artifact(model, _artifact())
            write_scored_jsonl([{"id": "query", "label": 0, "embedding": [1, 0]}], source)
            for suffix in (".jsonl", ".sdmdataset"):
                output = root / ("old" + suffix)
                if suffix == ".sdmdataset":
                    output.mkdir()
                    sentinel = output / "sentinel"
                else:
                    sentinel = output
                sentinel.write_bytes(b"previous output")
                with self.subTest(suffix=suffix), redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                    try:
                        status = main(["score", "--model", str(model), "--input", str(source),
                                       "--output", str(output), "--overwrite", "--matching_device", "cpu",
                                       "--nearest_exemplars", "-1"])
                    except SystemExit as error:
                        status = error.code
                self.assertNotEqual(status, 0)
                self.assertEqual(sentinel.read_bytes(), b"previous output")


if __name__ == "__main__":
    unittest.main()
