#!/usr/bin/env python3
# Copyright Reexpress AI, Inc. All rights reserved.
"""Regenerate small Python-produced model/score fixtures consumed by Swift tests.

Run with PYTHONPATH=src python scripts/generate_score_interchange_fixture.py from
the reexpress_sdm repository root. No network, accelerator, or research data is needed.
"""
from pathlib import Path
import json
import numpy as np

from reexpress_sdm import AdapterWeights, Region, SupportRecord, build_artifact, write_artifact
from reexpress_sdm.cli import main
from reexpress_sdm.math import dkw_errors, distance_band, sdm_probabilities, stable_softmax


def generate(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact(
        weights=AdapterWeights(np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32),
                               np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32)),
        support_vectors=np.asarray([[1, 0], [2, 0], [0, 1], [0, 2], [4, 4]], dtype=np.float32),
        support_records=tuple(SupportRecord(f"s{i}", label, prediction) for i, (label, prediction) in enumerate([(0, 0), (0, 0), (1, 1), (1, 1), (1, 0)])),
        distance_cdfs=((0, 1, 4), (0, 1, 4)),
        rescaled_similarity_cdfs=(tuple([0]*200+[1]*200+[2]*200),)*2,
        regions=(Region(0.9, 1, (0.9, 0.9)), Region(0.8, 1, (0.8, 0.8))),
        embedding_dimension=2, exemplar_dimension=2, number_of_classes=2,
        class_names=("A", "B"), max_neighbors=5, alpha_resolution=0.1,
        representation_fingerprint="scored-document-fixture-v1", model_id="python-score-interchange-v1",
    )
    artifact.manifest["createdAt"] = "2026-09-05T00:00:00Z"
    model = destination / "fixture.sdmkitmodel"
    write_artifact(model, artifact, overwrite=True)
    cases = {
        "inference": [
            {"id": "centroid-and-lower", "label": 0, "document": "Reliable A — line one\nline two", "embedding": [2], "attributes": [0], "metadata": {"source": "Python"}},
            {"id": "centroid-only", "label": 1, "document": "Centroid B", "embedding": [0], "attributes": [1]},
            {"id": "rejected", "label": -1, "document": "Ambiguous", "embedding": [0], "attributes": [0]},
            {"id": "ood", "label": 0, "embedding": [4], "attributes": [4]},
        ],
        "explicit": [
            {"id": "explicit-s0", "label": 0, "document": "An explicitly identified training example", "embedding": [1], "attributes": [0], "support_index": 0},
            {"id": "explicit-s3", "label": 1, "embedding": [0], "attributes": [2], "support_index": 3},
        ],
    }
    for name, rows in cases.items():
        source = destination / f"source-{name}.jsonl"
        source.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
        args = ["score", "--model", str(model), "--input", str(source), "--output", str(destination / f"scored-{name}.jsonl")]
        if name == "explicit":
            args += ["--identity_support_index_field", "support_index"]
        assert main(args) == 0
    actual = [json.loads(line) for line in (destination / "scored-inference.jsonl").read_text().splitlines()]
    assert any(row["centroidRegionAlpha"] == 0.9 and row["lowerRegionAlpha"] == 0.9 for row in actual)
    assert any(row["centroidRegionAlpha"] == 0 for row in actual)
    assert any(row["q"] == 0 for row in actual)
    assert all(not {"isOOD", "floorQPrime", "floorQPrimeLower"}.intersection(row) for row in actual)

    # The same Float32 activation has two legitimate multiplication
    # associations. Large common logits expose their difference, including a
    # higher reliability rung lying within one ULP of the producer's result.
    logits = np.asarray([100000, 100016], dtype=np.float32)
    distance = np.float32(1) - np.float32(0.9)
    lower, _ = distance_band(distance, dkw_errors([1001, 1001], 0.8))
    threshold = float(np.nextafter(sdm_probabilities(logits, 3, lower)[1], np.float32(np.inf)))
    edge = build_artifact(
        weights=AdapterWeights(np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32),
                               np.zeros((2, 2), dtype=np.float32), logits),
        support_vectors=np.asarray([[3, 0], [4, 0], [5, 0]], dtype=np.float32),
        support_records=tuple(SupportRecord(f"large-s{i}", 1, 1) for i in range(3)),
        distance_cdfs=(tuple(range(10)),)*2,
        rescaled_similarity_cdfs=(tuple([0]*1001),)*2,
        regions=(Region(0.8, 1, (1, threshold)), Region(0.7, 1, (1, 0.8))),
        embedding_dimension=2, exemplar_dimension=2, number_of_classes=2,
        class_names=("A", "B"), max_neighbors=3, alpha_resolution=0.1,
        representation_fingerprint="scored-document-rounding-v1", model_id="python-score-rounding-v1",
    )
    edge.manifest["createdAt"] = "2026-09-05T00:00:00Z"
    edge_model = destination / "rounding.sdmkitmodel"
    write_artifact(edge_model, edge, overwrite=True)
    source = destination / "source-rounding.jsonl"
    source.write_text(json.dumps({"id": "rounding-boundary", "label": 1, "embedding": [0, 0]})+"\n")
    assert main(["score", "--model", str(edge_model), "--input", str(source), "--output", str(destination / "scored-rounding.jsonl")]) == 0
    scored = json.loads((destination / "scored-rounding.jsonl").read_text())
    assert scored["centroidRegionAlpha"] == 0.8 and scored["lowerRegionAlpha"] == 0.7
    alternate = stable_softmax(logits*(distance*np.log(np.float32(5))))
    assert np.max(np.abs(np.asarray(scored["sdm"])-alternate)) > 2e-5
    assert stable_softmax(logits*(np.float32(lower)*np.log(np.float32(5))))[1] > threshold


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "score-interchange")
