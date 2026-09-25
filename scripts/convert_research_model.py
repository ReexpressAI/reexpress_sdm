# Copyright Reexpress AI, Inc. All rights reserved.
"""Convert a research-code SDM model directory (reexpress_mcp_server v2.5.0
layout) into a schema-v1 ``.sdmkitmodel`` package.

The research checkpoint stores the exemplar adaptor as a ``Conv1d`` whose
kernel spans the whole input vector plus a ``Linear`` classifier; the support
set as a serialized FAISS flat index; and calibration/uncertainty statistics in
``meta.json``. Everything the SDMKit artifact needs is present, so the
conversion is mechanical. Values the research layout does not carry (class
names, representation provenance) are supplied by flags or placeholders.

Usage (from the repository root in an environment with PyTorch and FAISS)::

    PYTHONPATH=src python scripts/convert_research_model.py \
        /path/to/model_dir /path/to/output.sdmkitmodel \
        --documents_db /path/to/support_documents.db \
        --class_name "NOT Verified" --class_name "Verified"
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np


def _read_json_lines_dict(path: Path) -> dict:
    """The research code writes one JSON object per line and keeps the last."""
    result: dict = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                result = json.loads(line)
    return result


def _load_tensor(path: Path) -> np.ndarray:
    import torch

    value = torch.load(path, weights_only=True, map_location="cpu")
    return value.detach().cpu().numpy()


def _support_vectors(path: Path) -> np.ndarray:
    import faiss

    serialized = np.load(path, allow_pickle=False)
    index = faiss.deserialize_index(serialized)
    flat = faiss.downcast_index(index)
    return np.ascontiguousarray(flat.reconstruct_n(0, flat.ntotal), dtype=np.float32)


def _documents(db_path: Path | None, ids: list[str]) -> dict[str, str]:
    if db_path is None:
        return {}
    documents: dict[str, str] = {}
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        wanted = set(ids)
        cursor = connection.execute(
            "SELECT document_id, user_question, ai_response, model1_summary FROM documents"
        )
        for document_id, question, response, summary in cursor:
            if document_id not in wanted:
                continue
            parts = []
            if question:
                parts.append(f"Question: {question}")
            if response:
                parts.append(f"Response: {response}")
            if not parts and summary:
                parts.append(summary)
            if parts:
                documents[document_id] = "\n\n".join(parts)
    finally:
        connection.close()
    return documents


def convert(
    model_dir: Path,
    output: Path,
    *,
    documents_db: Path | None,
    class_names: list[str] | None,
    fingerprint: str | None,
    include_ood_calibration_rows: bool,
    overwrite: bool,
) -> Path:
    from reexpress_sdm.artifact import write_artifact
    from reexpress_sdm.training import build_artifact
    from reexpress_sdm.types import AdapterWeights, Region, SupportRecord
    import torch

    meta = _read_json_lines_dict(model_dir / "meta.json")
    number_of_classes = int(meta["numberOfClasses"])
    embedding_dimension = int(meta["embedding_size"])
    exemplar_dimension = int(meta["exemplar_vector_dimension"])

    state = torch.load(model_dir / "compression_index.pt", weights_only=True, map_location="cpu")
    projection_weight = state["conv.weight"].detach().cpu().numpy().reshape(exemplar_dimension, embedding_dimension)
    weights = AdapterWeights(
        projection_weight=projection_weight,
        projection_bias=state["conv.bias"].detach().cpu().numpy(),
        classifier_weight=state["fc.weight"].detach().cpu().numpy(),
        classifier_bias=state["fc.bias"].detach().cpu().numpy(),
    )

    support_vectors = _support_vectors(model_dir / "support.npy")
    support_ids = _read_json_lines_dict(model_dir / "support_ids.json")["support_ids"]
    support_labels = _load_tensor(model_dir / "support_labels.pt").astype(int)
    support_predicted = _load_tensor(model_dir / "support_predicted.pt").astype(int)
    if not (len(support_ids) == support_vectors.shape[0] == support_labels.shape[0] == support_predicted.shape[0]):
        raise SystemExit(
            f"support size mismatch: ids={len(support_ids)} vectors={support_vectors.shape[0]} "
            f"labels={support_labels.shape[0]} predicted={support_predicted.shape[0]}"
        )
    documents = _documents(documents_db, support_ids)
    support_records = [
        SupportRecord(id=identifier, label=int(label), predicted_label=int(predicted), document=documents.get(identifier))
        for identifier, label, predicted in zip(support_ids, support_labels, support_predicted)
    ]

    def per_class(mapping: dict) -> list[list[float]]:
        return [[float(x) for x in mapping.get(str(index), mapping.get(index, []))] for index in range(number_of_classes)]

    distance_cdfs = per_class(meta["trueClass_To_dCDF"])
    rescaled_similarity_cdfs = per_class(meta["trueClass_To_qCumulativeSampleSizeArray"])
    regions = [
        Region(
            alpha=float(region["alpha"]),
            minimum_rescaled_similarity=float(region["min_rescaled_similarity"]),
            output_thresholds=tuple(float(x) for x in region["output_thresholds"]),
        )
        for region in meta["hr_regions"]
    ]

    calibration_rows = None
    calibration_uuids_path = model_dir / "calibration_uuids.json"
    if calibration_uuids_path.exists():
        calibration_ids = _read_json_lines_dict(calibration_uuids_path)["calibration_uuids"]
        calibration_labels = _load_tensor(model_dir / "calibration_labels.pt").astype(int)
        calibration_predicted = _load_tensor(model_dir / "calibration_predicted_labels.pt").astype(int)
        calibration_sdm = _load_tensor(model_dir / "calibration_sdm_outputs.pt").astype(np.float64)
        calibration_q_prime = _load_tensor(model_dir / "calibration_rescaled_similarity_values.pt").astype(np.float64).reshape(-1)
        calibration_labels = calibration_labels.reshape(-1)
        calibration_predicted = calibration_predicted.reshape(-1)
        calibration_sdm = calibration_sdm.reshape(len(calibration_ids), -1)
        ood_flags = list(meta.get("calibration_is_ood_indicators", [])) or [0] * len(calibration_ids)
        rows = []
        seen: set[str] = set()
        for index, identifier in enumerate(calibration_ids):
            if not include_ood_calibration_rows and int(ood_flags[index]) != 0:
                continue
            if identifier in seen:
                continue
            seen.add(identifier)
            sdm = calibration_sdm[index]
            total = float(sdm.sum())
            if total > 0:
                sdm = sdm / total
            rows.append(
                {
                    "id": identifier,
                    "label": int(calibration_labels[index]),
                    "prediction": int(calibration_predicted[index]),
                    "sdm": [float(x) for x in sdm],
                    "qPrime": float(max(0.0, calibration_q_prime[index])),
                }
            )
        calibration_rows = rows

    stats = meta.get("training_embedding_summary_stats", {})
    artifact = build_artifact(
        weights=weights,
        support_vectors=support_vectors,
        support_records=support_records,
        distance_cdfs=distance_cdfs,
        rescaled_similarity_cdfs=rescaled_similarity_cdfs,
        regions=regions,
        embedding_dimension=embedding_dimension,
        exemplar_dimension=exemplar_dimension,
        number_of_classes=number_of_classes,
        class_names=class_names,
        max_neighbors=int(meta.get("maxQAvailableFromIndexer", 2048)),
        q_offset=float(meta.get("q_rescale_offset", 2)),
        ood_limit=int(meta.get("ood_limit", 0)),
        alpha_resolution=float(meta.get("alpha_resolution", 0.05)),
        normalization_mean=float(stats.get("training_embedding_mean", 0.0)),
        normalization_standard_deviation=float(stats.get("training_embedding_std", 1.0)),
        representation_provider="precomputed",
        representation_fingerprint="embedding_v1" if fingerprint is None else fingerprint,
        model_id=str(meta.get("uncertaintyModelUUID") or "") or None,
        producer_name="reexpress_sdm convert_research_model",
        producer_version="0.1.0",
        metadata={
            "source": str(model_dir.name),
            "researchVersion": str(meta.get("version", "")),
            "uncertaintyModelUUID": str(meta.get("uncertaintyModelUUID", "")),
            "calibrationTrainingStage": int(meta.get("calibration_training_stage", 0)),
            "note": "Converted from the research checkpoint for testing; class names and provenance are placeholders.",
        },
        calibration_rows=calibration_rows,
    )
    return write_artifact(output, artifact, overwrite=overwrite)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a research checkpoint to a portable .sdmkitmodel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    parser.add_argument("model_dir", type=Path, help="research checkpoint directory")
    parser.add_argument("output", type=Path, help="destination .sdmkitmodel directory")
    parser.add_argument("--documents_db", type=Path, default=None, help="research support-documents SQLite database")
    parser.add_argument("--class_name", action="append", dest="class_names", help="class name in label order (repeat)")
    parser.add_argument("--fingerprint", default="embedding_v1", help="representation fingerprint datasets must declare")
    parser.add_argument("--include_ood_calibration_rows", action="store_true",
                        help="keep calibration rows the research code flagged as OOD (excluded by default)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output package")
    args = parser.parse_args(argv)
    path = convert(
        args.model_dir, args.output,
        documents_db=args.documents_db, class_names=args.class_names, fingerprint=args.fingerprint,
        include_ood_calibration_rows=args.include_ood_calibration_rows, overwrite=args.overwrite,
    )
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
