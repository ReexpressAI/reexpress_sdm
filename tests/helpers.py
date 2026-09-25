# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

import numpy as np

from reexpress_sdm import AdapterWeights, Region, SupportRecord, build_artifact


def make_artifact():
    weights = AdapterWeights(
        projection_weight=np.eye(2, dtype=np.float32),
        projection_bias=np.zeros(2, dtype=np.float32),
        classifier_weight=np.eye(2, dtype=np.float32),
        classifier_bias=np.zeros(2, dtype=np.float32),
    )
    support_vectors = np.asarray(
        [[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]], dtype=np.float32
    )
    records = (
        SupportRecord("s0", 0, 0),
        SupportRecord("s1", 0, 0),
        SupportRecord("s2", 1, 1),
        SupportRecord("s3", 1, 1),
    )
    return build_artifact(
        weights=weights,
        support_vectors=support_vectors,
        support_records=records,
        distance_cdfs=((0.0, 1.0, 4.0), (0.0, 1.0, 4.0)),
        rescaled_similarity_cdfs=((0.0, 1.0, 2.0), (0.0, 1.0, 2.0)),
        regions=(Region(0.9, 1.0, (0.9, 0.9)), Region(0.8, 1.0, (0.8, 0.8))),
        embedding_dimension=2,
        exemplar_dimension=2,
        number_of_classes=2,
        class_names=("zero", "one"),
        max_neighbors=4,
        alpha_resolution=0.1,
        representation_fingerprint="fixture-v1",
        model_id="fixture-model",
    )
