# SDMKit Model Artifact v1

`*.sdmkitmodel` is a directory package shared by SDMKit, Reexpress two, and the
framework-neutral SDK. The format is deliberately independent of PyTorch,
FAISS, MLX, and any agent framework.

## Files

- `manifest.json`: required typed metadata and calibration statistics.
- `weights.f32`: required adaptor parameters in little-endian IEEE-754
  `float32`, row-major, concatenated as `projectionWeight[M,D]`,
  `projectionBias[M]`, `classifierWeight[C,M]`, `classifierBias[C]`.
- `support.f32`: required training/support exemplar vectors in little-endian
  IEEE-754 `float32`, row-major with shape `[N,M]`.
- `support.jsonl`: required support records in the same row order as
  `support.f32`. Every row contains `id`, `label`, and `predictedLabel`;
  `document` and `metadata` are optional. Because these fields can contain raw
  training content, a full artifact may be sensitive even though credentials
  are never stored in it.
- `calibration.jsonl`: optional cached calibration diagnostics for inspection
  and fast re-calibration. Each row uses the canonical fields `id`, `label`,
  `prediction`, `sdm`, and `qPrime`; `q`, `d0`, `d`, and `zPrime` are optional.
  It is not required for inference.

Writers stage the package on the destination volume and publish it only after
every file and checksum has been written. Sandboxed applications use Foundation's
item-replacement directory for a user-selected destination.

All model JSON files reject duplicate object keys before decoding, including
distinct Unicode spellings that are canonically equivalent in Swift. Integer
tokens (including metadata) must lie in the shared range `[-2^63, 2^64-1]`;
decimal/exponent tokens use finite Double semantics. Writers reject metadata
outside this domain instead of silently rounding integers or merging keys.
Typed schema integers, such as `maxNeighbors` and `oodLimit`, use signed Int64;
the unsigned range is available to metadata values.
Cached `q` counts and `qPrime`, similarity CDF entries, and region minimum
similarities cannot exceed `min(maxNeighbors, support.count)` in Float32.
When cached `q` is present, `qPrime <= q` is also required.

## Required manifest fields

```json
{
  "schemaVersion": 1,
  "modelID": "UUID",
  "createdAt": "RFC-3339 timestamp",
  "producer": {"name": "SDMKit", "version": "1.0.0"},
  "configuration": {
    "numberOfClasses": 2,
    "classNames": ["negative", "positive"],
    "embeddingDimension": 3072,
    "exemplarDimension": 1000,
    "maxNeighbors": 2048,
    "qOffset": 2.0,
    "oodLimit": 0,
    "alphaResolution": 0.05,
    "distanceMetric": "squaredL2",
    "neighborTieBreak": "supportIndexAscending"
  },
  "normalization": {"mean": 0.0, "standardDeviation": 1.0},
  "representation": {
    "provider": "precomputed",
    "model": null,
    "revision": null,
    "inputTemplate": null,
    "fingerprint": "user-defined stable identifier"
  },
  "weights": {"file": "weights.f32", "elementCount": 5077002},
  "support": {
    "vectorsFile": "support.f32",
    "recordsFile": "support.jsonl",
    "count": 10000
  },
  "distanceCDFs": [[0.1, 0.2], [0.11, 0.23]],
  "rescaledSimilarityCDFs": [[0.0, 1.2], [0.0, 1.4]],
  "regions": [
    {
      "alpha": 0.95,
      "minimumRescaledSimilarity": 12.4,
      "outputThresholds": [0.96, 0.95]
    }
  ],
  "checksums": {
    "weights.f32": "optional lowercase SHA-256",
    "support.f32": "optional lowercase SHA-256",
    "support.jsonl": "optional lowercase SHA-256"
  }
}
```

All arrays indexed by class have exactly `C = numberOfClasses` entries. CDF
arrays and regions are sorted ascending and descending by alpha, respectively.
An empty class CDF is encoded as an empty array. Checksums may be omitted while
an artifact is being trained, but when present they cover every payload file
in the package (including `calibration.jsonl` when present); release artifacts
should include them.
Class names are nonempty and unique; `qOffset > 1`, `oodLimit >= 0`, and
`0.00005 <= alphaResolution < 0.5`, bounding the ladder to at most 10,000 candidate alpha levels before allocation. Every numeric tensor, CDF, threshold, and cached
diagnostic is finite. Readers reject mismatched dimensions, labels, byte
counts, filenames, checksums, and malformed categorical probability vectors.

## Numerical contract

The v1 contract follows `code/reexpress/sdm_model.py`:

1. The adaptor is two affine layers with no intervening nonlinearity. Input
   normalization uses one global training mean and sample standard deviation,
   not per-feature statistics. The intermediate `M`-vector is the exemplar.
2. Dense matching uses squared L2 distance. Results are ordered by distance and
   then by the original support index for deterministic exact ties.
   Exact matching exhaustively searches candidates using the computed Float32
   distances. Centered norm expansion can still lose very small distances at
   large magnitudes: support `[[0,0], [10001,10000], [10000,10000]]` and query
   `[10000,10000]` can produce a computed zero-distance tie for the last two
   rows, resolved to the lower support index. No precision fallback or reranking
   policy is introduced in this release. Parallel row selection uses the same
   matrix tiles and arithmetic as the serial matcher.
3. The prediction is always `argmax(z')`, including when `d == 0` makes the SDM
   distribution uniform. An argmax tie selects the lowest class index.
4. Similarity `q` counts the consecutive nearest support rows for which both
   `support.predictedLabel == support.label` and
   `support.predictedLabel == queryPrediction`. A training query excludes its
   identity row before calculating `q` and `d0`.
5. Distance quantiles use the exclusionary, left-insertion convention:
   `d_c = 1 - lowerBound(distanceCDF[c], d0) / count`, then
   `d = min_c d_c`. An absent class CDF contributes zero. Consequently `d0 ==
   0` maps to one when all recorded distances are positive, and values above a
   class maximum map to zero.
6. `SDM(z') = softmax(z' * d * ln(2 + q))` and
   `q' = min(q, (2 + q) ^ SDM(z')[prediction])`.
7. The effective sample size uses the inclusive, right-insertion convention:
   `n_c = upperBound(qPrimeCDF[c], q')`. Exact ties are included.
8. For an alpha rung, `epsilon_c = sqrt(ln(2/(1-alpha))/(2*n_c))` when
   `n_c > 0`, and one otherwise. The distance band subtracts/adds the maximum
   epsilon across classes and clamps to `[0,1]`.
9. Alpha rungs are `round(1 - k * resolution, 10)` while alpha is greater than
   `0.5`.

## Nested calibration contract

For each alpha rung, Algorithm 1 runs over the residual calibration rows:

1. Ignore rows with `floor(q') <= oodLimit` when constructing candidates and
   true-class CDFs.
2. Visit the unique residual `q'` values in ascending order.
3. At a candidate value, retain rows with `q' >= candidate`. For every true
   class, sort the SDM output assigned to that true class and select index
   `min(Int(round(1-alpha, 10) * count), count - 1)`. An empty class receives a
   threshold of zero, which prevents accepting an alpha above 0.5.
4. Record the first candidate for which every class threshold is at least the
   rung alpha.
5. Exclude only rows that pass that recorded region exactly as at inference:
   non-OOD, `q'` above the minimum, and a singleton thresholded prediction set
   containing `argmax(z')`.
6. If no finite candidate exists, record no region and exclude no rows.

At inference, walk recorded regions from highest to lowest alpha and assign the
first region whose gates pass. Zero means that no recorded region accepted the
row. The internal similarity gate rejects `floor(q') <= oodLimit` (equivalent
to `q == 0` under the default `oodLimit == 0`); this condition is distinct from
rejection from every region.

## Dataset interchange

The shared JSON Lines schema covers both raw and precomputed rows. Each row
requires `id` and `label`; it may contain `document`, `embedding`, `attributes`,
`logits`, and additional fields. Reexpress two can turn a document-only row
into a feature-ready row with a configured embedding provider. The
framework-neutral SDK intentionally has no implicit provider dependency, so
its numeric train/score/evaluate commands require `embedding`, `attributes`, or
both. A row used for adaptor training must ultimately have one fixed-length
feature vector. `label == -1` means unlabeled and `label == -99` is a
support-only OOD label, matching the Python conventions.

All splits participating in one trained artifact must have identical
representation fingerprints and embedding dimensions. Implementations reject,
rather than silently coerce, incompatible splits.

## Portable training history and continuation

New trainers may include `metadata.sourceData`, containing ordered SHA-256
digests of the original Float32 input vectors. These metadata-only digests permit
optional text/feature attachments without embedding original inputs in the model.
The [source attachment contract](model-source-attachments-v1.md) defines the
digest encoding and the original/winning split provenance. Their absence remains
backward compatible and does not affect inference or calibration.

Trainers write completed history to `metadata.trainingRun`, conforming to
`contracts/training-run-v1.schema.json`. This contains all completed iterations
and epochs, the selected iteration/epoch, options, backend, early-stop status,
and optional `sourceModelID`. It contains no local project or dataset UUIDs.
The seed is a decimal UInt64 **string** to avoid loss through JSON runtimes
that represent numbers as doubles. Imported history is available for examination
in Reexpress two; source datasets must still be supplied for further training.
The final `bestIteration`/`bestEpoch` pair identifies the exported checkpoint.
An epoch's `isBest` flag records an improvement at that phase of training and
does not by itself identify the final winner.

Each epoch uses explicit `balancedTrainingAccuracy`,
`balancedCalibrationAccuracy`, `balancedMeanTrainingQ`, and
`balancedMeanCalibrationQ`. Each metric first averages within each **true
class**, then averages the class means with equal weight. Accuracy is therefore
mean per-class recall. Training and calibration require every configured class;
missing classes are rejected. Balanced CE and SDM losses use the same class
weighting. `marginalTrainingLoss` is the distinct optimizer diagnostic, not the
displayed balanced SDM training loss. Unmeasured values are null, never zero.

The standard one-CE-epoch matching and checkpoint schedule is unchanged. If
`crossEntropyEpochs > 1`, those leading epochs use `q = e - qOffset`, `d = 1`,
and forward-only balanced CE evaluation, without matching training or calibration
exemplars. Only the best balanced-CE calibration checkpoint is subsequently fully
SDM-scored as a candidate. Training proceeds from the last CE weights and live
optimizer; if the CE winner differs, the last CE training q/d are prepared once.
Later candidates are compared using balanced SDM calibration loss; CE and SDM
values are never compared directly. All-CE runs and Stop during CE finish SDM
evaluation of the CE winner. No partially scored epoch is eligible.

Stop before a fully evaluated epoch cancels the attempt. Otherwise it completes
the current epoch/scoring, selects the best complete checkpoint, and skips later
epochs and iterations. Cancellation aborts even during required final scoring.
Continuation uses the source artifact's weights, normalization, architecture,
and activation constants; starts a new Adam optimizer; and validates the exact
class order, feature dimension, and representation fingerprint. Every J iteration
starts independently from the same saved weights. Export creates a new model ID.

History records the backend's checkpoint-selection measurements. Canonical
Float32 artifact finalization does not rewrite the history or reselect the
winner. A separately recorded canonical final loss can differ slightly because
matching and arithmetic can differ across backends.

## Score interchange

The framework-neutral numerical score uses `qPrime` and `qPrimeLower` for q′
and its lower estimate. Region assignments are `centroidRegionAlpha` and
`lowerRegionAlpha`; zero means rejection from all calibrated regions.
`cumulativeEffectiveSampleSizes` contains the inclusive, class-indexed CDF
counts used for the rung-specific `effectiveSampleSizeErrors`.
`isInMostConservativeRegion` and `isInMostConservativeRegionLower` are derived
flags that are true exactly when the corresponding assignment equals the
artifact's highest recorded alpha. Calibration and region membership apply the
model's internal similarity gate before accepting a region.

Transport context is deliberately layered on top of that numerical value.
Native `SDMScore` values remain independent of a request or artifact envelope
and identify the exemplar by `nearestSupportIndex`. Reexpress two JSONL exports are
labeled analysis records
that additionally contain dataset, split, label, document, and support-ID
context.

`contracts/scored-document-v1.schema.json` defines the uploadable full-score
form. It preserves the input `id`, `label`, document, selected embedding and/or
attributes, and metadata; adds the complete numerical fields above; and requires
`modelID`, `representationFingerprint`, `scoreSchemaVersion: 1`, and
`matchingSemanticsVersion: 2`. Reexpress two assigns its own dataset ID and split.
Python `score_document` returns one such object; the CLI's default `score --detail
full` writes this form. Compact reports are a separate format.

Python scores optionally add `nearestSupportMatches`, an array sorted by squared
L2 distance then support index. Each entry contains `supportIndex`, `id`, `label`,
`predictedLabel`, `squaredDistance`, and optional `document` text from the model's
support record. The default display count is 25, capped by eligible support rows;
zero omits the list. The singular `nearestSupportIndex` and `nearestSupportID`
remain unchanged. This diagnostic count is independent of `maxNeighbors` and
must not affect any calibration or scoring quantity. Explicitly excluded support
identities are also excluded from the display list. Reexpress two preserves an
imported list with its producer scores, strips it from source-only exports, and
drops it when native rescoring replaces those scores; its existing on-demand
nearest-match visualization is unchanged. The additive field does not change
the artifact, dataset, score, or matching semantics version.

An explicit training-self exclusion includes both `excludedSupportIndex` and
`excludedSupportID`. Both fields are omitted when no support row was excluded.
It is never inferred from a matching external document ID.
The source label and projected feature vector must agree with that support row.
Reexpress two binds the validated exclusion to the imported document's own identity
and model package, preserving it on rescoring and model reactivation. Numerical
edits invalidate this evidence. Without an explicit exclusion the row uses
ordinary inference matching.

Import requires the same active model and representation. It validates dimensions,
finite values, probability vectors, CDF/DKW diagnostics, and region assignments
before committing documents and scores together. It preserves valid producer
measurements within the documented numerical tolerances. Import does not repeat
the full dense matching search. Rescoring retained source features runs the
normal app scoring path, with the same exclusion when one was established.
