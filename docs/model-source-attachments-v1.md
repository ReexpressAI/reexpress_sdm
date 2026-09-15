# Optional model source attachments, version 1

An attachment is an ordinary `.sdmdataset` bundle. It adds original document
content, optional cached input features, and optional full scores to an existing
model without changing its weights, support vectors, calibration, or identity.
Training and calibration inputs can be exported and attached independently.
The model and dataset formats remain schema version 1.

## Binding and row membership

The dataset manifest contains `metadata.modelSource`:

```json
{
  "schemaVersion": 1,
  "modelID": "the-model-id",
  "modelFiles": {
    "manifest.json": "lowercase SHA-256 of exact file bytes",
    "weights.f32": "lowercase SHA-256",
    "support.f32": "lowercase SHA-256",
    "support.jsonl": "lowercase SHA-256",
    "calibration.jsonl": "lowercase SHA-256, when that file exists"
  },
  "role": "original-training"
}
```

`role` is `original-training`, `original-calibration`, `selected-training`, or
`selected-calibration`. File digests bind the attachment to the exact immutable
model package, including metadata. A shared model ID alone is insufficient.
Re-serializing or modifying the model generally requires exporting new
companions. A consumer must check all listed digests and require the complete
set of model files, including calibration when present.

Reexpress two also permits an attachment to a saved local descendant produced only
by alpha-ladder recalibration. The companion must still match the exact retained
ancestor package, and Reexpress two must verify the explicit `metadata.recalibration`
`sourceModelID` chain through project history. Weights, support vectors, support
records, and calibration rows must be byte-identical; normalization,
representation, and configuration other than alpha resolution must be unchanged.
A shared model ID or missing ancestor package is insufficient. This allows a
missing companion to be attached after local recalibration without accepting
unrelated models or numerical changes.

Every row contains `metadata.sdmSource`:

```json
{
  "poolIndex": 12,
  "selectedSplit": "training",
  "selectedIndex": 3,
  "featureSHA256": "lowercase SHA-256 when the feature digest is known"
}
```

Indexes are zero based. `selectedIndex` addresses support records for `training`
or saved calibration rows for `calibration`. `poolIndex`, when available,
addresses `[original training; original calibration]`. All indexes, IDs, labels,
and row ordering must agree with the model; consumers must not infer support
identity from a matching user-facing ID alone. Each original-role bundle has the
complete corresponding original split, in original order. Each selected-role
bundle has the complete corresponding winning split, in its saved order.

Original roles require valid `metadata.bestIterationSplits`: both index arrays
must partition the original pool exactly, and their aligned IDs must agree with
the model's support and calibration records. Its `indexConvention` must be
`zero-based in original training followed by original calibration`, and the sum
of original counts must equal the number of saved source records. A selected-role attachment can also
target a directly trained model without original-pool metadata; `poolIndex` is
then absent. Attaching only one original file after shuffling may fill portions
of both selected splits. Consumers show incomplete coverage until other source
rows are attached.

## Input feature evidence

New Python training artifacts add `metadata.sourceData`:

```json
{
  "schemaVersion": 1,
  "featureDigestAlgorithm": "sha256-f32le-v1",
  "trainingFeatureDigests": ["one digest per support record, in order"],
  "calibrationFeatureDigests": ["one digest per calibration row, in order"]
}
```

The digest is SHA-256 of one composed **raw input vector**, before model
normalization or projection, encoded as contiguous little-endian IEEE Float32
bytes. Composition is embedding followed by attributes, or whichever single
component was used for training. Normalize negative zero to positive zero before
hashing. There is no header, dimension prefix, or other framing. Nonfinite values
and boolean features are invalid. Digests in the attachment must agree with the
model's stored digest when available, and present features must reproduce that
digest. A text-only export may carry the model's expected digest without claiming
to have supplied or verified any feature values.

For interoperability checks, `[1.0, -0.0, -2.5]` and `[1.0, 0.0, -2.5]` both hash
to `865df0fca3c0d4db824a59a9e3b6b6cd7fd45e9b968dfc71f480574991c95063`.

Models without these digests, including older Python models and current native
Swift models, remain attachable. With full features, consumers can compare
projected training vectors to saved support vectors, or calibration logits to
saved calibration logits, using the existing Float32 tolerance. Such comparisons
provide projected numerical compatibility, not proof of identical raw input
features. Text-only ID/index/label attachments provide display content, never
numerical identity evidence or a reason to exclude a support row from matching.

## Features, scores, and continuation

Attachments use normal dataset feature matrices and representation fingerprints;
class names preserve the model's class order. Missing features stay missing.
Source `embedding` and `attributes` fields not used in the selected training
composition are omitted, preventing the app from silently changing composition.
Existing user metadata is retained except the reserved `sdmSource` key, which the
exporter replaces with validated membership.

Python `sdm dataset export-sources` exports one companion atomically. Either
original input may be supplied independently; selected-role output requires all
rows of that selected split across the supplied files. `--text-only` omits cached
features. `--with-scores` computes fresh full portable scores and requires full
features; these two flags cannot be combined. Training rows exclude their
explicitly validated support index; calibration rows never receive that
exclusion. Incoming score fields are discarded unless replaced by this fresh
scoring operation. Ordinary evaluation bundles remain independent of source
attachments.

Attaching content does not modify the numerical model or invalidate its existing
evaluation scores. Continuing training requires complete features and labels in
the selected training and calibration splits, with a fresh optimizer. Existing
alpha-ladder recalibration uses the saved calibration diagnostics and does not
require text or original input features.
