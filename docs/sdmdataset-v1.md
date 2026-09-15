# SDM dataset bundle, version 1

An `.sdmdataset` is an uncompressed directory package shared by Python and Swift.
It stores readable row metadata separately from optional cached numeric features.
It does not contain a trained model or an app-local split assignment. JSONL remains
supported. There is no requirement to install Python on the Mac to use this format.

## Files and manifest

`manifest.json` is UTF-8 JSON, at most 8 MiB, with these fields:

```json
{
  "format": "reexpress-dataset",
  "schemaVersion": 1,
  "rowCount": 3,
  "rows": {"path": "rows.jsonl", "sha256": "64 lowercase hexadecimal digits"},
  "matrices": {
    "embedding": {
      "path": "embeddings.npy", "dtype": "<f4", "shape": [2, 4],
      "sha256": "64 lowercase hexadecimal digits"
    }
  },
  "classNames": ["Negative", "Positive"],
  "representation": {"fingerprint": "example:embedding-v1"},
  "metadata": {"name": "Example dataset"}
}
```

`format`, `schemaVersion`, positive integer `rowCount`, `rows`, and `matrices`
are required. `matrices` may be empty; its only permitted keys are `embedding`
and `attributes`. Paths are exactly `rows.jsonl`, `embeddings.npy`, and
`attributes.npy` respectively. Every referenced file must be a regular file
inside the package; symbolic links and path traversal are rejected. SHA-256
digests cover each entire referenced file, including NPY headers, and are checked
by default. Numeric matrices must also match their manifest shape and dtype.

`classNames`, `representation`, and `metadata` are optional. Class names, when
present, contain at least two distinct, nonempty names and determine the meaning
of integer labels by array index. Python imposes no upper class count. Reexpress two
permits at most five classes and checks compatibility with the target project.
Representation is an object with a nonempty string `fingerprint` and may contain
additional JSON provenance fields. Metadata is an arbitrary JSON object.
Optional objects and arrays are omitted when absent, rather than encoded as null.
Row-level representation fingerprints must agree with one another and with the
manifest when present. Readers may infer the fingerprint from rows if the manifest
omits it. Class names must not have leading or trailing whitespace. Readers ignore
unreferenced extras, such as Finder's `.DS_Store`; exporters do not copy those files.
Fingerprint whitespace is significant: readers and project imports check for a
nonempty token without stripping its leading or trailing whitespace. Reexpress two
validates declared row fingerprints on JSONL imports as well as bundle imports.
For both Python and Reexpress two, an explicitly present top-level
`representationFingerprint` takes precedence over the legacy
`metadata.representationFingerprint`. The nested value is a fallback only when
the top-level key is absent. The effective value must be a nonempty string
(null is invalid) and must agree across rows and with a declared bundle
manifest or requested model fingerprint. Ignored legacy metadata is preserved.

## Metadata rows and missing features

`rows.jsonl` uses the existing dataset-row semantics: unique nonempty string `id`,
integer `label` (a class index, `-1` for unlabeled, or `-99` for support-only OOD),
and optional document, logits, metadata, label-edit provenance, and other source
fields. Source text may be empty or absent. Arbitrary JSON input/token fields and
existing portable scored-document fields are retained. JSON numbers must be
finite. This does not introduce a tokenizer or a differentiable training API.

When present, source `logits` must contain at least two finite numeric values
that remain finite in Float32; booleans are not numbers. The legacy `sourceLogits`
alias has the same requirements and cannot coexist with `logits`. Python retains
the original numeric values; Reexpress two stores source logits as Float32 and exports
the canonical `logits` key.

Reexpress two rejects duplicate JSON object keys before decoding source rows or
manifests, including distinct Unicode spellings that compare equal under canonical
equivalence (for example, `"\u00e9"` and `"e\u0301"`). It also rejects such
collisions between IDs within one dataset with an explicit error. This prevents
silent loss in Swift string dictionaries; it does not normalize or rename input
keys or IDs. Python can distinguish these spellings, so avoid these collisions
when preparing data for Reexpress two.

Imported `label_changed_at` ISO 8601 strings may include fractional seconds.
Reexpress two retains their source spelling and precision across project storage and
dataset exports until the timestamp is changed. Invalid timestamp values are
rejected rather than silently omitted. Existing numeric timestamps use seconds
since the Foundation reference date, 2001-01-01T00:00:00Z.

Source fields retain their original top-level keys, independently of the
`metadata` object. `document` alone supplies document and exemplar display text.
`prompt` is an independent source field retained on export; it is never used as
a fallback when `document` is missing, null, or empty.
Reexpress two preserves integer metadata and provenance exactly from Int64 minimum
(-9223372036854775808) through UInt64 maximum (18446744073709551615), including
nested JSON values. Larger integer literals are rejected with an actionable
error; encode larger identifiers as strings for portable use. Python can retain
larger integers. Fractional and exponent-form JSON numbers use Float64 semantics
in Reexpress two; cached feature vectors use Float32 as specified below.

Optional model source companions use manifest `metadata.modelSource` and row
`metadata.sdmSource` to bind original text and optional features/scores to an exact
saved model and its winning split membership. They remain ordinary dataset
bundles; see the [source attachment contract](model-source-attachments-v1.md) for
validation and independent training/calibration attachment semantics.

There are no inline `embedding` or `attributes` arrays in `rows.jsonl`.
Instead, a row with cached features uses the reserved integer `embeddingRow`
and/or `attributesRow` fields:

```json
{"id":"a","label":0,"document":"First document","embeddingRow":0}
{"id":"b","label":-1,"document":"Not yet embedded"}
{"id":"c","label":1,"document":"Third document","embeddingRow":1}
```

For each feature independently, references must be exactly `0, 1, ..., M-1`
in metadata-row order, where M is that matrix's first dimension. Each matrix row
is referenced once; absent references mean absent features. Thus partially
embedded datasets need neither padded zero vectors nor a separate validity mask.
`embeddingRow` and `attributesRow` are storage references, never user metadata.
The number of nonblank metadata rows must equal `rowCount`.

Scored documents retain the existing complete score schema and its model ID,
fingerprint, and explicit support-exclusion evidence. Importing cached scores
still applies the existing model/feature validation. A bundle does not confer
trust on arbitrary incoming scores. Raw dataset export and score export remain
distinct operations.

## Restricted NPY profile

Writers emit standard **NPY 1.0**, readable by `numpy.load(..., allow_pickle=False)`.
Readers accept NPY 1.0, 2.0, and 3.0 subject to the following profile:

- Header at most 65,536 bytes, with exactly `descr`, `fortran_order`, and `shape`.
- `descr` is exactly `'<f4'`: little-endian IEEE Float32.
- `fortran_order` is `False`: C/row-major storage.
- `shape` is a tuple of two positive integers `(M, D)`.
- All values are finite and the payload is exactly `4*M*D` bytes, without trailing
  data. Structured, object/pickle, Fortran-order, other-endian, and other-dtype
  arrays are rejected rather than implicitly converted by readers.

Writers may convert finite numeric inputs to Float32, but must reject overflow,
booleans, empty vectors, and inconsistent feature dimensions. The profile avoids
general Python-object interpretation in Swift. Its native parser reads the small
NPY header and maps the raw numeric payload directly; no converter is required.
See the [official NPY specification](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html).

## Access, ownership, and writing

Readers can map matrices and hydrate only requested rows. Metadata-only bundles
can be inspected, reviewed, and exported. Training/scoring require complete
selected features and produce an actionable error when they are absent.
Memory mapping does not eliminate checksum/finite-value validation or device
transfers. A consumer combining separately stored embedding and attribute matrices
may need to allocate a combined matrix or concatenate bounded batches.

Reexpress two may keep an app-local provisional identity for source-only imports
that declare no representation. The first matching-ID import that supplies cached
features can assign that dataset's actual identity, including during a partial
refill. This does not permit replacing a declared identity or an established
feature schema. Adoption and row updates commit in the same transaction.

Writers build a complete sibling temporary directory, finalize matrix headers and
checksums, and publish only a complete package. Cancellation/failure must preserve
any preexisting destination. Import into Reexpress two copies features into managed
project storage in bounded rows; the project never depends on the continued
existence of the original bundle. This release supports one matrix per feature;
sharding is not part of version 1.
