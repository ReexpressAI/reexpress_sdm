# reexpress_sdm

*Efficient SDK for Similarity-Distance-Magnitude (SDM) calibration, and the geometric analysis therein, of neural networks. PyTorch backend (CPU, MPS, or CUDA) for mac and Linux. Provides interoperable data formats with Reexpress two, an on-device macOS platform for Actionable Interpretability of language models.*

***

`reexpress_sdm` trains and runs Similarity-Distance-Magnitude (SDM) estimators
with PyTorch on CPU, Apple silicon (MPS), or NVIDIA GPUs (CUDA). It provides
exact nearest-exemplar matching, nested calibration, per-document uncertainty
estimates, aggregate evaluation reports, and the `sdm` command-line interface.

Models (`.sdmkitmodel`) and datasets (`.sdmdataset` or JSON Lines) can be exchanged
with the separate macOS app, Reexpress two. This repository
contains the Python package; building and using it does not require the macOS
app, Swift, or files from another repository. NumPy provides host arrays,
calibration math, and reporting; PyTorch is the execution backend.

Licensed under [Apache-2.0](LICENSE). See [re.express](https://re.express) for
project information and the [original research implementation](https://github.com/ReexpressAI/reexpress_mcp_server)
for the research baseline.

## Install

Use Python **3.10 or newer**. From the root of a checkout named `reexpress_sdm`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
sdm --help
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
The package installs NumPy (>=2.0) and PyTorch (>=2.5); it needs no optional
execution-backend extra. For CUDA, use a PyTorch installation compatible with
your GPU and driver. Device availability depends on that PyTorch installation.

Multiword options use underscores. Each command's help includes concise
explanations and defaults, for example `sdm train --help` or
`sdm dataset convert --help`. Some defaults resolve from the input metadata or
available hardware.

## Quick start

Prepare separate labeled training and calibration files with precomputed
features. A JSON Lines file contains one JSON object per line, for example:

```jsonl
{"id":"example-1","label":0,"document":"Example text","embedding":[0.1,0.2,0.3]}
{"id":"example-2","label":1,"document":"Other text","embedding":[0.4,0.5,0.6]}
```

These rows illustrate the format; use full datasets for training and calibration.
Labels are zero-based class indexes, and each split must contain every class.
Selected feature vectors must have the same length throughout the model's data.
`document` is optional display text. The package consumes precomputed vectors;
it does not generate embeddings.

```bash
# Train an adaptor and calibrate its nested SDM estimator.
sdm train --training train.jsonl --calibration calibration.jsonl \
  --number_of_classes 2 --representation_fingerprint embedding_v1 \
  --output model.sdmkitmodel

# Score documents; labels are optional.
sdm score --model model.sdmkitmodel --input evaluation.jsonl \
  --output scored.sdmdataset

# Summarize performance on a fully labeled evaluation set.
sdm evaluate --model model.sdmkitmodel --input evaluation.jsonl \
  --output report.json
```

Training uses automatic device selection: CUDA, then MPS, then CPU. Set
`--device cpu`, `--device mps`, or `--device cuda` to choose explicitly; scoring
and evaluation use `--matching_device` for the same purpose.

By default, training **pools and shuffles** the supplied training and calibration
rows, then divides them into two nearly equal splits. Add `--do_not_shuffle_data`
to retain the supplied memberships and sizes. The default feature composition
uses `embedding`, `attributes`, or their concatenation in that order when both
are present. Set `--composition embedding` or `--composition attributes` to use
only one field; use the same convention for training and subsequent scoring.

`sdm score` saves each document's prediction and diagnostics. `sdm evaluate`
writes aggregate statistics, including cumulative coverage at every calibrated
alpha in the model; it does not save individual scored documents. Both use the
same scoring algorithm. See [CLI reference](#cli-reference) for details.

Continue with:

- [JSONL and binary datasets](#jsonl-and-binary-datasets)
- [Training iterations and split shuffling](#j-independent-training-iterations-and-split-shuffling)
- [Continued training](#ce-phases-balanced-metrics-and-continuation) and [recalibration](#recalibrate-an-existing-model)
- [Model source attachments](#optional-model-source-attachments) and [Python/app score exchange](#upload-python-scored-documents-to-reexpress-two)
- [Python API](#python-api)
- [Tests and distribution builds](#tests-and-distribution-builds)

## JSONL and binary datasets

An `.sdmdataset` is a directory package containing `manifest.json`, `rows.jsonl`,
and optional `embeddings.npy` / `attributes.npy` matrices. Numeric features are
little-endian, row-major Float32. Metadata remains readable. Reexpress two reads and
writes this format natively; it needs neither NumPy nor a Python converter.

```bash
sdm dataset convert --input data.jsonl --output data.sdmdataset \
  --class_names Negative,Positive --representation_fingerprint embedding_v1
sdm dataset inspect --input data.sdmdataset
sdm dataset validate --input data.sdmdataset
sdm dataset convert --input data.sdmdataset --output restored.jsonl
sdm score --model model.sdmkitmodel --input data.sdmdataset \
  --output scored.sdmdataset --matching_device mps
```

All dataset-consuming model commands accept either format. Full score output may
also be a bundle, retaining the same portable scores as full JSONL output; compact
diagnostic output remains JSONL. Import the matching model before importing scored
data in Reexpress two. Model operations reject bundles whose declared class names or
class order differ from the model. Bundles without declared names use integer
labels in the model's class order. Conversion accepts missing JSONL labels and
records them as `-1` (unlabeled); the canonical stored contract requires a label.
When present, source `logits` must contain at least two finite numbers that remain
finite in Float32; booleans are rejected. The legacy `sourceLogits` alias follows
the same checks, and a row cannot contain both fields. Python preserves the
original JSON numbers and field spelling when converting either format.

`document` is the display text for a source row or retained nearest exemplar.
`prompt` is an independent source field, preserved through JSONL and `.sdmdataset`
exports. It is never substituted when `document` is missing, null, or empty.

```python
from reexpress_sdm import Dataset, DatasetBundle, write_dataset_bundle

write_dataset_bundle("raw.sdmdataset", [
    {"id": "a", "label": 0, "document": "Cached", "embedding": [0.1, 0.2]},
    {"id": "b", "label": -1, "metadata": {"tokenIDs": [12, 34]}},
], class_names=["Negative", "Positive"],
   representation={"fingerprint": "embedding_v1", "pooling": "last"})

with DatasetBundle.open("raw.sdmdataset") as bundle:
    print(bundle.manifest)
    for row in bundle:
        print(row["id"], "embedding" in row)

# For a fully featured bundle, a single selected matrix stays memory-mapped.
dataset = Dataset.load("data.sdmdataset", composition="embedding")
vectors = dataset.vectors
```

Cached embeddings and attributes may be absent from some or all rows. Missing
features remain missing. Source-only data can be stored, inspected, reviewed in
Reexpress two, and exported. Training/scoring require complete selected features;
these APIs do not generate embeddings. Documents can be blank or omitted.

Python maps metadata and matrices and hydrates rows as requested. Combining both
feature columns allocates a combined matrix, and device execution still transfers
data to Torch. Validation reads checksums and verifies finite values; mapping does
not make validation free. Reexpress two imports into its managed project database in
bounded rows and does not rely on the external bundle remaining present. In Data,
use **Import Data** to import or a card's **Export dataset** menu to save
JSONL or a bundle. Imported class names must match the project's order and may
contain at most five classes; Python has no five-class limit.

The [dataset specification](docs/sdmdataset-v1.md) defines the exact
NPY subset, sequential references for partial features, SHA-256 checks, and atomic
publication rules. The [model specification](docs/sdmkit-artifact-v1.md) defines the shared
`.sdmkitmodel` format.

### PyTorch execution

Training defaults to PyTorch. `--device` accepts `cpu`, `mps`, `cuda`, or `cuda:N`; `auto` prefers CUDA, then MPS, then CPU. Choose CPU explicitly for CPU measurements:

```bash
sdm train --device cpu \
  --training train.jsonl --calibration calibration.jsonl \
  --number_of_classes 2 --representation_fingerprint my_embedding_v1 \
  --output model.sdmkitmodel
```

Use `--device mps` on Apple silicon or `--device cuda` on a CUDA machine. `TorchTrainer` is also available directly from `reexpress_sdm`. PyTorch performs optimization, adaptor forwards, and exact support matching, including selected-checkpoint finalization. Saved weights and projected support are contiguous Float32 arrays in the portable artifact. Empirical CDFs, nested region fitting, and score diagnostics use shared host math. Training trajectories and values near numerical boundaries can vary slightly across devices.

Inference also defaults to PyTorch. Model weights and support are retained on the chosen device, and both adaptor projection and matching use that runtime:

```python
from reexpress_sdm import SDMModel

model = SDMModel.load("model.sdmkitmodel", device="mps")
scores = model.score(query_vectors)
```

```bash
sdm score --model model.sdmkitmodel --input eval.jsonl --output scores.jsonl \
  --matching_device mps
```

The `--matching_backend` and `--matching_device` options configure the entire inference runtime, including the adaptor forward pass. Matching tiles both queries and support. Equal computed Float32 distances are ordered by ascending support index.

For direct PyTorch matching:

```python
from reexpress_sdm import create_dense_index

index = create_dense_index(
    "torch", support_vectors, device="mps",
    query_batch_size=256, support_tile_size=16384,
)
distances, support_indices = index.search_many(query_vectors, k=25)
```

## CLI reference

Multiword options use underscores. Run `sdm --help` to list commands, then
`sdm train --help`, `sdm recalibrate --help`, `sdm score --help`, or a nested command such as
`sdm dataset convert --help` to see its options, brief explanations, and defaults.
Values such as the device, feature composition, class names, and representation
identity are resolved from available hardware or input metadata when omitted.

```bash
sdm artifact validate --model path/to/model.sdmkitmodel
sdm artifact inspect --model path/to/model.sdmkitmodel
sdm train --training train.jsonl --calibration calibration.jsonl \
  --number_of_classes 2 --representation_fingerprint my_embedding_v1 \
  --output model.sdmkitmodel
sdm score --model path/to/model.sdmkitmodel --input eval.jsonl --output scores.jsonl
sdm evaluate --model path/to/model.sdmkitmodel --input labeled.jsonl --output report.json
```

For training and scoring, JSON Lines rows contain `id` and fixed-length selected
features. Training and evaluation rows also contain an integer `label`. With the default
`--composition auto`, an `attributes` array is used alone, or concatenated after `embedding`
when both are present. The explicit alternatives are `embedding`, `attributes`,
and `embedding+attributes`. Use `--representation_fingerprint` to require an
exact match with the artifact's representation fingerprint.

`sdm score` writes one prediction and its uncertainty diagnostics per input
row. Labels are optional, and full output retains the source fields for import
into Reexpress two. `sdm evaluate` runs the same scoring algorithm on labeled data
and writes an aggregate JSON report: accuracy, class-conditional accuracy,
coverage at calibrated alpha levels, and score-distribution summaries. It does
not write individual scored documents or fit a new model.

The report's `distribution` section contains per-signal counts, minima, maxima,
means, quantiles, and histograms, plus predicted-class frequencies. These are
single-dataset descriptive statistics. `--histogram_bins` sets the histogram
bin count (10 by default); individual signal values remain available through
`sdm score` for the caller's own statistical comparisons.

Selection decisions from the Python API use the centroid estimator by default.
For DKW lower decisions, use `SelectionPolicy(estimator=EstimatorKind.LOWER)`.
Set `SelectionPolicy(minimum_alpha=0.95)` to require a minimum accepted alpha.
`sdm score` and `sdm evaluate` retain results for both estimators. Evaluation
reports cumulative results at every calibrated alpha saved in the model.

The portable dataset-row contract requires `label` (`-1` for unlabeled and
`-99` for support-only OOD). As a scoring convenience, `sdm score` also accepts
rows with no label and treats them as unlabeled. Both `sdm train` and
`sdm evaluate` require a known class label in `0 ..< numberOfClasses` for every
row. The Python evaluation API can also summarize mixed labeled/unlabeled/OOD
scores, excluding sentinels from accuracy and coverage and reporting their
counts.

For new training, omitted representation metadata uses fingerprint
`embedding_v1`, provider `precomputed`, and an unset representation model.
The fingerprint identifies compatible feature representations; it is not an
automatically computed hash. Provider and model describe the precomputed
features and do not call an embedding service. Set `--representation_fingerprint`,
`--representation_provider`, and `--representation_model` when those values are
known. Explicit or dataset-declared fingerprints are checked for conflicts;
omitted values inherit the initial model during continuation before falling
back to the defaults. Scoring and conversion retain declared identities and
never replace them with the generic training fallback.

New class names default to `Class0`, `Class1`, `Class2`, and so on. Use
`--class_names` for ordered comma-separated names. Dataset-declared names and
continuation model names are retained and checked for a consistent class order.

Cached calibration rows are validated as Float32 categorical distributions;
each probability vector must sum to one within absolute tolerance `1e-5`.

Training uses global-scalar sample normalization, the two-affine-layer adaptor, first-epoch cross-entropy-equivalent SDM settings, and then updated `q` and `d`; the best complete epoch minimizes balanced calibration SDM loss. Exact matching uses bounded query/support tiles. Nested calibration uses an ordered counting sweep per alpha. These choices preserve the portable numerical contract while keeping matrix operations on the selected runtime.

Device kernels and query batches can produce slightly different Float32 distances near ties. `ExactL2Index(..., batch_invariant=True)` uses single-query Torch execution when batch invariance is needed, at a throughput cost. It does not make results bit-identical across devices.

### J independent training iterations and split shuffling

The CLI and `train_iterations` Python API support J independent adaptor fits. To use
the pooled shuffle workflow:

```bash
sdm train \
  --training train.jsonl \
  --calibration calibration.jsonl \
  --output model.sdmkitmodel \
  --number_of_classes 2 \
  --representation_fingerprint my_embedding_v1 \
  --exemplar_dimension 1000 \
  --epochs 50 --batch_size 64 --learning_rate 0.00001 \
  --seed 0 --max_neighbors 2048 \
  --alpha_resolution 0.05 \
  --number_of_random_shuffles 5 \
  --backend torch --device mps \
  --report_output training-report.json
```

Omit the backend/device options to use PyTorch with automatic device selection. The SDK CLI defaults
to J=1 with pooled split shuffling enabled, including before iteration 1.
`--do_not_shuffle_data` retains the supplied split memberships and sizes.
`--shuffle_training_and_calibration` explicitly enables the default shuffling behavior. With shuffling disabled, J>1 uses the same supplied splits.
Fresh runs use different initializations; continued runs each start from the
same saved weights with a fresh optimizer.

With shuffling enabled, each iteration uniformly permutes the original pooled
training+calibration rows, then takes `floor(N/2)` for training and the remainder
for calibration. This is not stratified and does not preserve unequal input
split sizes. A shuffle that omits a class fails with an explanation; it is not
silently retried. Iteration i (zero-based) uses seed `(seed + i) % 2**64` for its
permutation and fresh trainer. This is reproducible within a backend, not a
promise to reproduce the research PyTorch or Swift RNG stream bit-for-bit.

The winning epoch and iteration minimize balanced calibration SDM loss, using
the last epoch/iteration on exact ties. The selected artifact contains exactly
that iteration's support rows and calibration rows. All iteration summaries,
seeds, source-pool indices, and selected IDs are preserved in manifest metadata
(`bestIterationSplits`); indices are zero-based in `[original training;
original calibration]`. The model's `metadata.trainingRun` and optional report
include the complete epoch history. `durationSeconds` records the
complete training attempt across all iterations through final calibration, excluding
input-file parsing and artifact export. Each epoch also records `durationSeconds`;
deferred CE scoring is charged to its winning epoch. The Python result exposes
`duration_seconds`. Reexpress two reads this history and total duration when importing
the model. Models without saved timing remain valid; their times cannot be recovered. Only the selected model is exported, avoiding J duplicate artifacts.
This is checkpoint selection, not an independent validation-set guarantee.

### CE phases, balanced metrics, and continuation

`TrainingConfig(cross_entropy_epochs=1)` keeps the standard schedule. With a
value greater than one, leading CE epochs skip all exemplar matching and measure
balanced CE train/calibration losses. At the transition, the best balanced-CE
calibration checkpoint is fully SDM-scored as a candidate. Training continues
from the last CE weights and optimizer state; its training q/d are prepared if
it differs from the CE winner. Later checkpoints compete on balanced SDM loss.
An all-CE run fully scores its CE winner before final calibration and export.
This multi-epoch warm-up is an experimental efficiency option: only the selected
best CE epoch is SDM-scored, intentionally excluding the other CE epochs from
SDM checkpoint competition.

History uses `balancedTrainingAccuracy`, `balancedCalibrationAccuracy`,
`balancedMeanTrainingQ`, and `balancedMeanCalibrationQ`: average within each
true class, then average class means equally. Both splits must contain every
configured class. The separate CE/SDM loss fields use the same weighting.
Unavailable measurements are `None`/JSON null, including q before matching.
`trainingLoss` is the separate marginal batch optimizer diagnostic.

Continue from a portable model with `initial_artifact=load_artifact(path)` in
`TorchTrainer.fit` or `train_iterations`; the CLI equivalent
is `--initial_model`. Continuation retains normalization, architecture, weights,
and activation constants, resets Adam, and writes a new model. Each J starts
independently from those saved weights. Class order, feature dimension, and
representation must match. For example, on a CUDA server:

```bash
sdm train --backend torch --device cuda \
  --initial_model previous.sdmkitmodel \
  --training updated-training.jsonl --calibration updated-calibration.jsonl \
  --number_of_classes 2 --representation_fingerprint my_embedding_v1 \
  --epochs 20 --cross_entropy_epochs 10 --learning_rate 0.000001 \
  --output continued.sdmkitmodel
```

The same artifact can be imported into Reexpress two, inspected, and continued with
CPU/Accelerate or MLX. The imported history displays its original backend while
the app's compute selector controls the next local run. Optimizer state is not
serialized. Accelerator selection losses remain the history/selection values;
selected-checkpoint finalization uses the same device and does not rerank checkpoints.

The Python API accepts a `TrainingControl`. Calling `request_stop()` from a
progress callback or another thread finishes the current epoch/scoring, saves
the best complete checkpoint, and skips later epochs/J iterations. Before any
complete epoch it requests cancellation and returns `False`. `cancel()` raises
`TrainingCancelled` at the next cooperative boundary, including final scoring,
and no result artifact is returned. Use a fresh control per attempt.

### Recalibrate an existing model

Use `sdm recalibrate` to fit a new set of nested regions from the model's
saved calibration diagnostics, for example with a finer alpha grid:

```bash
sdm recalibrate --model original.sdmkitmodel \
  --alpha_resolution 0.01 --output recalibrated.sdmkitmodel
```

This command requires a model that retains its calibration diagnostics.
It needs no training or calibration dataset, does not train the adaptor, and
does not rerun exemplar matching. The output is a complete `.sdmkitmodel`
artifact with the existing weights and support and newly fitted calibration
regions, ready to score in Python or import into Reexpress two. Use `--report_output`
to save its JSON report instead of printing it to stdout. Continued training
uses `sdm train --initial_model` with labeled training and calibration data.

For training, recalibration, and imported models, alpha resolution must be finite
and satisfy `0.00005 <= alpha_resolution < 0.5`. This limits the candidate ladder
to 9,999 levels.

Model output defaults to a new destination. `--overwrite` (or
`write_artifact(..., overwrite=True)`) replaces an existing model package only
after checking its manifest, regular payload files, and tensor sizes. It refuses
unrelated directories, ordinary files, and symlink destinations. This check does
not reread the discarded payload contents or verify their checksums.

## Python and macOS interchange

### Optional model source attachments

Export source companions after training to supply original text, optional cached
features, and optional full scores to the matching model in Reexpress two. The two
original source files are independently optional. No retraining is required to
export companions for a compatible existing model.

```bash
# Small companions for browsing and nearest-exemplar text.
sdm dataset export-sources --model model.sdmkitmodel \
  --training train.jsonl --role original-training \
  --text_only --output training-text.sdmdataset
sdm dataset export-sources --model model.sdmkitmodel \
  --calibration calibration.jsonl --role original-calibration \
  --text_only --output calibration-text.sdmdataset

# Add exact cached inputs and fresh scores, ready for analysis or continuation.
sdm dataset export-sources --model model.sdmkitmodel \
  --training train.jsonl --role original-training \
  --with_scores --matching_device mps --output training-full.sdmdataset
sdm dataset export-sources --model model.sdmkitmodel \
  --calibration calibration.jsonl --role original-calibration \
  --with_scores --matching_device mps --output calibration-full.sdmdataset

# Export a complete winning split in saved order, even after J shuffles.
sdm dataset export-sources --model model.sdmkitmodel \
  --training train.jsonl --calibration calibration.jsonl \
  --role selected-training --output best-training.sdmdataset
```

Use the same `--composition` as training when selecting only embeddings or only
attributes. The default composes embedding followed by attributes. Export removes
unselected feature columns and incoming SDM score fields; `--with_scores` computes
new complete scores. Each winning training row excludes its explicitly validated
support index, including rows from the original calibration file after shuffling.
Winning calibration rows receive no such exclusion. `--text_only` and
`--with_scores` cannot be combined. To retain full cached features without running
scoring, omit both flags.

In Reexpress two, import the exact model package, then use **Attach model sources**
for its companions. You can attach either original file first and add the other
later. Original file membership and winning iteration membership are preserved
separately; a single original file can fill portions of both winning splits after
shuffling. Text-only attachments support display; continuing training also needs
complete compatible cached features in both selected splits. Pre-scored evaluation
bundles continue to use ordinary dataset import.

```python
from reexpress_sdm import export_source_dataset

export_source_dataset(
    "model.sdmkitmodel", "training.sdmdataset",
    training="train.jsonl", role="original-training", text_only=True,
)
```

The exporter binds companions to exact model file hashes and validates saved
IDs, labels, ordered indexes, and available input-feature digests. New models save
Float32 feature digests without embedding original text or input vectors. Models
without saved feature digests use projected numerical compatibility checks when features are supplied;
those checks cannot prove identical raw inputs. Models trained directly with
`TorchTrainer.fit` support `selected-training` and `selected-calibration`; original
roles additionally require `bestIterationSplits` from `train_iterations` / the CLI.
See the [source attachment contract](docs/model-source-attachments-v1.md) for
the complete companion format and attachment rules.

### Upload Python-scored documents to Reexpress two

`sdm score --detail full` (the default) retains the source document and features
alongside the complete score, model ID, representation fingerprint, and matching
semantics. Import the same `.sdmkitmodel` first, then import the resulting JSONL
through the Data tab. The app validates and saves documents and scores together.
It can display the scores immediately and rescore the retained features locally.
`--detail compact` is a report format and is not an uploadable scored document.

Scores include up to **25 exact nearest exemplars** by default. Set
`sdm score --nearest_exemplars N` or pass
`nearest_exemplars=N` to `SDMModel.score`, `SDMController.score`,
`score_document`, or `score_dataset_rows`. Zero omits the expanded list; the
existing `nearestSupportIndex` and `nearestSupportID` fields remain present at
every setting. Fewer matches are returned when the support is smaller, and an
explicitly excluded training identity never appears in the list.

`nearestSupportMatches` is a ranked array containing each exemplar's support
index, ID, true label, predicted label, squared L2 distance, and document text
when stored in the model. Prompt and metadata fields are not exemplar display
text. The array appears in full and compact JSONL and full
`.sdmdataset` output. The Python value is `score.nearest_support_matches`, a tuple
of `SupportMatch` objects. The count controls returned diagnostics only; it
does not change `maxNeighbors`, `q`, training, or calibration. Existing matching
results are reused when sufficient; requesting more neighbors than the scoring
search retained requires an additional exact search. Larger lists increase
output size, especially when exemplars contain long documents.

Reexpress two preserves imported lists with their original scores. Native app
rescoring clears the imported list; the app continues to compute its own 25
nearest matches on demand. Source-only exports omit this score field.

For one document:

```python
from reexpress_sdm import SDMModel, load_artifact, score_document, write_scored_jsonl

model = SDMModel(load_artifact("continued.sdmkitmodel"))
scored = score_document(model, {
    "id": "document-001", "label": -1,
    "document": "Example text", "embedding": embedding_vector,
})
write_scored_jsonl([scored], "scored-document.jsonl")
```

By default no support row is excluded, even if an external document ID matches
a support ID. To score a known training exemplar, explicitly supply
`identity_support_index`; the exporter verifies its label and projected features
and records the excluded support ID. Reexpress two preserves this validated identity
for subsequent local rescoring. Model/fingerprint mismatches or malformed scores
are rejected. Existing document IDs follow the Data tab's usual Skip/Overwrite
choice. Backend rounding can affect very close neighbor ties or region boundaries.

### Read documents scored by Reexpress two in Python

In Reexpress two, choose **Data → Export scored dataset · Python / app** (or the
**Scored dataset** option under Analysis export). Both JSONL and `.sdmdataset`
retain the full source features and producer's original diagnostics:

```python
from reexpress_sdm import iter_dataset_rows

for row in iter_dataset_rows("app-scored.sdmdataset"):  # JSONL also works
    print(row["id"], row["prediction"], row["lowerRegionAlpha"])
```

Reading these rows does not recompute their scores. Keep the matching model
package with the export. For intentional rescoring of a row with saved support
exclusion evidence, pass its `excludedSupportIndex` explicitly as
`identity_support_index` to `score_document`; the existing identity checks still
apply. Batch rescoring takes the aligned `identity_support_indices` list.
Reexpress two scores that do not retain every diagnostic need an explicit
Rescore before full export. Compact score reports and source-only exports serve
their existing separate purposes.

## Python API

Iteration orchestration is separate from the single-fit trainer APIs.
`train_iterations` defaults to pooled split shuffling for both fresh and
continued training. Pass `shuffle_training_and_calibration=False` to keep input
memberships. Direct `TorchTrainer.fit` calls always use their supplied splits;
the orchestration layer owns split shuffling.

```python
from reexpress_sdm import TrainingConfig, train_iterations

result = train_iterations(
    TrainingConfig(number_of_classes=2, epochs=50),
    train_vectors, train_labels, calibration_vectors, calibration_labels,
    train_ids=train_ids, calibration_ids=calibration_ids,
    representation_fingerprint="my_embedding_v1",
    number_of_random_shuffles=5,
    backend_options={"device": "mps"},  # PyTorch is the default training backend.
)
print(result.best_iteration, result.best_epoch)
print(result.training_pool_indices, result.calibration_pool_indices)
```

```python
from reexpress_sdm import EstimatorKind, SDMController, SelectionPolicy

controller = SDMController.load(
    "model.sdmkitmodel",
    policy=SelectionPolicy(0.95, EstimatorKind.CENTROID),
)
scores = controller.score([[0.1, 0.2, 0.3]], ids=["example"])
decisions = controller.decide(scores)
```

For direct batched matching, this configuration uses an 8 MiB Float32 distance block per tile pair. Selection buffers, weights, support, and source features need additional memory:

```python
from reexpress_sdm import ExactL2Index

index = ExactL2Index(
    support_vectors,
    query_tile_size=128,
    support_tile_size=16_384,
)
distances, support_indices = index.search_many(query_vectors, k=25)
```

Rejection from every calibrated region is represented by a region alpha of zero. The two
`isInMostConservativeRegion*` fields are derived by comparing their assigned
alpha with the artifact's highest recorded alpha.

## Tests and distribution builds

Run the test suite from the repository root after installing the package:

```bash
python -m unittest discover -s tests -v
```

The required test fixtures are included under `tests/fixtures/`; no parent
repository, private dataset, or Swift build is required. Tests skip unavailable
device backends. The optional Swift artifact check uses a two-dimensional input;
set `SDMKIT_SWIFT_ARTIFACT` to the absolute path of a compatible Swift-exported
test model to enable it. The optional training-export check uses
`SDMKIT_SWIFT_TRAINING_ARTIFACT` and its matching `.expected.json` companion.

Build a wheel and source distribution with:

```bash
python -m pip install build
python -m build
```

The build uses setuptools >=77, installed in the build tool's isolated
environment. Outputs are written to `dist/`; generated wheels, source archives,
and build directories do not need to be committed to the repository.

## Converting a research checkpoint

`scripts/convert_research_model.py` converts a research-code model directory
(the `reexpress_mcp_server` v2.5.0 layout: `compression_index.pt`,
`support.npy`, `support_ids.json`, `meta.json`, calibration tensors) into a
schema-v1 `.sdmkitmodel`. This optional utility additionally requires **FAISS**
(`import faiss`) to read the serialized research index. FAISS is not a package
dependency and is not needed for ordinary training, scoring, or dataset import.
Run from the repository root in an environment with the package and FAISS installed:

```bash
python scripts/convert_research_model.py MODEL_DIR OUT.sdmkitmodel \
    --documents_db MODEL_DIR/reexpress_mcp_server_db/reexpress_mcp_server_support_documents.db \
    --class_name "NOT Verified" --class_name "Verified"
```

Class names and representation provenance are not stored by the research code,
so pass them (or accept the placeholders). Calibration rows flagged OOD by the
research code are omitted unless `--include_ood_calibration_rows` is given;
recalibrating from the exported rows reproduces the checkpoint's regions.
Datasets scored against the converted model must declare the same
representation fingerprint (default `embedding_v1`).
