# Training capacity calculator

[`estimate_capacity.py`](estimate_capacity.py) estimates the memory needed for a
proposed training run, or searches for an approximate row-count range for a given
machine. It supports **Reexpress two**, **Python with JSONL input**, and **Python
with `.sdmdataset` input**.

The script uses only Python's standard library. Download or copy this one file
and run it with Python 3.10 or later; installing `reexpress_sdm`, NumPy, PyTorch,
or MLX is unnecessary. It does not read datasets, allocate feature matrices,
detect your hardware, or run training. You can plan for another machine by
entering that machine's RAM and GPU memory.

**This is a planning tool, not a guarantee of fit or a benchmark.** It estimates
training memory, including allowances for model metadata and export. It does
not estimate data-import peaks, inference/evaluation memory, full project disk
requirements, or training duration. Start with a smaller representative run
before committing to a large one. Millions of rows may fit in memory while
exact matching remains impractically slow.

## Check a proposed run

From the directory containing the downloaded script:

```bash
python3 estimate_capacity.py estimate \
  --profile app --device mps --ram_gib 128 \
  --training_rows 1000000 --calibration_rows 1000000 \
  --embedding_dimension 16130 --exemplar_dimension 1000
```

For the app, `mps` denotes its Apple GPU/MLX training backend. This example passes
the format-capacity checks but exceeds the memory planning budget: the input
Float32 values alone occupy about **120.18 GiB**, before projections, support,
metadata, and temporary arrays. The default budget leaves 25% of installed RAM
for macOS and other activity.

Counts describe the **original** training and calibration files. By default,
the calculator models pooling and reshuffling into equal halves, with an odd
extra row going to calibration. Add `--do_not_shuffle_data` to preserve the
original split sizes. Evaluation datasets are not included in these counts.

Use `--attribute_dimension` when attributes are selected as well. The total
input dimension is `embedding_dimension + attribute_dimension`; use
`--embedding_dimension 0` for attributes only. Do not include an unselected
vector column in this combined dimension.

## Estimate a row-count range

```bash
python3 estimate_capacity.py max_rows \
  --profile app --device mps --ram_gib 128 \
  --embedding_dimension 128 --exemplar_dimension 32
```

This reports **total training plus calibration rows**, not rows per split:

- The **cautious** end uses the higher memory allowances.
- The **broader** end uses the lower memory allowances.
- The separate **capacity-check ceiling** ignores RAM and running time. It
  follows from the manifest budget and tensor limits, not a blanket row limit.

Neither end is a measured hardware maximum. Even the cautious estimate can be
too high for unusual data or additional live models. A result of zero means
that no configuration satisfies that set of assumptions. Use
`--training_fraction 0.8 --do_not_shuffle_data` to model an 80:20 split; without
the second option the original counts are still reshuffled into halves.

## Python and CUDA

```bash
python3 estimate_capacity.py estimate \
  --profile python_sdmdataset --device cuda \
  --ram_gib 256 --gpu_memory_gib 80 \
  --training_rows 500000 --calibration_rows 500000 \
  --embedding_dimension 16128 --exemplar_dimension 128
```

CUDA checks **system RAM and one selected GPU's memory separately**. The current
training implementation does not pool memory across multiple GPUs. Supply the
usable capacity of one device, not the sum for a two-GPU server. The full
projected support index resides on that GPU; reducing the matching tile size
does not remove it.

Use `--profile python_jsonl` for JSONL/NDJSON input. Python retains decoded
feature lists in addition to dense arrays, so JSONL needs substantially more
RAM. A `.sdmdataset` avoids those Python feature-number objects, but training
still creates arrays. Memory-mapped input is not treated as free memory.
The app imports either format into its project and materializes training rows;
its training estimate therefore uses the same `app` profile for both formats.

**JSONL caveat:** the estimate assumes there are no additional unselected
feature arrays in the original rows. Python retains those arrays even if they
are excluded from the input representation. Remove unused vector columns or
budget their decoded memory separately. Ordinary document text and non-vector
metadata are covered by `--average_text_bytes`.

## Adjust assumptions and inspect the output

```bash
python3 estimate_capacity.py --help

# JSON includes every setting and the memory components, all in bytes.
python3 estimate_capacity.py estimate \
  --ram_gib 128 --training_rows 452833 --calibration_rows 34599 \
  --embedding_dimension 1000 --exemplar_dimension 1000 \
  --average_text_bytes 2048 --memory_fraction 0.70 --json > capacity.json
```

The defaults are **calculator assumptions**, not automatically read from an
installed package or project. Set them to match the intended run:

| Setting | Default / meaning |
| --- | --- |
| Available memory fraction | 0.75 of each applicable memory pool; adjust for other processes |
| Text and non-vector metadata | 1,024 bytes per row; include `document`, `prompt`, and other source metadata |
| Source IDs | 36 ASCII-escaped JSON content bytes per row, excluding quotes; suitable for UUIDs |
| Classes / epochs / iterations | 2 / 300 / 2; epochs and iterations contribute to history metadata |
| Alpha resolution | 0.01; finer ladders reserve more manifest space |
| Additional model metadata | 0 MiB beyond the generated-metadata formula |
| Training minibatch / maximum neighbors | 64 / 2,048 |
| Matching query batch / support tile | 256 / 16,384; app CPU uses 4,096 support rows |

Matching batch/tile overrides apply only to the Python profiles. Reexpress two
uses the fixed values above. Its training minibatch and maximum-neighbor
settings remain adjustable. Long or Unicode IDs may require a larger escaped
ID size; the script does not inspect your files to calculate this mean.

Each memory pool reports one of three statuses: the whole planning range fits
the budget, the budget falls inside that range, or even the lower estimate
exceeds it. The program exits successfully after reporting a capacity warning;
invalid arguments are usage errors. For automation, inspect the JSON fields
rather than interpreting exit status as permission to train.

## What the formulas represent

Let `N` be total rows, `T` the effective training/support rows, `D` the combined
input dimension, `H` the exemplar dimension, and `C` the class count. Exact
Float32 payload sizes are:

| Payload | Bytes |
| --- | --- |
| Input vectors (`I`) | `4 × N × D` |
| All projections (`P`) | `4 × N × H` |
| Projected support (`S`) | `4 × T × H` |
| Adaptor weights (`W`) | `4 × (D × H + H + C × H + C)` |

The main implementation-dependent allowances are:

| Component | App | Python |
| --- | --- | --- |
| Input matrices | `I`; `2I` when embedding and attributes are concatenated | `4–6I` shuffled; `2–4I` unshuffled; add `I` for composed `.sdmdataset` columns |
| Decoded JSONL feature objects | Covered by project import; not retained as Python objects | Additional `8–10I` for JSONL |
| Projections and host support copies | `P–2P` plus `4S–8S` | `P–2P` plus `2S–4S` |
| Parameters, optimizer, snapshots | `6W–12W` | `6W–10W` on the selected device; CUDA also reserves `2W–4W` in host RAM |
| Metadata and serialization | `4–8×` conservative manifest budget | Same |

The script also adds per-record/text allowances, framework overhead, bounded
matching and minibatch workspace, and Python's device-resident support copy.
CPU/MPS allocations contribute to one system/unified pool; CUDA device
allocations have a separate budget. Training reduces neighbor results in query
batches, so the estimate does **not** assume a persistent `N × maximum_neighbors`
table. Whole-dataset inference/evaluation can have different allocations.

These allowances come from inspecting the implementations, not fitting a
hardware-specific benchmark. In one CPython check, float objects and list
pointers occupied about 8.1 times their Float32 payload. A synthetic 2.47 GiB
manifest load/export check used roughly five times its encoded size at peak.
Those observations motivate the ranges; they do not establish bounds. Phase
peaks may not coincide, aliases may share memory, and allocator caching or
unusually large records may increase actual usage. An already loaded model or
additional project data can also require more memory than modeled here.

The work indicator is `N × T` candidate comparisons for **one full matching
pass**. At fixed split proportions, ten times as many rows implies about one
hundred times as many comparisons. Exemplar dimension also affects the work per
comparison. Multiple passes occur during training; this indicator is not a
runtime forecast.

## Policy snapshot and maintenance

This version models the **2026-10-03 shared 4 GiB manifest implementation** and
the existing two-billion-element limits for support and weights. Earlier
released readers may still have a 64 MiB manifest ceiling. Check the installed
software's release documentation before relying on the larger limit. This
calculator does not change any limit in an installed app or package.

The generated-manifest formula mirrors
[`capacity.py`](../../src/reexpress_sdm/capacity.py), substituting the estimated
mean ID length for the production scan of actual IDs. Export still validates
actual encoded bytes. There is no new row-count policy in this tool.

Policy constants and memory coefficients are centralized near the top of the
script. If the training implementation or format limits change, update those
assumptions and the snapshot date together. The utility is a repository/download
tool, not an `sdm` subcommand or a script installed by the package wheel. It can
be copied to a separate documentation repository with its Apache-2.0 license
and copyright notice.

Run its standalone tests from the repository root:

```bash
python3 -B -I -S -m unittest discover -s docs/tools/tests -v
```

The tests require only the standard library and include running a copy of the
single script with site packages and local-package imports disabled.
