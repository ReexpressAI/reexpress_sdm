# Copyright Reexpress AI, Inc. All rights reserved.
"""Standalone training-capacity planner for Reexpress two and reexpress_sdm.

Python 3.10+, standard library only. No datasets are read or allocated.
Run with --help, or see the adjacent README.md for formulas and limitations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math


# Policy snapshot: 2026-10-03, the shared 4 GiB manifest implementation.
# Earlier released readers may still have a 64 MiB ceiling. These constants
# must track SDMArtifactLimits / artifact.py and both training-capacity helpers.
POLICY_SNAPSHOT = "2026-10-03: 4 GiB manifest policy"
GIB = 1024**3
MIB = 1024**2
MAX_MANIFEST_BYTES = 4 * GIB
MAX_TENSOR_ELEMENTS = 2_000_000_000
MAX_DIMENSION = 2**31 - 1
MIN_ALPHA_RESOLUTION = 0.00005

# Planning allowances, NOT measured bounds or confidence intervals. Centralize
# them here so future measurements can refine the tool without changing SDM.
# Python float objects + list pointers measured about 8.1x Float32 payload on
# one CPython build. A synthetic 2.47 GiB manifest used ~5x its bytes at peak.
MANIFEST_MEMORY_FACTORS = (4, 8)
ROW_OBJECT_BYTES = (512, 2048)       # Excludes vectors, text and source IDs.
TEXT_MEMORY_FACTORS = (1, 4)        # Decoded text/metadata, copies and Unicode.
RUNTIME_BYTES = (GIB, 2 * GIB)      # Framework/allocator allowance, per pool.
PYTHON_JSON_FEATURE_FACTORS = (8, 10)
PROFILES = {
    "app": {"projection": (1, 2), "support": (4, 8), "weights": (6, 12),
            "workspace": (16, 64)},
    "python_jsonl": {"projection": (1, 2), "support": (2, 4), "weights": (6, 10),
                     "workspace": (40, 96)},
    "python_sdmdataset": {"projection": (1, 2), "support": (2, 4), "weights": (6, 10),
                          "workspace": (40, 96)},
}


@dataclass(frozen=True)
class Settings:
    ram_gib: float
    embedding_dimension: int
    attribute_dimension: int = 0
    exemplar_dimension: int = 1000
    profile: str = "app"
    device: str = "mps"
    gpu_memory_gib: float | None = None
    memory_fraction: float = 0.75
    number_of_classes: int = 2
    epochs: int = 300
    iterations: int = 2
    alpha_resolution: float = 0.01
    do_not_shuffle_data: bool = False
    average_id_bytes: int = 36
    average_text_bytes: int = 1024
    additional_metadata_mib: float = 0.0
    batch_size: int = 64
    matching_query_batch_size: int = 256
    matching_support_tile_size: int = 16384
    max_neighbors: int = 2048

    @property
    def dimension(self) -> int:
        return self.embedding_dimension + self.attribute_dimension


def candidate_alpha_count(resolution: float) -> int:
    # Match the production ladder's rounding, including near-0.5 edge cases.
    count = 0
    for k in range(1, 10_001):
        if round(1.0 - k * resolution, 10) <= 0.5:
            break
        count += 1
    return count


def manifest_budget(s: Settings, total: int, calibration: int) -> int:
    """Same conservative formula as production, with caller-estimated ID bytes."""
    return (MIB + total * (112 + len(str(total - 1)))
            + total * (s.average_id_bytes + 2) + calibration * 80
            + candidate_alpha_count(s.alpha_resolution) * (512 + 40 * s.number_of_classes)
            + s.epochs * s.iterations * 2048 + s.iterations * 2048
            + math.ceil(s.additional_metadata_mib * MIB))


def evaluate(s: Settings, training: int, calibration: int) -> dict:
    total = training + calibration
    support = training if s.do_not_shuffle_data else total // 2
    selected_calibration = total - support
    d, h, c = s.dimension, s.exemplar_dimension, s.number_of_classes
    inputs = 4 * total * d
    projected = 4 * total * h
    support_bytes = 4 * support * h
    weights = 4 * (d * h + h + c * h + c)
    manifest = manifest_budget(s, total, selected_calibration)
    checks = {
        "support_tensor": support * h <= MAX_TENSOR_ELEMENTS,
        "weight_tensor": weights // 4 <= MAX_TENSOR_ELEMENTS,
        "estimated_manifest_budget": manifest <= MAX_MANIFEST_BYTES,
    }
    composed = s.embedding_dimension > 0 and s.attribute_dimension > 0
    profile = PROFILES[s.profile]
    query_batch = min(s.matching_query_batch_size, max(support, selected_calibration))
    tile = min(s.matching_support_tile_size, support)
    neighbors = min(s.max_neighbors, support)
    batch_vectors = 4 * max(min(s.batch_size, support), query_batch) * (d + h + c)
    host = [0, 0]
    accelerator = [0, 0]
    components: dict[str, list[int]] = {}

    def reserve(name: str, low: int, high: int) -> None:
        components[name] = [low, high]
        host[0] += low
        host[1] += high

    if s.profile == "app":
        reserve("input_vectors", inputs * (2 if composed else 1), inputs * (2 if composed else 1))
    else:
        copies = (2, 4) if s.do_not_shuffle_data else (4, 6)
        extra = int(composed and s.profile == "python_sdmdataset")
        reserve("input_vectors_and_training_copies", inputs * (copies[0] + extra), inputs * (copies[1] + extra))
        if s.profile == "python_jsonl":
            reserve("decoded_json_feature_lists", inputs * PYTHON_JSON_FEATURE_FACTORS[0],
                    inputs * PYTHON_JSON_FEATURE_FACTORS[1])
    reserve("projections_and_support_copies",
            projected * profile["projection"][0] + support_bytes * profile["support"][0],
            projected * profile["projection"][1] + support_bytes * profile["support"][1])
    reserve("manifest_and_provenance", manifest * MANIFEST_MEMORY_FACTORS[0], manifest * MANIFEST_MEMORY_FACTORS[1])
    reserve("records_text_and_ids",
            total * (ROW_OBJECT_BYTES[0] + s.average_text_bytes * TEXT_MEMORY_FACTORS[0] + s.average_id_bytes),
            total * (ROW_OBJECT_BYTES[1] + s.average_text_bytes * TEXT_MEMORY_FACTORS[1] + 4 * s.average_id_bytes))
    reserve("host_runtime", *RUNTIME_BYTES)
    if s.device == "cuda":
        # NumPy initialization and best/CE checkpoint weights remain in RAM.
        reserve("host_weight_snapshots", 2 * weights, 4 * weights)

    for end in (0, 1):
        accelerator[end] = (weights * profile["weights"][end]
                            + query_batch * (tile + neighbors) * profile["workspace"][end]
                            + batch_vectors * (2 if end == 0 else 4))
        if s.profile != "app":
            # The entire centered support index resides on the selected device.
            accelerator[end] += support_bytes + 4 * support + RUNTIME_BYTES[end]
    if s.device != "cuda":
        reserve("parameters_and_matching_workspace", *accelerator)

    budget = math.floor(s.ram_gib * GIB * s.memory_fraction)
    gpu_budget = math.floor(s.gpu_memory_gib * GIB * s.memory_fraction) if s.device == "cuda" else None

    def pool(values: list[int], limit: int) -> dict:
        status = ("within_planning_budget" if values[1] <= limit else
                  "above_planning_budget" if values[0] > limit else "uncertain_within_range")
        return {"lower_bytes": values[0], "upper_bytes": values[1], "budget_bytes": limit, "status": status}

    return {
        "original_rows": {"training": training, "calibration": calibration, "total": total},
        "effective_rows": {"training_support": support, "calibration": selected_calibration},
        "payload_bytes": {"input_float32": inputs, "all_projected_float32": projected,
                          "support_float32": support_bytes, "adaptor_weights_float32": weights},
        "estimated_manifest_bytes": manifest,
        "capacity_checks": checks,
        "memory": {"system_or_unified": pool(host, budget),
                   "cuda_device": pool(accelerator, gpu_budget) if gpu_budget is not None else None,
                   "system_components_bytes": components},
        "candidate_comparisons_per_full_matching_pass": total * support,
    }


def maximum_rows(s: Settings, fraction: float) -> dict:
    """Monotone binary search; upper bound comes from per-row manifest bytes."""
    def split(n: int) -> tuple[int, int]:
        training = math.floor(n * fraction)
        return training, n - training

    upper = MAX_MANIFEST_BYTES // 112  # Derived from manifest formula, not a row policy.
    original_minima = (2, 1) if s.profile == "app" and not s.do_not_shuffle_data else (s.number_of_classes,) * 2
    def enough_original_rows(n: int) -> bool:
        return all(count >= required for count, required in zip(split(n), original_minima))

    # Reject impossible splits before dividing by an arbitrarily tiny fraction.
    if not enough_original_rows(upper):
        return {"original_training_fraction": fraction, "cautious_total_rows": 0,
                "broader_total_rows": 0, "format_checks_total_rows": 0,
                "cautious_configuration": None, "broader_configuration": None}
    # Enough original/effective rows for every class; label balance still matters.
    minimum = max(2 * s.number_of_classes,
                  math.ceil(original_minima[0] / fraction),
                  math.ceil(original_minima[1] / (1 - fraction)))
    while not enough_original_rows(minimum):
        minimum += 1

    def search(end: str | None) -> int:
        def accepts(n: int) -> bool:
            report = evaluate(s, *split(n))
            if not all(report["capacity_checks"].values()):
                return False
            if end is None:
                return True
            return all(p[end] <= p["budget_bytes"] for p in
                       (report["memory"]["system_or_unified"], report["memory"]["cuda_device"]) if p is not None)
        if minimum > upper or not accepts(minimum):
            return 0
        lo, hi = minimum, upper
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if accepts(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    cautious, broader, metadata_limit = search("upper_bytes"), search("lower_bytes"), search(None)
    return {
        "original_training_fraction": fraction,
        "cautious_total_rows": cautious,
        "broader_total_rows": broader,
        "format_checks_total_rows": metadata_limit,
        "cautious_configuration": evaluate(s, *split(cautious)) if cautious else None,
        "broader_configuration": evaluate(s, *split(broader)) if broader else None,
    }


def finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    return value


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Estimate TRAINING capacity without loading data or installing SDM/NumPy/PyTorch. "
                                "Memory ranges are planning allowances, not guarantees or runtime estimates.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    p.add_argument("mode", choices=["estimate", "max_rows"], help="Check a proposed run or estimate a row-count range")
    p.add_argument("--ram_gib", type=finite_float, required=True, help="Installed system/unified RAM in GiB")
    p.add_argument("--embedding_dimension", type=int, required=True, help="Selected embedding dimensions; use 0 for attributes only")
    p.add_argument("--attribute_dimension", type=int, default=0, help="Selected attributes dimensions, concatenated after embedding")
    p.add_argument("--exemplar_dimension", type=int, default=1000, help="Projected exemplar dimensions")
    p.add_argument("--profile", choices=list(PROFILES), default="app", help="App training or Python input format")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps", help="Target compute device; app/mps means MLX")
    p.add_argument("--gpu_memory_gib", type=finite_float, help="CUDA memory of ONE selected GPU; do not sum GPUs")
    p.add_argument("--memory_fraction", type=finite_float, default=0.75, help="Fraction of each memory pool available for this run")
    p.add_argument("--training_rows", type=int, help="Original training rows, required for estimate")
    p.add_argument("--calibration_rows", type=int, help="Original calibration rows, required for estimate")
    p.add_argument("--training_fraction", type=finite_float, help="Original training fraction for max_rows; default 0.5")
    p.add_argument("--do_not_shuffle_data", action="store_true", help="Keep original split sizes instead of pooling/splitting 50:50")
    p.add_argument("--number_of_classes", type=int, default=2, help="App supports 2–5 classes")
    p.add_argument("--epochs", type=int, default=300, help="Requested epochs per iteration, for metadata budgeting")
    p.add_argument("--iterations", type=int, default=2, help="Requested training iterations, for metadata budgeting")
    p.add_argument("--alpha_resolution", type=finite_float, default=0.01, help="Calibration alpha spacing, at least 0.00005 and below 0.5")
    p.add_argument("--average_id_bytes", type=int, default=36, help="Mean ASCII-escaped JSON ID content bytes, excluding quotes; UUID=36")
    p.add_argument("--average_text_bytes", type=int, default=1024, help="Mean document/prompt/non-vector metadata bytes per row; excludes IDs")
    p.add_argument("--additional_metadata_mib", type=finite_float, default=0, help="Extra model metadata beyond generated source evidence")
    p.add_argument("--batch_size", type=int, default=64, help="Training minibatch rows")
    p.add_argument("--matching_query_batch_size", type=int, default=256, help="Matching query rows per batch; adjustable for Python only")
    p.add_argument("--matching_support_tile_size", type=int, help="Support rows per tile; adjustable for Python only; default 4096 for app/cpu, otherwise 16384")
    p.add_argument("--max_neighbors", type=int, default=2048, help="Maximum neighbors per query; training requires at least 2")
    p.add_argument("--json", action="store_true", help="Print machine-readable results and assumptions")
    return p


def validate(p: argparse.ArgumentParser, a: argparse.Namespace) -> Settings:
    for name, multiplier in (("ram_gib", GIB), ("gpu_memory_gib", GIB), ("additional_metadata_mib", MIB)):
        value = getattr(a, name)
        if value is not None and not math.isfinite(value * multiplier):
            p.error(f"{name} is too large to express in bytes")
    if a.ram_gib <= 0 or not 0 < a.memory_fraction <= 1:
        p.error("RAM must be positive and memory_fraction must be in (0, 1]")
    if a.embedding_dimension < 0 or a.attribute_dimension < 0 or a.embedding_dimension + a.attribute_dimension <= 0:
        p.error("feature dimensions must be nonnegative with a positive combined dimension")
    if a.embedding_dimension + a.attribute_dimension > MAX_DIMENSION:
        p.error("combined input dimensions exceed the shared Int32 limit")
    for name in ("exemplar_dimension", "epochs", "iterations", "batch_size", "matching_query_batch_size"):
        if not 1 <= getattr(a, name) <= MAX_DIMENSION:
            p.error(f"{name} must be between 1 and {MAX_DIMENSION}")
    if not 2 <= a.number_of_classes <= MAX_DIMENSION or (a.profile == "app" and a.number_of_classes > 5):
        p.error("number_of_classes must be at least 2 (and at most 5 for app)")
    if a.max_neighbors < 2 or a.average_id_bytes < 1 or a.average_text_bytes < 0 or a.additional_metadata_mib < 0:
        p.error("max_neighbors >= 2, average_id_bytes >= 1, and text/extra metadata sizes >= 0 are required")
    if not MIN_ALPHA_RESOLUTION <= a.alpha_resolution < 0.5:
        p.error("alpha_resolution must be at least 0.00005 and below 0.5")
    if a.device == "cuda":
        if a.profile == "app" or a.gpu_memory_gib is None or a.gpu_memory_gib <= 0:
            p.error("CUDA requires a Python profile and positive --gpu_memory_gib for one GPU")
    elif a.gpu_memory_gib is not None:
        p.error("--gpu_memory_gib applies only to CUDA; MPS uses unified RAM")
    if a.matching_support_tile_size is None:
        a.matching_support_tile_size = 4096 if a.profile == "app" and a.device == "cpu" else 16384
    if not 1 <= a.matching_support_tile_size <= MAX_DIMENSION:
        p.error("matching_support_tile_size must be positive and within Int32")
    if a.profile == "app" and (a.matching_query_batch_size != 256 or
                               a.matching_support_tile_size != (4096 if a.device == "cpu" else 16384)):
        p.error("app matching batches are fixed: 256 queries, 4096 support rows on CPU or 16384 on MLX")
    if a.mode == "estimate":
        if a.training_fraction is not None:
            p.error("--training_fraction is for max_rows; estimate uses the supplied row counts")
        minimum_train, minimum_calibration = ((2, 1) if a.profile == "app" and not a.do_not_shuffle_data
                                               else (a.number_of_classes, a.number_of_classes))
        if (a.training_rows is None or a.calibration_rows is None or
                a.training_rows < minimum_train or a.calibration_rows < minimum_calibration or
                a.training_rows + a.calibration_rows < 2 * a.number_of_classes):
            p.error("estimate requires both original split counts, sufficient for every class in each effective split")
    else:
        if a.training_rows is not None or a.calibration_rows is not None:
            p.error("max_rows takes --training_fraction, not fixed row counts")
        if a.training_fraction is None:
            a.training_fraction = 0.5
        if not 0 < a.training_fraction < 1:
            p.error("training_fraction must be between 0 and 1, exclusive")
    return Settings(**{name: getattr(a, name) for name in Settings.__dataclass_fields__})


def gib(value: int) -> str:
    return f"{value / GIB:,.2f} GiB"


def print_estimate(r: dict) -> None:
    original, effective = r["original_rows"], r["effective_rows"]
    print(f"Rows: {original['total']:,} total; supplied {original['training']:,} training + {original['calibration']:,} calibration")
    print(f"Effective splits: {effective['training_support']:,} support + {effective['calibration']:,} calibration")
    for name, size in r["payload_bytes"].items():
        print(f"  {name.replace('_', ' ')}: {gib(size)} ({size:,} bytes)")
    print(f"Estimated manifest budget: {gib(r['estimated_manifest_bytes'])} / 4 GiB")
    print("Capacity checks: " + ", ".join(f"{key}={'within limit' if ok else 'EXCEEDS LIMIT'}" for key, ok in r["capacity_checks"].items()))
    for label, key in (("System/unified RAM", "system_or_unified"), ("One CUDA GPU", "cuda_device")):
        memory = r["memory"][key]
        if memory is not None:
            print(f"{label}: {gib(memory['lower_bytes'])}–{gib(memory['upper_bytes'])} planning range; "
                  f"budget {gib(memory['budget_bytes'])}; {memory['status'].replace('_', ' ')}")
    print(f"Candidate comparisons per full matching pass: {r['candidate_comparisons_per_full_matching_pass']:,}")


def main(argv: list[str] | None = None) -> int:
    p = parser()
    a = p.parse_args(argv)
    s = validate(p, a)
    result = evaluate(s, a.training_rows, a.calibration_rows) if a.mode == "estimate" else maximum_rows(s, a.training_fraction)
    notes = [
        "Training planning only: ranges are heuristic allowances, not measured peaks, confidence intervals, or guarantees of fit.",
        "Defaults include 1 KiB of text/non-vector metadata per row and 25% memory headroom. Adjust for your data and other applications.",
        "Manifest budgeting assumes the supplied mean escaped ID length and extra metadata. The actual software validates real encoded bytes.",
        "Assumes valid data, class coverage, and distinct IDs. Existing models/checkpoints already held by an application may need additional RAM.",
        "Python JSONL estimates assume no additional unselected feature arrays: those remain in decoded rows and would require extra RAM.",
        "This does not estimate inference/evaluate, data-import peak memory, whole-project disk size, or practical running time.",
        "Exact matching scales with queries times support; 10x both counts means about 100x candidate comparisons at fixed exemplar dimension.",
        "Uses the 4 GiB manifest policy. Earlier released readers may have smaller limits; see the policy snapshot before planning a run.",
    ]
    report = {"policy_snapshot": POLICY_SNAPSHOT, "mode": a.mode, "settings": asdict(s), "result": result, "notes": notes}
    if a.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(f"Reexpress training capacity planner — {POLICY_SNAPSHOT}")
        print(f"Profile: {s.profile}; device: {s.device}; memory budget: {s.memory_fraction:.0%} of installed RAM")
        print(f"Features: {s.embedding_dimension} embedding + {s.attribute_dimension} attributes; exemplars: {s.exemplar_dimension}")
        print(f"Assumptions: {s.number_of_classes} classes; {s.epochs} epochs x {s.iterations} iterations; alpha step {s.alpha_resolution}; "
              f"IDs {s.average_id_bytes} escaped bytes; text/metadata {s.average_text_bytes} bytes/row")
        print(f"Split shuffling: {'off' if s.do_not_shuffle_data else 'on (50:50)'}; batch {s.batch_size}; "
              f"matching {s.matching_query_batch_size} x {s.matching_support_tile_size}; neighbors {s.max_neighbors}\n")
        if a.mode == "estimate":
            print_estimate(result)
        else:
            print(f"Planning capacity: {result['cautious_total_rows']:,}–{result['broader_total_rows']:,} TOTAL training + calibration rows")
            print("The cautious end uses higher memory allowances; the broader end uses lower allowances. Neither is a guarantee.")
            print(f"Capacity-check ceiling alone: {result['format_checks_total_rows']:,} total rows (ignores RAM and running time)")
            for label, key in (("Cautious configuration", "cautious_configuration"), ("Broader configuration", "broader_configuration")):
                print(f"\n{label}:")
                if result[key] is None:
                    print("No configuration meets this planning budget, even at the minimum row count.")
                else:
                    print_estimate(result[key])
        print("\nNotes:")
        for note in notes:
            print(f"- {note}")
        print("Use --json for the complete assumptions and component-by-component memory allowances.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
