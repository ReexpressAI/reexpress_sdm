# Copyright Reexpress AI, Inc. All rights reserved.
"""Command-line interface for portable SDM artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence, TextIO

from .artifact import load_artifact, write_artifact
from .bundle import DatasetBundle, iter_dataset_rows, validate_dataset_rows, write_dataset_bundle, write_dataset_jsonl
from .controller import SDMController, _artifact_information
from .dataset import Dataset
from .errors import SDMError
from .model import SDMModel
from .iterative_training import train_iterations
from .policy import SelectionPolicy
from .recalibration import recalibrate_artifact
from .score_io import score_dataset_rows, write_scored_jsonl
from .source_io import export_source_dataset, SOURCE_ROLES
from .training import TrainingConfig
from .types import EstimatorKind


class _ArgumentParser(argparse.ArgumentParser):
    """Show concise option defaults and require exact option spellings."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", argparse.ArgumentDefaultsHelpFormatter)
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)


def _write_json(value: Any, destination: str) -> None:
    encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    if destination == "-":
        sys.stdout.write(encoded)
    else:
        Path(destination).write_text(encoded, encoding="utf-8")


def _write_jsonl(values: Iterable[Mapping[str, Any]], destination: str) -> None:
    stream: TextIO
    should_close = destination != "-"
    stream = Path(destination).open("w", encoding="utf-8") if should_close else sys.stdout
    try:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
            stream.write("\n")
    finally:
        if should_close:
            stream.close()


def _controller(args: argparse.Namespace) -> SDMController:
    policy = SelectionPolicy(estimator=EstimatorKind(args.estimator))
    artifact = load_artifact(args.model, verify_checksums=not args.no_verify_checksums)
    return SDMController(
        SDMModel(artifact, backend=args.matching_backend, device=args.matching_device,
                 query_batch_size=args.matching_query_batch_size,
                 support_tile_size=args.matching_support_tile_size),
        policy=policy,
    )


def _dataset(args: argparse.Namespace, controller: SDMController, require_labels: bool) -> Dataset:
    dataset = Dataset.load(
        args.input,
        composition=args.composition,
        expected_dimension=controller.model.embedding_dimension,
        number_of_classes=controller.model.number_of_classes,
        require_labels=require_labels,
        representation_fingerprint=args.representation_fingerprint,
    )
    dataset.validate_class_names(controller.model.configuration["classNames"])
    return dataset


def _score_dataset(controller: SDMController, dataset: Dataset):
    return controller.score(
        dataset.vectors,
        ids=dataset.ids,
        representation_fingerprint=dataset.representation_fingerprint,
    )


def _run_artifact_validate(args: argparse.Namespace) -> int:
    artifact = load_artifact(args.model, verify_checksums=not args.no_verify_checksums)
    _write_json(
        {
            "valid": True,
            "schemaVersion": artifact.manifest["schemaVersion"],
            "modelID": artifact.model_id,
        },
        args.output,
    )
    return 0


def _run_artifact_inspect(args: argparse.Namespace) -> int:
    artifact = load_artifact(args.model, verify_checksums=not args.no_verify_checksums)
    _write_json(_artifact_information(artifact), args.output)
    return 0


def _class_names(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = tuple(name.strip() for name in value.split(","))
    if len(names) < 2 or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("--class_names requires at least two nonempty, unique comma-separated names")
    return names


def _dataset_information(path: str) -> dict[str, Any]:
    """Validate source rows without requiring cached features or a GPU runtime."""
    bundle = DatasetBundle.open(path) if Path(path).is_dir() else None
    rows = bundle.iter_rows() if bundle is not None else validate_dataset_rows(iter_dataset_rows(path))
    dimensions: dict[str, int] = {}
    feature_counts = {"embedding": 0, "attributes": 0}
    labels: dict[str, int] = {}
    fingerprint = bundle.fingerprint if bundle is not None else None
    count = 0
    for count, row in enumerate(rows, start=1):
        label = row.get("label", -1)
        labels[str(label)] = labels.get(str(label), 0) + 1
        fingerprint = row.get("representationFingerprint") or fingerprint
        for name in feature_counts:
            if name in row:
                dimensions[name] = len(row[name])
                feature_counts[name] += 1
    result = {
        "format": "sdmdataset" if bundle is not None else "jsonl",
        "rowCount": count,
        "classNames": list(bundle.class_names) if bundle is not None and bundle.class_names is not None else None,
        "representationFingerprint": fingerprint,
        "featureDimensions": dimensions,
        "rowsWithFeatures": feature_counts,
        "labelCounts": labels,
    }
    if bundle is not None:
        result["manifest"] = bundle.manifest
    return result


def _run_dataset_inspect(args: argparse.Namespace) -> int:
    _write_json(_dataset_information(args.input), args.output)
    return 0


def _run_dataset_validate(args: argparse.Namespace) -> int:
    information = _dataset_information(args.input)
    _write_json({"valid": True, **information}, args.output)
    return 0


def _run_dataset_convert(args: argparse.Namespace) -> int:
    source, destination = Path(args.input), Path(args.output)
    if args.output != "-" and source.resolve() == destination.resolve():
        raise ValueError("dataset conversion input and output must be different paths")
    bundle = DatasetBundle.open(source) if source.is_dir() else None
    names = _class_names(args.class_names)
    if names is None and bundle is not None:
        names = bundle.class_names
    representation = dict(bundle.representation or {}) if bundle is not None else {}
    fingerprint = args.representation_fingerprint
    if fingerprint is not None:
        if not fingerprint:
            raise ValueError("--representation_fingerprint must be nonempty")
        if representation.get("fingerprint") not in (None, fingerprint):
            raise ValueError("requested and source representation fingerprints differ")
        representation["fingerprint"] = fingerprint

    def rows():
        source_rows = bundle.iter_rows() if bundle is not None else iter_dataset_rows(source)
        for row in source_rows:
            if "label" not in row:
                row = {**row, "label": -1}
            if fingerprint is not None:
                if row.get("representationFingerprint") not in (None, fingerprint):
                    raise ValueError("requested and source representation fingerprints differ")
                row = {**row, "representationFingerprint": fingerprint}
            yield row

    if destination.suffix.lower() == ".sdmdataset":
        write_dataset_bundle(destination, rows(), class_names=names,
                             representation=representation or None,
                             metadata=bundle.metadata if bundle is not None else None,
                             overwrite=args.overwrite)
    elif args.output == "-":
        write_scored_jsonl(validate_dataset_rows(rows()), "-")
    elif destination.suffix.lower() in (".jsonl", ".ndjson"):
        write_dataset_jsonl(destination, rows(), overwrite=args.overwrite)
    else:
        raise ValueError("dataset output must end in .sdmdataset, .jsonl, or .ndjson, or be - for JSONL stdout")
    return 0


def _run_dataset_export_sources(args: argparse.Namespace) -> int:
    export_source_dataset(args.model, args.output, training=args.training, calibration=args.calibration,
                          role=args.role, composition=args.composition, text_only=args.text_only,
                          with_scores=args.with_scores, device=args.matching_device,
                          query_batch_size=args.matching_query_batch_size,
                          support_tile_size=args.matching_support_tile_size, overwrite=args.overwrite)
    return 0


def _run_score(args: argparse.Namespace) -> int:
    binary_output = Path(args.output).suffix.lower() == ".sdmdataset"
    if binary_output and args.detail != "full":
        raise ValueError(".sdmdataset output requires --detail full; compact scores are JSONL reports")
    if Path(args.output).resolve() == Path(args.input).resolve():
        raise ValueError("score input and output must be different paths")
    controller = _controller(args)
    dataset = _dataset(args, controller, require_labels=False)
    identities = None
    if args.identity_support_index_field:
        identities = [row.get(args.identity_support_index_field) for row in dataset.rows]
    if args.detail == "full":
        rows = score_dataset_rows(controller.model, dataset, composition=args.composition,
                                  batch_size=args.score_batch_size, identity_support_indices=identities,
                                  nearest_exemplars=args.nearest_exemplars)
    else:
        scores = controller.score(dataset.vectors, ids=dataset.ids,
                                  representation_fingerprint=dataset.representation_fingerprint,
                                  identity_support_indices=identities, nearest_exemplars=args.nearest_exemplars)
        rows = ({**score.to_dict(detail="compact"), "label": label} for score, label in zip(scores, dataset.labels))
    if binary_output:
        write_dataset_bundle(args.output, rows,
                             class_names=controller.model.configuration["classNames"],
                             representation=controller.model.artifact.manifest["representation"],
                             overwrite=args.overwrite)
    else:
        write_scored_jsonl(rows, args.output)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    controller = _controller(args)
    dataset = _dataset(args, controller, require_labels=True)
    scores = _score_dataset(controller, dataset)
    _write_json(
        {
            "evaluation": controller.evaluate(scores, dataset.labels),
            "distribution": controller.summarize_distribution(
                scores, histogram_bins=args.histogram_bins
            ),
        },
        args.output,
    )
    return 0


def _run_recalibrate(args: argparse.Namespace) -> int:
    initial_artifact = load_artifact(args.model)
    artifact = recalibrate_artifact(initial_artifact, alpha_resolution=args.alpha_resolution)
    write_artifact(args.output, artifact, overwrite=args.overwrite)
    _write_json(
        {
            "modelID": artifact.model_id,
            "artifact": str(Path(args.output)),
            "sourceModelID": initial_artifact.model_id,
            "alphaResolution": artifact.manifest["configuration"]["alphaResolution"],
            "calibrationSource": "savedCalibrationDiagnostics",
            "calibrationCount": len(artifact.calibration_rows),
            "regions": len(artifact.regions),
        },
        args.report_output,
    )
    return 0


def _run_train(args: argparse.Namespace) -> int:
    initial_artifact = load_artifact(args.initial_model) if args.initial_model else None
    class_names = (
        tuple(value.strip() for value in args.class_names.split(","))
        if args.class_names
        else None
    )
    if class_names is not None and len(class_names) != args.number_of_classes:
        raise ValueError("--class_names must provide one comma-separated name per class")
    if class_names is not None and (
        any(not value for value in class_names) or len(set(class_names)) != len(class_names)
    ):
        raise ValueError("--class_names must be nonempty and unique")
    training = Dataset.load(
        args.training,
        composition=args.composition,
        number_of_classes=args.number_of_classes,
        require_labels=True,
        representation_fingerprint=args.representation_fingerprint,
    )
    calibration = Dataset.load(
        args.calibration,
        composition=args.composition,
        expected_dimension=training.vectors.shape[1],
        number_of_classes=args.number_of_classes,
        require_labels=True,
        representation_fingerprint=args.representation_fingerprint,
    )
    fingerprints = {
        value for value in (training.representation_fingerprint, calibration.representation_fingerprint)
        if value is not None
    }
    if len(fingerprints) > 1:
        raise ValueError("training and calibration representation fingerprints differ")
    representation_fingerprint = next(iter(fingerprints), None)
    if class_names is None:
        class_names = training.class_names or calibration.class_names
    for split_name, declared in (("training", training.class_names), ("calibration", calibration.class_names)):
        if declared is not None and (len(declared) != args.number_of_classes or tuple(declared) != tuple(class_names)):
            raise ValueError(f"{split_name} dataset class names do not match the selected class names")
    configuration = TrainingConfig(
        number_of_classes=args.number_of_classes,
        exemplar_dimension=args.exemplar_dimension,
        epochs=args.epochs,
        cross_entropy_epochs=args.cross_entropy_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_neighbors=args.max_neighbors,
        alpha_resolution=args.alpha_resolution,
    )
    backend_options = {
        "device": args.device,
        "matching_query_batch_size": args.matching_query_batch_size,
        "matching_support_tile_size": args.matching_support_tile_size,
    }
    def progress(values):
        print(json.dumps(dict(values), separators=(",", ":")), file=sys.stderr)

    result = train_iterations(
        configuration,
        training.vectors,
        training.labels,
        calibration.vectors,
        calibration.labels,
        backend=args.backend,
        backend_options=backend_options,
        number_of_random_shuffles=args.number_of_random_shuffles,
        shuffle_training_and_calibration=args.shuffle_training_and_calibration,
        representation_fingerprint=representation_fingerprint,
        train_ids=training.ids,
        calibration_ids=calibration.ids,
        class_names=class_names,
        representation_provider=args.representation_provider,
        representation_model=args.representation_model,
        progress=progress,
        initial_artifact=initial_artifact,
    )
    write_artifact(args.output, result.artifact, overwrite=args.overwrite)
    _write_json(
        {
            "modelID": result.artifact.model_id,
            "artifact": str(Path(args.output)),
            "bestEpoch": result.best_epoch,
            "bestIteration": result.best_iteration,
            "durationSeconds": result.duration_seconds,
            "trainingIterations": result.number_of_iterations,
            "shuffledTrainingAndCalibration": args.shuffle_training_and_calibration,
            "bestBalancedCalibrationSDMLoss": result.best_balanced_calibration_loss,
            "bestIterationSplits": result.artifact.manifest["metadata"]["bestIterationSplits"],
            "history": list(result.history),
        },
        args.report_output,
    )
    return 0


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Path to a .sdmkitmodel directory")
    parser.add_argument("--no_verify_checksums", action="store_true", help='Skip artifact SHA-256 verification')
    parser.add_argument("--estimator", choices=["centroid", "lower"], default="centroid", help='Estimator used by selection decisions')
    parser.add_argument(
        "--matching_backend",
        choices=["torch"],
        default="torch",
        help="Inference backend for adaptor projection and exact matching (PyTorch)",
    )
    parser.add_argument(
        "--matching_device",
        default="auto",
        help="Inference device: cpu, mps, cuda, or cuda:N; auto selects available hardware",
    )
    parser.add_argument("--matching_query_batch_size", type=int, default=256, help='Queries per exact-matching batch')
    parser.add_argument("--matching_support_tile_size", type=int, default=16384, help='Support rows per exact-matching tile')


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="JSONL file or .sdmdataset bundle")
    parser.add_argument(
        "--composition",
        choices=["auto", "embedding", "attributes", "embedding+attributes"],
        default="auto",
        help="Feature columns to use; auto resolves from the dataset",
    )
    parser.add_argument("--representation_fingerprint", help='Require this representation identity; omitted uses the dataset declaration')


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="sdm", description="Portable SDM SDK with PyTorch execution")
    subparsers = parser.add_subparsers(dest="command", required=True)

    artifact = subparsers.add_parser("artifact", help="Validate or inspect a portable model")
    artifact_subcommands = artifact.add_subparsers(dest="artifact_command", required=True)
    validate = artifact_subcommands.add_parser("validate", help="Check model structure and checksums")
    validate.add_argument("--model", required=True, help='Path to a .sdmkitmodel directory')
    validate.add_argument("--output", default="-", help='JSON output file, or - for stdout')
    validate.add_argument("--no_verify_checksums", action="store_true", help='Skip artifact SHA-256 verification')
    validate.set_defaults(run=_run_artifact_validate)
    inspect = artifact_subcommands.add_parser("inspect", help="Print model metadata without loading a scoring runtime")
    _add_model_arguments(inspect)
    inspect.add_argument("--output", default="-", help='JSON output file, or - for stdout')
    inspect.set_defaults(run=_run_artifact_inspect)

    dataset = subparsers.add_parser("dataset", help="Convert, inspect, or validate JSONL and binary datasets")
    dataset_subcommands = dataset.add_subparsers(dest="dataset_command", required=True)
    for name, action in (("inspect", _run_dataset_inspect), ("validate", _run_dataset_validate)):
        command = dataset_subcommands.add_parser(name, help="Print dataset metadata" if name == "inspect" else "Check dataset structure and checksums")
        command.add_argument("--input", required=True, help="JSONL file or .sdmdataset bundle; features are optional")
        command.add_argument("--output", default="-", help='JSON output file, or - for stdout')
        command.set_defaults(run=action)
    convert = dataset_subcommands.add_parser("convert", help="Convert between JSONL and binary datasets")
    convert.add_argument("--input", required=True, help="JSONL file or .sdmdataset bundle; absent JSONL labels become -1 (unlabeled)")
    convert.add_argument("--output", required=True, help=".sdmdataset, .jsonl, .ndjson, or - for JSONL stdout")
    convert.add_argument("--class_names", help="Optional comma-separated class names")
    convert.add_argument("--representation_fingerprint", help='Require this representation identity; omitted uses the dataset declaration')
    convert.add_argument("--overwrite", action="store_true", help='Replace an existing output')
    convert.set_defaults(run=_run_dataset_convert)

    sources = dataset_subcommands.add_parser("export-sources", help="Attach original or winning source rows to an exact model")
    sources.add_argument("--model", required=True, help='Path to a .sdmkitmodel directory')
    sources.add_argument("--training", help="Original training JSONL or .sdmdataset; independently optional")
    sources.add_argument("--calibration", help="Original calibration JSONL or .sdmdataset; independently optional")
    sources.add_argument("--role", required=True, choices=SOURCE_ROLES, help='Source split to export')
    sources.add_argument("--output", required=True, help="Destination .sdmdataset source companion")
    sources.add_argument("--composition", choices=["auto", "embedding", "attributes", "embedding+attributes"], default="auto", help='Feature columns to use; auto resolves from the dataset')
    sources.add_argument("--text_only", action="store_true", help="Omit cached features and any incoming score fields")
    sources.add_argument("--with_scores", action="store_true", help="Compute full scores with validated training self-exclusions")
    sources.add_argument("--matching_device", default="auto", help="Scoring/feature-check device: cpu, mps, cuda, or auto for available hardware")
    sources.add_argument("--matching_query_batch_size", type=int, default=256, help='Queries per exact-matching batch')
    sources.add_argument("--matching_support_tile_size", type=int, default=16384, help='Support rows per exact-matching tile')
    sources.add_argument("--overwrite", action="store_true", help='Replace an existing output')
    sources.set_defaults(run=_run_dataset_export_sources)

    score = subparsers.add_parser("score", help='Write individual predictions and diagnostics; labels are optional')
    _add_model_arguments(score)
    _add_dataset_arguments(score)
    score.add_argument("--output", default="-", help="Full scored JSONL, stdout (-), or a .sdmdataset bundle")
    score.add_argument("--overwrite", action="store_true", help="Replace an existing .sdmdataset output")
    score.add_argument("--detail", choices=["full", "compact"], default="full",
                       help="Full includes source fields and importable SDM diagnostics; compact is a report only")
    score.add_argument("--score_batch_size", type=int, default=256, help='Source rows scored per output batch')
    score.add_argument("--nearest_exemplars", type=int, default=25,
                       help="Nearest exemplars to return; 0 keeps only singular nearest-support fields")
    score.add_argument("--identity_support_index_field",
                       help="Explicit source field holding support indices to exclude; IDs alone never exclude a row")
    score.set_defaults(run=_run_score)

    evaluate = subparsers.add_parser("evaluate", help='Summarize predictions on labeled data')
    _add_model_arguments(evaluate)
    _add_dataset_arguments(evaluate)
    evaluate.add_argument("--output", default="-", help='JSON output file, or - for stdout')
    evaluate.add_argument("--histogram_bins", type=int, default=10, help='Bins used in distribution summaries')
    evaluate.set_defaults(run=_run_evaluate)

    recalibrate = subparsers.add_parser("recalibrate", help="Refit calibrated regions from a saved model",
        description="Refit nested regions at a new alpha resolution using saved calibration diagnostics. "
                    "Saves an updated model without training or exemplar matching.")
    recalibrate.add_argument("--model", required=True, help="Source .sdmkitmodel retaining its calibration diagnostics")
    recalibrate.add_argument("--output", required=True, help="Destination .sdmkitmodel directory")
    recalibrate.add_argument("--alpha_resolution", type=float, default=0.05,
                            help="Spacing between descending calibrated alpha levels; finite, at least 0.00005 and below 0.5")
    recalibrate.add_argument("--report_output", default="-", help="Recalibration report JSON file, or - for stdout")
    recalibrate.add_argument("--overwrite", action="store_true", help="Replace an existing model package after checking its structure")
    recalibrate.set_defaults(run=_run_recalibrate)

    train = subparsers.add_parser("train", help="Train a new model or continue training",
        description="Fit the adaptor and calibrated regions from labeled training and calibration data. "
                    "Use --initial_model to continue training from a saved model.")
    train.add_argument("--training", required=True, help='Labeled training JSONL or .sdmdataset')
    train.add_argument("--calibration", required=True, help='Labeled calibration JSONL or .sdmdataset')
    train.add_argument("--output", required=True,
                       help="Destination .sdmkitmodel directory")
    train.add_argument("--report_output", default="-", help='Training report JSON file, or - for stdout')
    train.add_argument("--overwrite", action="store_true", help='Replace an existing model package after checking its structure')
    train.add_argument("--number_of_classes", type=int, required=True, help='Number of classes, with labels 0 through N-1')
    train.add_argument("--class_names", help='Ordered comma-separated names; otherwise use dataset/model names or Class0, Class1, ...')
    train.add_argument("--exemplar_dimension", type=int, default=1000, help='Adaptor exemplar dimension; continuation uses the saved dimension')
    train.add_argument("--epochs", type=int, default=20, help='Maximum complete training epochs per iteration')
    train.add_argument("--cross_entropy_epochs", type=int, default=1,
                       help="Leading CE-equivalent epochs; values above one defer matching until CE checkpoint selection")
    train.add_argument("--initial_model",
                       help="Continue training from saved weights and normalization with fresh Adam state")
    train.add_argument("--batch_size", type=int, default=64, help='Training rows per optimizer batch')
    train.add_argument("--learning_rate", type=float, default=1.0e-5, help='Adam learning rate')
    train.add_argument(
        "--backend",
        choices=["torch"],
        default="torch",
        help="Training uses PyTorch on CPU, MPS, or CUDA",
    )
    train.add_argument(
        "--device",
        default="auto",
        help="PyTorch device (auto, cpu, mps, cuda, or cuda:N); auto prefers CUDA, then MPS, then CPU",
    )
    train.add_argument(
        "--matching_query_batch_size",
        type=int,
        default=256,
        help="PyTorch matching query batch size",
    )
    train.add_argument(
        "--matching_support_tile_size",
        type=int,
        default=16384,
        help="PyTorch matching support tile size",
    )
    train.add_argument("--seed", type=int, default=0, help='Initial seed; independent iterations increment it')
    train.add_argument(
        "--number_of_random_shuffles", type=int, default=1,
        help="J independent training iterations; lowest balanced calibration SDM loss wins (last tie)",
    )
    shuffle_options = train.add_mutually_exclusive_group()
    shuffle_options.add_argument(
        "--shuffle_training_and_calibration",
        dest="shuffle_training_and_calibration", action="store_true",
        help="Pool and uniformly repartition 50:50 before each iteration, including iteration 1",
    )
    shuffle_options.add_argument(
        "--do_not_shuffle_data",
        dest="shuffle_training_and_calibration", action="store_false",
        help="Keep original split membership; split shuffling defaults to %(default)s",
    )
    train.set_defaults(shuffle_training_and_calibration=True)
    train.add_argument("--max_neighbors", type=int, default=2048, help='Maximum neighbors used for similarity q')
    train.add_argument("--alpha_resolution", type=float, default=0.05, help='Spacing between descending calibrated alpha levels; finite, at least 0.00005 and below 0.5')
    train.add_argument("--representation_fingerprint", help='Feature identity; omitted uses dataset declarations, the initial model, or embedding_v1')
    train.add_argument("--representation_provider", help='Feature provider metadata; omitted uses the initial model or precomputed')
    train.add_argument("--representation_model", help='Feature model metadata; omitted retains the initial model value or stays unset')
    train.add_argument(
        "--composition",
        choices=["auto", "embedding", "attributes", "embedding+attributes"],
        default="auto",
        help="Feature columns to use; auto resolves from the dataset",
    )
    train.set_defaults(run=_run_train)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.run(args))
    except (SDMError, OSError, ValueError, TypeError, ImportError, json.JSONDecodeError) as error:
        print(f"sdm: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
