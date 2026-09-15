# Copyright Reexpress AI, Inc. All rights reserved.
"""Portable Similarity-Distance-Magnitude SDK with PyTorch execution."""

from .artifact import load_artifact, validate_artifact, validate_manifest, write_artifact
from .backends import create_dense_index, create_training_backend
from .calibration import NestedCalibrator, calibrate_nested_regions
from .controller import SDMController
from .dataset import Dataset
from .bundle import DatasetBundle, iter_dataset_rows, validate_dataset_rows, write_dataset_bundle, write_dataset_jsonl
from .errors import (
    ArtifactValidationError,
    CalibrationError,
    DatasetValidationError,
    DimensionMismatchError,
    PolicyValidationError,
    RepresentationMismatchError,
    SDMError,
)
from .index import DenseIndex, ExactL2Index
from .iterative_training import IterativeTrainingResult, train_iterations
from .metrics import Evaluator, evaluate_scores
from .model import SDMModel
from .monitoring import distribution_summary
from .policy import SelectionPolicy
from .recalibration import recalibrate_artifact
from .score_io import score_document, score_dataset_rows, score_to_document, write_scored_jsonl
from .source_io import export_source_dataset
from .source_data import feature_digest
from .training import TrainingBackend, TrainingConfig, TrainingResult, TrainingControl, TrainingCancelled, build_artifact
from .torch_backend import TorchTrainer
from .types import (
    AdapterWeights,
    ControlDecision,
    EstimatorKind,
    Region,
    SDMArtifact,
    SDMScore,
    SupportMatch,
    SupportRecord,
)

__all__ = [
    "AdapterWeights",
    "ArtifactValidationError",
    "CalibrationError",
    "ControlDecision",
    "Dataset",
    "DatasetBundle",
    "DatasetValidationError",
    "DenseIndex",
    "DimensionMismatchError",
    "EstimatorKind",
    "Evaluator",
    "ExactL2Index",
    "IterativeTrainingResult",
    "NestedCalibrator",
    "TorchTrainer",
    "PolicyValidationError",
    "Region",
    "RepresentationMismatchError",
    "SDMArtifact",
    "SDMController",
    "SDMError",
    "SDMModel",
    "SDMScore",
    "SelectionPolicy",
    "SupportRecord",
    "SupportMatch",
    "TrainingBackend",
    "TrainingConfig",
    "TrainingControl",
    "TrainingCancelled",
    "TrainingResult",
    "build_artifact",
    "create_dense_index",
    "create_training_backend",
    "calibrate_nested_regions",
    "distribution_summary",
    "evaluate_scores",
    "export_source_dataset",
    "feature_digest",
    "load_artifact",
    "recalibrate_artifact",
    "validate_artifact",
    "validate_manifest",
    "write_artifact",
    "iter_dataset_rows",
    "validate_dataset_rows",
    "write_dataset_bundle",
    "write_dataset_jsonl",
    "score_document",
    "score_dataset_rows",
    "score_to_document",
    "write_scored_jsonl",
    "train_iterations",
]

__version__ = "0.4.5"
