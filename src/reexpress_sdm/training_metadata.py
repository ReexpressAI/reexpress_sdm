# Copyright Reexpress AI, Inc. All rights reserved.
"""Portable, backend-independent completed training history.

The same ``metadata.trainingRun`` object is written by Reexpress two and Python.
It carries measurements and options, never machine-local dataset identifiers.
"""

from __future__ import annotations

import copy
import math
import struct
from typing import Any, Mapping, Sequence

from .math import MIN_ALPHA_RESOLUTION

_METRICS = (
    "balancedTrainingSDMLoss", "balancedCalibrationSDMLoss",
    "balancedTrainingCELoss", "balancedCalibrationCELoss",
    "balancedTrainingAccuracy", "balancedCalibrationAccuracy",
    "balancedMeanTrainingQ", "balancedMeanCalibrationQ",
)


def build_training_run(
    configuration: Any, history: Sequence[Mapping[str, Any]], *,
    best_epoch: int, best_balanced_calibration_loss: float, backend: str,
    best_iteration: int = 1, completed_iterations: int = 1,
    requested_iterations: int = 1, shuffle_training_and_calibration: bool = False,
    source_model_id: str | None = None, stopped_early: bool = False,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    rows = []
    for row in history:
        rows.append({
            "iteration": int(row.get("iteration", 1)),
            "epoch": int(row["epoch"]),
            "marginalTrainingLoss": float(row["trainingLoss"]),
            **{key: None if row.get(key) is None else float(row[key]) for key in _METRICS},
            "isBest": bool(row.get("isBest", False)),
            "durationSeconds": None if row.get("durationSeconds") is None else float(row["durationSeconds"]),
        })
    value = {
        "schemaVersion": 1,
        "configuration": {
            "epochs": int(configuration.epochs),
            "batchSize": int(configuration.batch_size),
            "learningRate": float(configuration.learning_rate),
            "exemplarDimension": int(configuration.exemplar_dimension),
            "maximumNeighbors": int(configuration.max_neighbors),
            "alphaResolution": float(configuration.alpha_resolution),
            # JSONValue uses Double in Swift; decimal text preserves all 64 bits.
            "seed": str(configuration.seed),
            "crossEntropyEpochs": int(configuration.cross_entropy_epochs),
            "iterations": int(requested_iterations),
            "shuffleTrainingAndCalibration": bool(shuffle_training_and_calibration),
            "initialization": "activeModel" if source_model_id is not None else "fresh",
        },
        "backend": backend,
        "history": rows,
        "bestIteration": int(best_iteration),
        "iterationCount": int(completed_iterations),
        "bestEpoch": int(best_epoch),
        "bestBalancedCalibrationLoss": float(best_balanced_calibration_loss),
        "sourceModelID": source_model_id,
        "stoppedEarly": bool(stopped_early),
        "durationSeconds": duration_seconds,
    }
    return validate_training_run(value)


def validate_training_run(value: Any) -> dict[str, Any]:
    """Reject partial, mislabeled, or inconsistent portable histories."""
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError("metadata.trainingRun: " + message)

    def integer(item: Any, minimum: int = 1) -> bool:
        return isinstance(item, int) and not isinstance(item, bool) and item >= minimum

    def number(item: Any, minimum: float = 0.0) -> bool:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            return False
        try:
            return math.isfinite(item) and item >= minimum
        except OverflowError:
            return False

    require(isinstance(value, Mapping), "must be an object")
    require(type(value.get("schemaVersion")) is int and value["schemaVersion"] == 1, "unsupported schemaVersion")
    config = value.get("configuration")
    require(isinstance(config, Mapping), "configuration must be an object")
    for key in ("epochs", "batchSize", "exemplarDimension", "maximumNeighbors", "crossEntropyEpochs", "iterations"):
        require(integer(config.get(key)), f"configuration.{key} must be positive")
    require(config["maximumNeighbors"] >= 2, "maximumNeighbors must allow identity exclusion")
    require(config["crossEntropyEpochs"] <= config["epochs"], "crossEntropyEpochs exceeds epochs")
    require(number(config.get("learningRate")) and config["learningRate"] > 0, "learningRate must be positive")
    try:
        rate = struct.unpack("<f", struct.pack("<f", config["learningRate"]))[0]
    except OverflowError:
        rate = math.inf
    require(math.isfinite(rate) and rate > 0, "learningRate must remain finite and positive as Float32")
    require(number(config.get("alphaResolution")) and MIN_ALPHA_RESOLUTION <= config["alphaResolution"] < 0.5, "invalid alphaResolution")
    seed = config.get("seed")
    require(isinstance(seed, str) and seed.isascii() and seed.isdecimal() and len(seed) <= 20 and int(seed) < 2**64, "seed must be a decimal UInt64 string")
    require(config.get("initialization") in ("fresh", "activeModel"), "invalid initialization")
    require(type(config.get("shuffleTrainingAndCalibration")) is bool, "shuffleTrainingAndCalibration must be boolean")
    require(isinstance(value.get("backend"), str) and bool(value["backend"]), "backend must be a nonempty string")
    for key in ("bestIteration", "iterationCount", "bestEpoch"):
        require(integer(value.get(key)), f"{key} must be positive")
    require(value["bestIteration"] <= value["iterationCount"] <= config["iterations"], "invalid iteration bounds")
    require(value["bestEpoch"] <= config["epochs"], "invalid bestEpoch")
    require(number(value.get("bestBalancedCalibrationLoss")), "invalid selected loss")
    require(type(value.get("stoppedEarly")) is bool, "stoppedEarly must be boolean")
    require(value.get("durationSeconds") is None or number(value["durationSeconds"]), "invalid durationSeconds")
    source = value.get("sourceModelID")
    require(source is None or isinstance(source, str) and bool(source), "sourceModelID must be a nonempty string or null")
    require((source is not None) == (config["initialization"] == "activeModel"), "initialization and sourceModelID disagree")
    rows = value.get("history")
    require(isinstance(rows, list) and len(rows) > 0, "history must contain completed epochs")
    previous = (1, 0)
    selected = None
    for row in rows:
        require(isinstance(row, Mapping), "history row must be an object")
        iteration, epoch = row.get("iteration"), row.get("epoch")
        require(integer(iteration) and integer(epoch), "invalid history position")
        require(iteration <= value["iterationCount"] and epoch <= config["epochs"], "history position exceeds configuration")
        require((iteration, epoch) == (previous[0], previous[1] + 1) or
                (iteration == previous[0] + 1 and epoch == 1 and previous[1] == config["epochs"]),
                "history must contain contiguous completed epochs")
        previous = (iteration, epoch)
        require(number(row.get("marginalTrainingLoss")), "invalid marginalTrainingLoss")
        require(type(row.get("isBest")) is bool, "isBest must be boolean")
        require(row.get("durationSeconds") is None or number(row["durationSeconds"]), "invalid epoch durationSeconds")
        for key in _METRICS:
            require(key in row, f"missing metric {key}")
            require(row[key] is None or number(row[key]), f"invalid {key}")
        for key in ("balancedTrainingAccuracy", "balancedCalibrationAccuracy"):
            require(number(row[key]) and row[key] <= 1, f"{key} must be in [0, 1]")
        sdm_keys = ("balancedTrainingSDMLoss", "balancedCalibrationSDMLoss", "balancedMeanTrainingQ", "balancedMeanCalibrationQ")
        require(all(row[k] is None for k in sdm_keys) or all(row[k] is not None for k in sdm_keys), "incomplete SDM scoring")
        require((row["balancedTrainingCELoss"] is None) == (row["balancedCalibrationCELoss"] is None), "incomplete CE scoring")
        require(row["balancedCalibrationSDMLoss"] is not None or row["balancedCalibrationCELoss"] is not None, "epoch has no complete score")
        if (iteration, epoch) == (value["bestIteration"], value["bestEpoch"]):
            selected = row
    require(previous[0] == value["iterationCount"], "iterationCount differs from completed history")
    require(value["stoppedEarly"] == (previous != (config["iterations"], config["epochs"])), "stoppedEarly differs from completed history")
    require(selected is not None and selected["balancedCalibrationSDMLoss"] is not None, "selected checkpoint lacks complete SDM scoring")
    require(math.isclose(selected["balancedCalibrationSDMLoss"], value["bestBalancedCalibrationLoss"], rel_tol=1e-5, abs_tol=1e-7), "selected loss differs from history")
    return copy.deepcopy(dict(value))
