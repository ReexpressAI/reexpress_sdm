# Copyright Reexpress AI, Inc. All rights reserved.
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reexpress_sdm import (
    ArtifactValidationError, CalibrationError, NestedCalibrator, TrainingConfig,
    load_artifact, recalibrate_artifact, validate_artifact, validate_manifest,
    write_artifact,
)
from reexpress_sdm.math import ladder_alphas
from reexpress_sdm.training_metadata import validate_training_run

from helpers import make_artifact


class AlphaResolutionSafeguardTests(unittest.TestCase):
    def test_entry_points_reject_unsafe_resolutions(self):
        source = make_artifact()
        invalid = (
            1e-20, np.nextafter(0.00005, 0), 0, -0.1, 0.5,
            float("inf"), float("nan"), True, "0.05", None,
        )
        entry_points = (
            (ladder_alphas, ValueError),
            (lambda value: TrainingConfig(2, alpha_resolution=value), ValueError),
            (lambda value: NestedCalibrator(2, value), CalibrationError),
            (lambda value: recalibrate_artifact(source, alpha_resolution=value), CalibrationError),
        )
        for entry_point, error_type in entry_points:
            for value in invalid:
                with self.subTest(entry_point=entry_point, value=value):
                    with self.assertRaisesRegex(error_type, "alpha_resolution"):
                        entry_point(value)

    def test_manifest_and_loader_reject_before_building_ladder_or_reading_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_artifact(Path(directory) / "model", make_artifact())
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["configuration"]["alphaResolution"] = 1e-20
            manifest["regions"] = []
            manifest_path.write_text(json.dumps(manifest))
            with patch("reexpress_sdm.artifact.ladder_alphas", side_effect=AssertionError("unsafe ladder")), \
                 patch("reexpress_sdm.artifact._read_exact_f32", side_effect=AssertionError("payload read")):
                for operation in (lambda: validate_manifest(manifest), lambda: load_artifact(path)):
                    with self.assertRaisesRegex(ArtifactValidationError, "alphaResolution.*0.00005"):
                        operation()

    def test_lower_boundary_is_supported_and_existing_ladders_are_unchanged(self):
        levels = ladder_alphas(0.00005)
        self.assertEqual(len(levels), 9999)
        self.assertEqual((levels[0], levels[-1]), (0.99995, 0.50005))
        self.assertTrue(all(a > b for a, b in zip(levels, levels[1:])))
        self.assertEqual(ladder_alphas(0.05), (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55))
        self.assertEqual(ladder_alphas(0.1), (0.9, 0.8, 0.7, 0.6))
        self.assertEqual(TrainingConfig(2, alpha_resolution=0.00005).alpha_resolution, 0.00005)
        self.assertEqual(NestedCalibrator(2, 0.00005).alpha_resolution, 0.00005)
        source = make_artifact()
        manifest = deepcopy(source.manifest)
        manifest["configuration"]["alphaResolution"] = 0.00005
        with tempfile.TemporaryDirectory() as directory:
            path = write_artifact(Path(directory) / "model", replace(source, manifest=manifest))
            self.assertEqual(load_artifact(path).configuration["alphaResolution"], 0.00005)
        rows = (
            {"id": "c0", "label": 0, "prediction": 0, "sdm": [0.99, 0.01], "qPrime": 2.0},
            {"id": "c1", "label": 1, "prediction": 1, "sdm": [0.01, 0.99], "qPrime": 2.0},
        )
        result = recalibrate_artifact(replace(source, calibration_rows=rows), alpha_resolution=0.00005)
        self.assertEqual(result.configuration["alphaResolution"], 0.00005)
        self.assertEqual(result.calibration_rows, rows)
        validate_artifact(result)

    def test_training_history_uses_same_lower_bound(self):
        fixture = Path(__file__).resolve().parent / "fixtures/contracts-v1/portable-training-run.json"
        value = json.loads(fixture.read_text())
        value["configuration"]["alphaResolution"] = 0.00005
        self.assertEqual(validate_training_run(value), value)
        for resolution in (1e-20, np.nextafter(0.00005, 0), 0.5):
            value["configuration"]["alphaResolution"] = float(resolution)
            with self.subTest(resolution=resolution), self.assertRaisesRegex(ValueError, "alphaResolution"):
                validate_training_run(value)


class ArtifactOverwriteSafeguardTests(unittest.TestCase):
    def test_unrelated_directory_and_regular_file_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unrelated = root / "unrelated"
            unrelated.mkdir()
            marker = unrelated / "important.txt"
            marker.write_text("keep this")
            ordinary_file = root / "ordinary-file"
            ordinary_file.write_text("also keep this")
            for target in (unrelated, ordinary_file):
                with self.subTest(target=target), self.assertRaisesRegex(ArtifactValidationError, "refusing to overwrite"):
                    write_artifact(target, make_artifact(), overwrite=True)
            self.assertEqual(marker.read_text(), "keep this")
            self.assertEqual(ordinary_file.read_text(), "also keep this")
            self.assertEqual(set(root.iterdir()), {unrelated, ordinary_file})

    def test_symlink_destinations_are_preserved_even_if_the_referent_is_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = write_artifact(root / "model", make_artifact())
            original = (model / "manifest.json").read_bytes()
            for name, referent in (("link", model), ("dangling", root / "missing")):
                link = root / name
                link.symlink_to(referent, target_is_directory=True)
                with self.subTest(name=name), self.assertRaisesRegex(ArtifactValidationError, "non-symlink"):
                    write_artifact(link, make_artifact(), overwrite=True)
                self.assertTrue(link.is_symlink())
            self.assertEqual((model / "manifest.json").read_bytes(), original)
            self.assertFalse((root / "missing").exists())

    def test_incomplete_or_malformed_model_structure_is_preserved(self):
        for damage in ("manifest", "missing-records", "tensor-size", "symlink-member", "dangling-calibration"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = write_artifact(root / "model", make_artifact())
                marker = model / "keep.txt"
                marker.write_text("do not remove")
                if damage == "manifest":
                    (model / "manifest.json").write_text("{}")
                elif damage == "missing-records":
                    (model / "support.jsonl").unlink()
                elif damage == "tensor-size":
                    (model / "support.f32").write_bytes(b"short")
                elif damage == "symlink-member":
                    (model / "support.f32").rename(root / "outside.f32")
                    (model / "support.f32").symlink_to(root / "outside.f32")
                else:
                    (model / "calibration.jsonl").symlink_to(root / "missing")
                with self.assertRaisesRegex(ArtifactValidationError, "valid model package structure"):
                    write_artifact(model, make_artifact(), overwrite=True)
                self.assertEqual(marker.read_text(), "do not remove")

    def test_valid_model_overwrite_checks_structure_without_loading_old_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = write_artifact(root / "model", make_artifact())
            source = make_artifact()
            manifest = deepcopy(source.manifest)
            manifest["modelID"] = "replacement"
            # Loading the freshly staged output remains required. Loading the
            # model being discarded would duplicate large tensor allocations.
            real_load = load_artifact
            def staged_load(path, **kwargs):
                self.assertNotEqual(Path(path), model)
                return real_load(path, **kwargs)
            with patch("reexpress_sdm.artifact.load_artifact", side_effect=staged_load) as loader:
                write_artifact(model, replace(source, manifest=manifest), overwrite=True)
                self.assertEqual(loader.call_count, 1)
            self.assertEqual(load_artifact(model).model_id, "replacement")
            self.assertEqual(list(root.iterdir()), [model])

    def test_existing_model_still_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            model = write_artifact(Path(directory) / "model", make_artifact())
            with self.assertRaises(FileExistsError):
                write_artifact(model, make_artifact())
            self.assertEqual(load_artifact(model).model_id, "fixture-model")

    def test_newly_appearing_or_replaced_destination_is_preserved(self):
        for initially_exists in (False, True):
            with self.subTest(initially_exists=initially_exists), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / "model"
                saved = root / "saved-original"
                if initially_exists:
                    write_artifact(target, make_artifact())
                real_load = load_artifact
                def intervene(path, **kwargs):
                    result = real_load(path, **kwargs)
                    if initially_exists:
                        target.rename(saved)
                    target.mkdir()
                    (target / "important.txt").write_text("preserve new destination")
                    return result
                with patch("reexpress_sdm.artifact.load_artifact", side_effect=intervene):
                    with self.assertRaisesRegex(FileExistsError, "destination changed"):
                        write_artifact(target, make_artifact(), overwrite=True)
                self.assertEqual((target / "important.txt").read_text(), "preserve new destination")
                expected = {target}
                if initially_exists:
                    expected.add(saved)
                    self.assertEqual(load_artifact(saved).model_id, "fixture-model")
                self.assertEqual(set(root.iterdir()), expected)


if __name__ == "__main__":
    unittest.main()
