# Copyright Reexpress AI, Inc. All rights reserved.
"""Inference uses selected-device weights/matching with unchanged portable data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

from reexpress_sdm import Dataset, SDMModel, load_artifact, write_dataset_bundle


GOLDEN = Path(__file__).resolve().parent / "fixtures/contracts-v1"


class DeviceRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = load_artifact(GOLDEN / "python-training-lifecycle.sdmkitmodel")
        cls.inputs = json.loads((GOLDEN / "python-training-lifecycle-inputs.json").read_text())
        cls.expected = json.loads((GOLDEN / "python-training-lifecycle-expected.json").read_text())
        cls.values = np.asarray([row["embedding"] for row in cls.inputs["evaluation"]], np.float32)

    def check_runtime(self, backend, device):
        model = SDMModel(self.artifact, backend=backend, device=device, query_batch_size=2, support_tile_size=3)
        weights = model._runtime.weights
        index = model._index
        scores = model.score(self.values)
        for score, expected in zip(scores, self.expected["scores"]):
            self.assertEqual(score.prediction, expected["prediction"])
            self.assertEqual(score.q, expected["q"])
            np.testing.assert_allclose(score.z_prime, expected["zPrime"], atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(score.sdm, expected["sdm"], atol=2e-5, rtol=2e-5)
        model.score(self.values)
        self.assertIs(model._index, index)
        self.assertTrue(all(a is b for a, b in zip(weights, model._runtime.weights)))
        self.assertEqual(model.backend, backend)
        self.assertEqual(model.device.split(":"), device.split(":"))
        return model

    def test_torch_cpu_matches_golden_and_reuses_resident_weights_support(self):
        model = self.check_runtime("torch", "cpu")
        original = model._index.search_many_device
        received = []
        def inspect(values, *args, **kwargs):
            self.assertIsInstance(values, torch.Tensor)
            self.assertEqual(values.device.type, "cpu")
            received.append(values)
            return original(values, *args, **kwargs)
        with patch.object(model._index, "search_many_device", side_effect=inspect):
            model.score(torch.as_tensor(self.values))
        self.assertTrue(received)

    def test_transform_is_device_projected_and_host_interchange_is_preserved(self):
        model = SDMModel(self.artifact, device="cpu")
        weights = self.artifact.weights
        normalized = (self.values - model.normalization_mean) / model.normalization_standard_deviation
        expected_exemplars = normalized @ weights.projection_weight.T + weights.projection_bias
        expected_logits = expected_exemplars @ weights.classifier_weight.T + weights.classifier_bias
        logits, exemplars = model.transform(torch.as_tensor(self.values, dtype=torch.float64))
        np.testing.assert_allclose(exemplars, expected_exemplars, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(logits, expected_logits, atol=2e-6, rtol=2e-6)
        self.assertEqual((logits.dtype, exemplars.dtype), (np.dtype("float32"), np.dtype("float32")))
        self.assertEqual(model.score(np.empty((0, model.embedding_dimension), np.float32)), ())
        for bad in (torch.ones((1, model.embedding_dimension), dtype=torch.bool), torch.full((1, model.embedding_dimension), float("nan"))):
            with self.assertRaises(ValueError): model.score(bad)

    def check_readonly_bundle_batches(self, device):
        model = SDMModel(self.artifact, device=device, query_batch_size=2, support_tile_size=3)
        with tempfile.TemporaryDirectory() as temporary:
            path = write_dataset_bundle(Path(temporary) / "evaluation.sdmdataset", self.inputs["evaluation"])
            source_files = {file.name: file.read_bytes() for file in path.iterdir()}
            dataset = Dataset.load(path, composition="embedding")
            try:
                self.assertIsInstance(dataset.vectors, np.memmap)
                self.assertFalse(dataset.vectors.flags.writeable)
                self.assertTrue(np.shares_memory(model._runtime.inputs(dataset.vectors), dataset.vectors))
                writable = np.array(dataset.vectors, copy=True)
                expected = model.score(writable)
                as_tensor = torch.as_tensor
                for kind, values in (("readonly", dataset.vectors), ("writable", writable),
                                     ("resident", torch.tensor(writable, device=device))):
                    with self.subTest(input=kind):
                        batch_sizes = []

                        def inspect(batch, *args, **kwargs):
                            batch_sizes.append(len(batch))
                            if isinstance(batch, np.ndarray):
                                # Detect unsafe conversion even if Torch has already
                                # suppressed its once-per-process warning.
                                self.assertTrue(batch.flags.writeable)
                                self.assertEqual(np.shares_memory(batch, values), kind == "writable")
                                if kind == "readonly":
                                    self.assertTrue(batch.flags.owndata)
                            else:
                                self.assertIsInstance(batch, torch.Tensor)
                                self.assertEqual(batch.untyped_storage().data_ptr(),
                                                 values.untyped_storage().data_ptr())
                            return as_tensor(batch, *args, **kwargs)

                        with warnings.catch_warnings():
                            warnings.filterwarnings("error", message="The given NumPy array is not writable",
                                                    category=UserWarning)
                            with patch.object(torch, "as_tensor", side_effect=inspect):
                                actual = model.score(values)
                        self.assertEqual(actual, expected)
                        self.assertEqual(batch_sizes, [min(2, dataset.count - start)
                                                      for start in range(0, dataset.count, 2)])
                np.testing.assert_array_equal(dataset.vectors, writable)
                self.assertFalse(dataset.vectors.flags.writeable)
                self.assertEqual({file.name: file.read_bytes() for file in path.iterdir()}, source_files)
            finally:
                dataset.rows.close()

    def test_cpu_readonly_bundle_copies_only_input_batches_without_changing_scores(self):
        self.check_readonly_bundle_batches("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "native MPS is unavailable")
    def test_mps_readonly_bundle_copies_only_input_batches_without_changing_scores(self):
        self.check_readonly_bundle_batches("mps")

    def test_mixed_identity_exclusions_match_separate_scoring(self):
        model = SDMModel(self.artifact, device="cpu", query_batch_size=3)
        rows = np.asarray([row["embedding"] for row in self.inputs["training"][:3]], np.float32)
        identities = [0, None, 2]
        scores = model.score(rows, identity_support_indices=identities)
        for row, identity, score in zip(rows, identities, scores):
            individual = model.score(row, identity_support_indices=[identity])[0]
            self.assertEqual(score.q, individual.q)
            self.assertEqual(score.nearest_support_index, individual.nearest_support_index)
            np.testing.assert_allclose(score.sdm, individual.sdm, rtol=1e-5, atol=1e-5)

    def test_retired_backends_are_not_accepted(self):
        for backend in ("numpy", "mlx", "mlx-python"):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "inference backend must be 'torch'"):
                SDMModel(self.artifact, backend=backend)

    def test_outer_autocast_does_not_reduce_float32_inference_precision(self):
        model = SDMModel(self.artifact, device="cpu")
        expected = model.score(self.values)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model.score(torch.as_tensor(self.values))
        self.assertEqual(actual, expected)

    @unittest.skipUnless(torch.backends.mps.is_available(), "native MPS is unavailable")
    def test_mps_accepts_resident_input_tensors(self):
        model = self.check_runtime("torch", "mps")
        tensor = torch.tensor(self.values, device="mps")
        scores = model.score(tensor)
        self.assertEqual(len(scores), len(self.values))
        self.assertTrue(all(value.device.type == "mps" for value in model._runtime.weights))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_accepts_resident_input_tensors(self):
        device = f"cuda:{torch.cuda.current_device()}"
        model = self.check_runtime("torch", device)
        self.assertEqual(len(model.score(torch.tensor(self.values, device=device))), len(self.values))
