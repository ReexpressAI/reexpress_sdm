# Copyright Reexpress AI, Inc. All rights reserved.
"""Standard-library tests for the standalone documentation calculator."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "estimate_capacity.py"
SPEC = importlib.util.spec_from_file_location("estimate_capacity", SCRIPT)
calculator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calculator
SPEC.loader.exec_module(calculator)


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.settings = calculator.Settings(ram_gib=128, embedding_dimension=128, exemplar_dimension=32)

    def test_known_production_manifest_budget(self):
        result = calculator.evaluate(replace(self.settings, embedding_dimension=16130, exemplar_dimension=1000),
                                     1_000_000, 1_000_000)
        self.assertEqual(result["estimated_manifest_bytes"], 396_310_480)
        self.assertEqual(result["payload_bytes"]["input_float32"], 129_040_000_000)
        self.assertEqual(result["payload_bytes"]["support_float32"], 4_000_000_000)
        self.assertTrue(all(result["capacity_checks"].values()))

    def test_alpha_ladder_rounding(self):
        self.assertEqual(calculator.candidate_alpha_count(.01), 49)
        self.assertEqual(calculator.candidate_alpha_count(.00005), 9999)
        self.assertEqual(calculator.candidate_alpha_count(.49999999999), 0)

    def test_split_policy_changes_support_and_calibration_budget(self):
        shuffled = calculator.evaluate(self.settings, 91, 10)
        fixed = calculator.evaluate(replace(self.settings, do_not_shuffle_data=True), 91, 10)
        self.assertEqual(shuffled["effective_rows"], {"training_support": 50, "calibration": 51})
        self.assertEqual(fixed["effective_rows"], {"training_support": 91, "calibration": 10})
        self.assertEqual(shuffled["estimated_manifest_bytes"] - fixed["estimated_manifest_bytes"], 41 * 80)
        self.assertEqual(fixed["candidate_comparisons_per_full_matching_pass"], 101 * 91)

    def test_tensor_limit_is_inclusive(self):
        s = replace(self.settings, exemplar_dimension=1000)
        self.assertTrue(calculator.evaluate(s, 2_000_000, 2_000_000)["capacity_checks"]["support_tensor"])
        self.assertFalse(calculator.evaluate(s, 2_000_001, 2_000_001)["capacity_checks"]["support_tensor"])
        self.assertFalse(calculator.evaluate(replace(s, embedding_dimension=2_000_000), 2, 2)
                         ["capacity_checks"]["weight_tensor"])

    def test_jsonl_has_additional_python_feature_objects(self):
        bundle = calculator.evaluate(replace(self.settings, profile="python_sdmdataset"), 1000, 1000)
        jsonl = calculator.evaluate(replace(self.settings, profile="python_jsonl"), 1000, 1000)
        payload = bundle["payload_bytes"]["input_float32"]
        for end, factor in (("lower_bytes", 8), ("upper_bytes", 10)):
            self.assertEqual(jsonl["memory"]["system_or_unified"][end] - bundle["memory"]["system_or_unified"][end],
                             payload * factor)

    def test_composed_app_input_reserves_concatenation(self):
        single = calculator.evaluate(self.settings, 1000, 1000)
        composed = calculator.evaluate(replace(self.settings, embedding_dimension=64, attribute_dimension=64), 1000, 1000)
        self.assertEqual(single["payload_bytes"], composed["payload_bytes"])
        payload = single["payload_bytes"]["input_float32"]
        self.assertEqual(composed["memory"]["system_components_bytes"]["input_vectors"], [payload * 2] * 2)

    def test_cuda_pools_and_host_snapshots_are_separate(self):
        s = replace(self.settings, profile="python_sdmdataset", device="cuda", gpu_memory_gib=24)
        result = calculator.evaluate(s, 1_000_000, 1_000_000)
        self.assertEqual(result["memory"]["cuda_device"]["budget_bytes"], 18 * calculator.GIB)
        self.assertEqual(result["memory"]["system_or_unified"]["budget_bytes"], 96 * calculator.GIB)
        weights = result["payload_bytes"]["adaptor_weights_float32"]
        self.assertEqual(result["memory"]["system_components_bytes"]["host_weight_snapshots"], [2 * weights, 4 * weights])
        unified = calculator.evaluate(replace(s, device="mps", gpu_memory_gib=None), 1_000_000, 1_000_000)
        self.assertIsNone(unified["memory"]["cuda_device"])
        for end, multiplier in (("lower_bytes", 2), ("upper_bytes", 4)):
            self.assertEqual(unified["memory"]["system_or_unified"][end],
                             result["memory"]["system_or_unified"][end] + result["memory"]["cuda_device"][end] - multiplier * weights)

    def test_max_rows_boundaries_and_monotonicity(self):
        result = calculator.maximum_rows(self.settings, .5)
        self.assertGreater(result["cautious_total_rows"], 0)
        self.assertLessEqual(result["cautious_total_rows"], result["broader_total_rows"])
        for name, end in (("cautious_total_rows", "upper_bytes"), ("broader_total_rows", "lower_bytes")):
            n = result[name]
            at = calculator.evaluate(self.settings, n // 2, n - n // 2)
            after = calculator.evaluate(self.settings, (n + 1) // 2, n + 1 - (n + 1) // 2)
            self.assertTrue(all(at["capacity_checks"].values()))
            self.assertLessEqual(at["memory"]["system_or_unified"][end], at["memory"]["system_or_unified"]["budget_bytes"])
            self.assertTrue(not all(after["capacity_checks"].values()) or
                            after["memory"]["system_or_unified"][end] > after["memory"]["system_or_unified"]["budget_bytes"])
        smaller = calculator.maximum_rows(replace(self.settings, ram_gib=64), .5)
        self.assertLessEqual(smaller["cautious_total_rows"], result["cautious_total_rows"])

    def test_max_rows_respects_cuda_budget(self):
        s = replace(self.settings, profile="python_sdmdataset", device="cuda", gpu_memory_gib=4)
        result = calculator.maximum_rows(s, .5)
        n = result["cautious_total_rows"]
        self.assertGreater(n, 0)
        self.assertLessEqual(result["cautious_configuration"]["memory"]["cuda_device"]["upper_bytes"], 3 * calculator.GIB)
        following = calculator.evaluate(s, (n + 1) // 2, n + 1 - (n + 1) // 2)
        self.assertTrue(any(pool["upper_bytes"] > pool["budget_bytes"] for pool in
                            (following["memory"]["system_or_unified"], following["memory"]["cuda_device"])))

    def test_format_ceiling_does_not_depend_on_small_dimensions(self):
        a = calculator.maximum_rows(replace(self.settings, ram_gib=10000), .5)
        b = calculator.maximum_rows(replace(self.settings, ram_gib=10000, embedding_dimension=64), .5)
        self.assertEqual(a["format_checks_total_rows"], 21_680_084)
        self.assertEqual(a["format_checks_total_rows"], b["format_checks_total_rows"])

    def test_unaffordable_or_impossible_splits(self):
        self.assertEqual(calculator.maximum_rows(replace(self.settings, ram_gib=.01), .5)["cautious_total_rows"], 0)
        for fraction in (5e-324, 1 - 2**-53):
            self.assertEqual(calculator.maximum_rows(self.settings, fraction)["format_checks_total_rows"], 0)

    def test_exact_matching_work_scales_quadratically(self):
        a = calculator.evaluate(self.settings, 1000, 1000)
        b = calculator.evaluate(self.settings, 10000, 10000)
        self.assertEqual(b["candidate_comparisons_per_full_matching_pass"], 100 * a["candidate_comparisons_per_full_matching_pass"])


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args, script=SCRIPT):
        # -I -S excludes site-packages, user settings, and the local SDM package.
        return subprocess.run([sys.executable, "-I", "-S", str(script), *args], capture_output=True, text=True)

    def test_copied_file_runs_without_package_or_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "estimate_capacity.py"
            shutil.copyfile(SCRIPT, copied)
            result = self.run_cli("estimate", "--ram_gib", "128", "--embedding_dimension", "1000",
                                  "--training_rows", "452833", "--calibration_rows", "34599", "--json", script=copied)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["result"]["original_rows"]["total"], 487432)
        self.assertEqual(report["result"]["effective_rows"]["training_support"], 243716)

    def test_invalid_inputs_are_usage_errors(self):
        cases = (("--ram_gib", "nan"), ("--ram_gib", "1e308"),
                 ("--additional_metadata_mib", "1e308"), ("--memory_fraction", "0"),
                 ("--embedding_dimension", "-1"), ("--number_of_classes", "6"),
                 ("--device", "cuda", "--gpu_memory_gib", "80"),
                 ("--gpu_memory_gib", "80"), ("--alpha_resolution", ".5"),
                 ("--matching_query_batch_size", "8"), ("--matching_support_tile_size", "64"),
                 ("--training_rows", "20"))
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self.run_cli("max_rows", "--ram_gib", "128", "--embedding_dimension", "128", *arguments)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertNotIn("Traceback", result.stderr)

    def test_tiny_fraction_returns_zero_without_traceback(self):
        result = self.run_cli("max_rows", "--ram_gib", "128", "--embedding_dimension", "128",
                              "--training_fraction", "5e-324", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["result"]["cautious_total_rows"], 0)

    def test_app_cpu_resolves_matching_tile(self):
        result = self.run_cli("max_rows", "--ram_gib", "128", "--embedding_dimension", "128", "--device", "cpu", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["settings"]["matching_support_tile_size"], 4096)

    def test_app_pooling_accepts_small_original_calibration(self):
        args = ("estimate", "--ram_gib", "128", "--embedding_dimension", "128",
                "--training_rows", "3", "--calibration_rows", "1", "--json")
        result = self.run_cli(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["result"]["effective_rows"], {"training_support": 2, "calibration": 2})
        self.assertEqual(self.run_cli(*args, "--do_not_shuffle_data").returncode, 2)
        self.assertEqual(self.run_cli(*args, "--profile", "python_jsonl").returncode, 2)


if __name__ == "__main__":
    unittest.main()
