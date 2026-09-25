# Copyright Reexpress AI, Inc. All rights reserved.
"""Shared model Unicode domain without changing accepted source spellings."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from reexpress_sdm import ArtifactValidationError, SDMModel, load_artifact, write_artifact
from helpers import make_artifact


class ModelJSONDomainTests(unittest.TestCase):
    def base(self):
        rows = tuple({"id": f"cal-{i}", "label": i, "prediction": i,
                      "sdm": [.8, .2] if i == 0 else [.2, .8], "qPrime": 1} for i in range(2))
        return replace(make_artifact(), calibration_rows=rows)

    def test_canonical_equivalent_model_identities_are_rejected(self):
        for kind in ("classNames", "support", "calibration"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                base = self.base()
                manifest = copy.deepcopy(base.manifest)
                support, calibration = list(base.support_records), list(base.calibration_rows)
                if kind == "classNames":
                    manifest["configuration"]["classNames"] = ["é", "e\u0301"]
                elif kind == "support":
                    support[:2] = [replace(support[0], id="é"), replace(support[1], id="e\u0301")]
                else:
                    calibration = [{**calibration[0], "id": "é"}, {**calibration[1], "id": "e\u0301"}]
                invalid = replace(base, manifest=manifest, support_records=tuple(support), calibration_rows=tuple(calibration))
                with self.assertRaisesRegex(ArtifactValidationError, "canonic"):
                    SDMModel(invalid)
                path = Path(directory) / "model.sdmkitmodel"
                write_artifact(path, base)
                if kind == "classNames":
                    stored = json.loads((path / "manifest.json").read_text())
                    stored["configuration"]["classNames"] = manifest["configuration"]["classNames"]
                    (path / "manifest.json").write_text(json.dumps(stored))
                else:
                    filename = "support.jsonl" if kind == "support" else "calibration.jsonl"
                    stored = [json.loads(line) for line in (path / filename).read_text().splitlines()]
                    stored[0]["id"], stored[1]["id"] = "é", "e\u0301"
                    (path / filename).write_text("".join(json.dumps(row) + "\n" for row in stored))
                with self.assertRaisesRegex(ArtifactValidationError, "canonic"):
                    load_artifact(path, verify_checksums=False)

    def test_surrogate_metadata_values_and_keys_are_rejected_at_both_boundaries(self):
        for metadata in ({"nested": ["\ud800"]}, {"\udfff": "value"}):
            with self.subTest(metadata=repr(metadata)), tempfile.TemporaryDirectory() as directory:
                base = self.base()
                manifest = copy.deepcopy(base.manifest)
                manifest["metadata"] = metadata
                with self.assertRaisesRegex(ArtifactValidationError, "surrogate"):
                    SDMModel(replace(base, manifest=manifest))
                invalid_support = replace(base, support_records=(replace(base.support_records[0], metadata=metadata), *base.support_records[1:]))
                with self.assertRaisesRegex(ArtifactValidationError, "surrogate"):
                    SDMModel(invalid_support)
                path = Path(directory) / "model.sdmkitmodel"
                write_artifact(path, base)
                stored = json.loads((path / "manifest.json").read_text())
                stored["metadata"] = metadata
                (path / "manifest.json").write_text(json.dumps(stored))
                with self.assertRaisesRegex(ArtifactValidationError, "surrogate"):
                    load_artifact(path)

    def test_surrogates_in_known_model_text_are_rejected(self):
        for field in ("modelID", "producer", "classNames", "representation", "supportID", "supportDocument", "calibrationID"):
            with self.subTest(field=field):
                base = self.base()
                manifest = copy.deepcopy(base.manifest)
                support, calibration = list(base.support_records), list(base.calibration_rows)
                if field == "modelID": manifest["modelID"] = "\ud800"
                elif field == "producer": manifest["producer"]["name"] = "\ud800"
                elif field == "classNames": manifest["configuration"]["classNames"][0] = "\ud800"
                elif field == "representation": manifest["representation"]["inputTemplate"] = "\ud800"
                elif field == "supportID": support[0] = replace(support[0], id="\ud800")
                elif field == "supportDocument": support[0] = replace(support[0], document="\ud800")
                else: calibration[0] = {**calibration[0], "id": "\ud800"}
                with self.assertRaisesRegex(ArtifactValidationError, "surrogate"):
                    SDMModel(replace(base, manifest=manifest, support_records=tuple(support), calibration_rows=tuple(calibration)))

    def test_valid_unicode_spellings_are_preserved_exactly(self):
        base = self.base()
        manifest = copy.deepcopy(base.manifest)
        manifest["modelID"] = "mode\u0301l"
        manifest["configuration"]["classNames"] = ["e\u0301", "other"]
        manifest["metadata"] = {"e\u0301": ["é", "e\u0301", "🌍"]}
        source = replace(base.support_records[0], id="e\u0301", document="e\u0301 🌍", metadata={"e\u0301": "text"})
        artifact = replace(base, manifest=manifest, support_records=(source, *base.support_records[1:]),
                           calibration_rows=({**base.calibration_rows[0], "id": "cal-e\u0301"}, base.calibration_rows[1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.sdmkitmodel"
            write_artifact(path, artifact)
            loaded = load_artifact(path)
            self.assertEqual(loaded.manifest["modelID"].encode(), manifest["modelID"].encode())
            self.assertEqual(loaded.configuration["classNames"], manifest["configuration"]["classNames"])
            self.assertEqual(loaded.manifest["metadata"], manifest["metadata"])
            self.assertEqual(loaded.support_records, artifact.support_records)
            self.assertEqual(loaded.calibration_rows, artifact.calibration_rows)


if __name__ == "__main__":
    unittest.main()
