# Shared format and numerical fixtures

These small synthetic fixtures are vendored here so the Python tests run from
this repository and its source distribution without another checkout. They
also provide fixed reference data for cross-language interoperability checks.

| Fixture | Purpose |
| --- | --- |
| `datasets/dense.sdmdataset/` | Complete mapped embedding vectors and dataset metadata. |
| `datasets/partial.sdmdataset/` | Missing embeddings/attributes, independent feature matrices, and source metadata. |
| `portable-training-run.json` | Balanced training metrics, selected checkpoints, optional values, and seed serialization. |
| `python-training-lifecycle.sdmkitmodel/` | Fixed trained model used to compare runtime outputs. |
| `python-training-lifecycle-inputs.json` | Inputs associated with that model. |
| `python-training-lifecycle-expected.json` | Expected numerical outputs and metadata. |

Keep every file inside each model and dataset directory, including `.npy` and
`.f32` arrays. The manifests contain hashes of their corresponding data files.
Building or installing the package does not regenerate or modify these fixtures.

The neighboring `sdm-golden-v1.json` and `score-interchange/` fixtures cover the
SDM math and scored-document interchange. The latter can be regenerated with
`PYTHONPATH=src python scripts/generate_score_interchange_fixture.py` from the
repository root.
