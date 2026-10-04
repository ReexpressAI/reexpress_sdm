# Changelog

## 0.4.9

This update enables training models with larger input files. Earlier versions had a more conservative limit on capacity. Additional preflight training and data checks have been added, along with memory advisories. (See docs/tools/README.md for a tool to estimate the capacity available for given compute constraints.) This is a minor update that does not change the core behavior or training/calibration methods. Concretely:

- Increase the shared model-manifest safety budget from 64 MiB to 4 GiB,
  without adding a document-count limit or changing model/scoring formats.
- Check tensor shapes and a conservative ID-aware generated-manifest budget
  before pooled training copies or normalization; check exact encoded manifest
  bytes before export writes large payloads. Models above the old 64 MiB limit
  require updated readers.
- Add a non-blocking CLI memory advisory on stderr reporting raw Float32 input
  size and installed RAM when available. It is not a peak-memory estimate.

- Add a non-blocking CLI advisory before opening model manifests larger than
  512 MiB. The notice checks file size only and leaves stdout reports unchanged.

## 0.4.8

- Revised tutorials.
- Public GitHub artifacts starting from here.

## 0.4.7

- Publish the getting-started tutorial from `docs/tutorials/` for GitHub Pages,
  with direct links from the README and package metadata.
- Load tutorial screenshots from their original public URLs so the page stays
  small and displays updated images.
- By design, the tutorial HTML, companion shell source, and Pages configuration
  marker (docs/.nojekyll) remain outside the source distribution (due to MANIFEST.in).

## 0.4.6

- Rename cumulative evaluation report keys from `coverageCount` to
  `admissionCount` and from `coverage` to `admission` in both the `centroid` and
  `lower` `perAlphaCumulative` results. CLI report consumers and Python API
  callers must update their key lookups; the old keys are no longer emitted.
- Admission is the proportion of evaluated documents whose assigned region meets or exceeds the selected α threshold.
- This release changes terminology and report keys only; selection thresholds,
  denominators, scoring, calibration, and numerical results are unchanged.
