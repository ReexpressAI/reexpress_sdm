# Changelog

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
