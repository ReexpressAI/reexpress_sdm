# Changelog

## 0.4.6

- Rename cumulative evaluation report keys from `coverageCount` to
  `admissionCount` and from `coverage` to `admission` in both the `centroid` and
  `lower` `perAlphaCumulative` results. CLI report consumers and Python API
  callers must update their key lookups; the old keys are no longer emitted.
- Admission is the proportion of evaluated documents whose assigned region meets or exceeds the selected α threshold.
- This release changes terminology and report keys only; selection thresholds,
  denominators, scoring, calibration, and numerical results are unchanged.
