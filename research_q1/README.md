# Q1 research extension

This directory is an additive, non-destructive research layer. It imports the
existing project but does not modify its source, reports, tests, or dashboard.

The proposed contribution is **PI-MCS**: Physics-Informed Mondrian Conformal
Selection with an Exact Risk Budget. The method and hypotheses are
pre-specified in `configs/study.json`. Results are generated only from approved,
locally supplied data that pass `validation/data_contract.py`.

No PI-MCS performance result is currently claimed. The repository checkout did
not contain the raw trajectory, weather, aircraft-performance, operator, or
multi-year label tables required for the confirmatory experiments. See
`outputs/STATUS.md` and `external/DATA_REQUIREMENTS.md`.

## Safety boundary

- The frozen future-confirmation labels are not read by this extension.
- Splits are grouped by time, route, aircraft type, or aircraft family.
- Test labels never determine a split, tune a model, or calibrate an interval.
- Synthetic data are permitted only in unit tests and are never reported as
  research evidence.
- The existing dashboard remains unchanged.

