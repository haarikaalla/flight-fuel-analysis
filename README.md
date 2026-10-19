# Flight Fuel Analysis

Research project on reliable aircraft fuel-burn estimation from flight trajectories.

## Status

**Phase 2 metadata audit complete; full trajectory audit pending.** No model has been fitted. Read the [Phase 2 data acceptance report](docs/phase-2-data-acceptance.md), [project charter](docs/project-charter.md), [critical literature review](docs/phase-1-literature-review.md), [Phase 1 report draft](docs/phase-1-report-draft.md), and [submission requirements](docs/submission-requirements.md). The Phase 1 draft predates the Phase 2 findings and must be revised before submission.

## Proposed research question

How do missing ADS-B observations and imperfect ACARS-derived labels limit fuel estimation over completed flight intervals? Can a coverage-aware ML model use a metadata-only fallback or abstain when its trajectory input is inadequate?

This is retrospective interval estimation. It is not a claim to forecast future fuel use from a partial trajectory or to measure operational efficiency.

## Candidate dataset

[EUROCONTROL PRC 2025 Aircraft Fuel Burn Estimation Data Challenge](https://zenodo.org/records/19184662), DOI [10.5281/zenodo.19184662](https://doi.org/10.5281/zenodo.19184662). Its released files contain ACARS-derived interval fuel labels, ADS-B trajectories, and aircraft metadata. The dataset README specifies CC BY 4.0 licensing and describes label-processing limitations. Cite the associated [dataset paper](https://doi.org/10.59490/joas.2026.8750).

The seven small metadata/label files have been downloaded and checksum verified locally; raw data are not stored in Git. The large trajectory ZIP archives have only been range-probed, so population-wide coverage and ZIP checksums remain open.

## Reproduce the Phase 2 audit

With Python 3.11+, install `requirements.txt`, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m src.flight_fuel.data_audit --download
python -m src.flight_fuel.trajectory_probe --limit 2
python -m src.flight_fuel.dev_subset
```

The first command downloads about 6 MB of released metadata and labels into ignored `data/raw/`, verifies each published MD5 and regenerates `reports/phase-2-metadata-audit.json`. The second reads six flights via HTTP range requests from the official archives and regenerates `reports/phase-2-trajectory-probe.json`; it does **not** verify the full ZIP checksums. The third freezes a small train/rank flight-ID subset by aircraft type in `configs/dev-flights.json`; final-period flight IDs are deliberately excluded. The report describes the sampling and limitations. Downloads and range probing require network access to Zenodo.

## Work sequence

0. Scope, feasibility, submission mapping.
1. Critical literature review and distinct research hypothesis.
2. Data ingestion, quality and feature-availability audit, fixed splits.
3. Reproducible statistical and ML baselines.
4. Observation-availability and label-granularity experiments; test coverage-aware fallback/abstention against matched baselines. Calibrated uncertainty is secondary.
5. Focused demonstration and robustness checks.
6. College report, presentation, viva preparation, and paper draft.

A Q1 journal submission is an aspiration conditional on a demonstrable contribution and rigorous results; it is not a guaranteed outcome. The Phase 1 report and final evaluation take priority.

The previous [flight-fuel-efficiency-analysis](https://github.com/milantony05/flight-fuel-efficiency-analysis) project used physics-generated fuel targets. It is background context and will not be reused as evidence of measured-fuel accuracy.
