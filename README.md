# Flight Fuel Analysis

Research project on reliable aircraft fuel-burn estimation from flight trajectories.

## Status

**Phase 0: project scope and submission requirements documented.** No dataset or model results have been produced in this repository yet. Read the [project charter](docs/project-charter.md) and [submission requirements](docs/submission-requirements.md).

## Proposed research question

How accurately can fuel consumed over a completed flight interval be estimated from ADS-B trajectory and aircraft metadata, and how do missing trajectory observations affect error and the calibration of prediction intervals?

This is retrospective interval estimation. It is not a claim to forecast future fuel use from a partial trajectory or to measure operational efficiency.

## Candidate dataset

[EUROCONTROL PRC 2025 Aircraft Fuel Burn Estimation Data Challenge](https://zenodo.org/records/19184662), DOI [10.5281/zenodo.19184662](https://doi.org/10.5281/zenodo.19184662). Its released files contain ACARS-derived interval fuel labels, ADS-B trajectories, and aircraft metadata. The dataset README specifies CC BY 4.0 licensing and describes label-processing limitations. Cite the associated [dataset paper](https://doi.org/10.59490/joas.2026.8750).

Raw data are not stored in this repository. Download, integrity, label quality, and usable sample counts need hands-on verification in Phase 2.

## Work sequence

0. Scope, feasibility, submission mapping.
1. Critical literature review and distinct research hypothesis.
2. Data ingestion, quality and feature-availability audit, fixed splits.
3. Reproducible statistical and ML baselines.
4. Missingness and calibrated-uncertainty experiments with ablations.
5. Focused demonstration and robustness checks.
6. College report, presentation, viva preparation, and paper draft.

A Q1 journal submission is an aspiration conditional on a demonstrable contribution and rigorous results; it is not a guaranteed outcome. The Phase 1 report and final evaluation take priority.

The previous [flight-fuel-efficiency-analysis](https://github.com/milantony05/flight-fuel-efficiency-analysis) project used physics-generated fuel targets. It is background context and will not be reused as evidence of measured-fuel accuracy.
