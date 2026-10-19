# Flight Fuel Analysis

Research project on reliable aircraft fuel-burn estimation from flight trajectories.

## Status

**Phase 4 selective estimation and controlled observation-loss experiments implemented.** Read the [Phase 4 report](docs/phase-4-results.md) and [Jupyter walkthrough](notebooks/phase-4-walkthrough.ipynb), plus the [Phase 3 baseline report](docs/phase-3-baselines.md), [Phase 2 data acceptance report](docs/phase-2-data-acceptance.md), [project charter](docs/project-charter.md), [critical literature review](docs/phase-1-literature-review.md), [Phase 1 report draft](docs/phase-1-report-draft.md), and [submission requirements](docs/submission-requirements.md). The Phase 1 draft predates the revised research direction and must be updated before submission.

## Proposed research question

How do missing ADS-B observations and imperfect ACARS-derived labels limit fuel estimation over completed flight intervals? Can a coverage-aware ML model use a metadata-only fallback or abstain when its trajectory input is inadequate?

This is retrospective interval estimation. It is not a claim to forecast future fuel use from a partial trajectory or to measure operational efficiency.

## Candidate dataset

[EUROCONTROL PRC 2025 Aircraft Fuel Burn Estimation Data Challenge](https://zenodo.org/records/19184662), DOI [10.5281/zenodo.19184662](https://doi.org/10.5281/zenodo.19184662). Its released files contain ACARS-derived interval fuel labels, ADS-B trajectories, and aircraft metadata. The dataset README specifies CC BY 4.0 licensing and describes label-processing limitations. Cite the associated [dataset paper](https://doi.org/10.59490/joas.2026.8750).

The seven small metadata/label files and all three trajectory ZIP archives were downloaded and verified against the published MD5s. Raw data are not stored in Git. Full train/rank/final ADS-B interval features have been extracted. The October final split was evaluated with the committed Phase 4 protocol; see the Phase 4 report for provenance, a 100% comparator correction, and limits.

## Reproduce the Phase 2 audit

With Python 3.11+, install `requirements.txt`, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m src.flight_fuel.data_audit --download
python -m src.flight_fuel.trajectory_probe --limit 2
python -m src.flight_fuel.dev_subset
```

The first command downloads about 6 MB of released metadata and labels into ignored `data/raw/`, verifies each published MD5 and regenerates `reports/phase-2-metadata-audit.json`. The second reads six flights via HTTP range requests from the official archives and regenerates `reports/phase-2-trajectory-probe.json`; it does **not** verify the full ZIP checksums. The third freezes a small train/rank flight-ID subset by aircraft type in `configs/dev-flights.json`; final-period flight IDs are deliberately excluded. The report describes the sampling and limitations. Downloads and range probing require network access to Zenodo.

For the **pending population-wide gate**, allow about 4.9 GB of download space plus extracted working space, then run:

```bash
python -m src.flight_fuel.download_archives
python -m src.flight_fuel.full_coverage_audit
python -m unittest discover -s tests
```

The downloader resumes interrupted `.part` files and verifies each archive against the Zenodo MD5. The optional coverage audit streams one flight at a time and writes aggregate results to `reports/phase-2-full-coverage-audit.json`. The Phase 3 feature pipeline already verifies the train/rank ZIPs and counts intervals with no ADS-B position; the separate all-three-split coverage audit has not been run. October remains reserved for a locked test.

## Phase 3 baselines

```bash
python -m src.flight_fuel.baselines --output reports/phase-3-metadata-rank.json
python -m src.flight_fuel.features --splits train rank
python -m src.flight_fuel.baselines --feature-dir data/processed --output reports/phase-3-rank-baselines.json
python -m src.flight_fuel.plots --report reports/phase-3-rank-baselines.json
python -m unittest discover -s tests
```

The [baseline report](docs/phase-3-baselines.md) has the fixed feature contract, metrics, subgroup counts and limitations. The Phase 3 model command refuses to read the final split. The September ADS-B model achieves 92.95 kg MAE and 256.70 kg RMSE; the simple aircraft-type median-rate baseline achieves 94.81 kg and 266.59 kg. The ADS-B model is **not** better on every aircraft or on intervals with no ADS-B input.

## Phase 4 selective estimation

The [protocol](configs/phase-4-protocol.json) fixes a type-rate fallback on intervals with zero ADS-B positions, trains an absolute-error risk selector on early September, calibrates acceptance thresholds in mid-September, and tests them in late September and October. A controlled outage experiment removes scattered or contiguous ADS-B points from the same September validation flights before re-extracting features. The [report](docs/phase-4-results.md) states the error, acceptance, aircraft-group access, label-granularity diagnostics, and important limits. The [Jupyter notebook](notebooks/phase-4-walkthrough.ipynb) offers figures and tables without duplicating the repeatable Python training pipeline.

```bash
python -m src.flight_fuel.features --splits final
python -m src.flight_fuel.phase4 --output reports/phase-4-results.json
python -m unittest discover -s tests
```

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
