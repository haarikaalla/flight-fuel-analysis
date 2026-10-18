# Project charter — Phase 0

**Status:** provisional scope, updated 18 October 2026 after the Phase 1 literature review. This is a planning document, not a results report.

## Problem and purpose

Fuel consumption is not directly present in ordinary ADS-B aircraft states. The previous prototype generated fuel labels from its own physics formula, making its ML evaluation a test of formula reproduction. This project instead investigates fuel estimation against ACARS-derived interval labels and explicitly measures reliability when trajectory observations are incomplete.

**Research question:** How accurately can fuel consumed (kg) during a *completed* flight interval be estimated using ADS-B trajectory measurements and aircraft type/metadata available at the end of that interval? How do missing observations affect error and calibrated uncertainty?

**Provisional title:** Reliable Aircraft Fuel Burn Estimation from Flight Trajectories. Final title and scope are subject to guide review and the Phase 1 literature gap.

## Measurable objectives

1. Reproduce a leakage-audited, flight-separated, chronological evaluation on the PRC 2025 challenge data, with dataset version, exclusions, and usable counts recorded.
2. Establish simple duration/aircraft-type and strong tabular ML baselines for interval fuel in kg; report MAE and RMSE overall and by aircraft type or group, interval duration, and trajectory completeness.
3. Quantify degradation under naturally occurring gaps and controlled missing-observation scenarios, with at least one documented method for predictive intervals. Report empirical coverage and mean interval width, including subgroup results.
4. Test one focused robustness or uncertainty improvement against the baselines using paired evaluation, ablations, and repeated seeds where stochastic methods are used. An improvement is a hypothesis, not a predetermined claim.
5. Deliver a reproducible pipeline and a small demo that explains estimates, uncertainty, and known limitations without claiming measured operational efficiency.

## Prediction boundary and feature policy

- **Unit of prediction:** a specified interval within a flight, bounded by supplied start and end timestamps. Target is `fuel_kg`, the difference of consecutive ACARS fuel-on-board reports; it is not instantaneous fuel flow.
- **Inference setting:** retrospective estimation after the interval has ended. Features may summarize ADS-B observations within that interval and aircraft metadata available independently of fuel telemetry. Interval duration is allowed. This is not a real-time future prediction task.
- **Excluded from predictors:** `fuel_kg`, fuel-on-board values, fuel-report differences, ACARS-only Mach/TAS/CAS, and any feature computed from test labels. Filter trajectory rows to `source == 'adsb'` for the primary claim. Dataset-provided interval boundaries can be used to define the task, but their availability outside this benchmark must be stated.
- **Weather:** optional only after time, location, and altitude alignment is reproducible; no weather-based novelty claim from merely adding variables.
- **Output:** point estimate in kg and, for the main study, a calibrated prediction interval. CO₂ conversion is secondary and remains an estimate, not measured emissions.

## Data feasibility and limitations

The [released dataset](https://zenodo.org/records/19184662) reports 11,037 training, 1,888 ranking, and 2,836 final flights; 193,275 labeled intervals across 27 aircraft types; and roughly 4.9 GB of files. It includes trajectory archives, metadata, and interval labels. The [README](https://zenodo.org/records/19184662/files/readme.txt?download=1) documents CC BY 4.0 and explains that fuel labels are differences of sparse ACARS reports. Ambiguous units were checked using OpenAP; intervals shorter than 5 or longer than 60 minutes were excluded. These are observed-derived, quality-controlled labels, but not error-free direct engine-flow measurements.

**Verification still outstanding:** actual download and checksum check; row-level schema and joins; usable ADS-B intervals after quality filters; aircraft-type coverage; disk/RAM budget; interpretation of missing intervals; possible repeated aircraft/route patterns across periods. Phase 2 begins on a small fixed subset, then scales. Do not upload raw 4.9 GB data or generated models to GitHub.

## Evaluation protocol

1. Use train period (April–August 2025) for fitting, ranking period (September) for model choice/calibration, and final period (October) as a locked chronological test. Check flight ID and any duplicated segment/aircraft identifiers across partitions. Never randomly split intervals from one flight between train and validation.
2. Compare duration/type and physics-inspired references with a strong tabular model before attempting a sequence or deep-learning model. Compute overall MAE/RMSE in kg, normalized or relative metrics only where denominators are safe, and subgroup metrics. Evaluate by **flight** as well as by interval to avoid large flights dominating claims.
3. Learn preprocessing and imputation on training data only. Fit interval calibration on held-out ranking data, then measure coverage and width once on final data. Analyze whether global coverage hides poor coverage for rare aircraft or sparse trajectories.
4. Define realistic missingness using the observed gap distribution; distinguish natural gaps from artificial random or contiguous removal. Alter the input only, never the label. Report error and coverage at each severity.
5. Preserve experiment configuration, seeds, package versions, dataset citation and reproducible figures. Never tune against the locked final split.

## Research positioning

Published PRC 2025 challenge papers already report physics-guided boosting and weather-enhanced fuel estimation. A paper published in October 2026 also studies data quality and conformal uncertainty with a small observed subset of this benchmark alongside synthetic experiments. The [Phase 1 comparison matrix](phase-1-literature-review.md) narrows our question to a full-benchmark, ADS-B-only, natural-gap and aircraft-group calibration study under chronological testing. Whether this is sufficiently distinct for a Q1 journal remains open; no quartile, acceptance, or performance improvement is promised.

Core references: [dataset paper](https://doi.org/10.59490/joas.2026.8750); [physics-guided challenge paper](https://doi.org/10.59490/joas.2026.8764); [weather-enhanced challenge paper](https://doi.org/10.59490/joas.2026.8766); [recent reliability paper](https://doi.org/10.3390/en19204797). The comparison matrix includes related trajectory and uncertainty studies.

## Gates, risks, and mitigations

| Gate/risk | Check and response |
| --- | --- |
| Dataset usability | Verify archive integrity, labels, joins, ADS-B-only features, and workable subset. If not feasible, revisit target and dataset before training. |
| Feature leakage | Maintain an allowlist with provenance and observation time; review each field before baseline experiments. |
| Uncertain labels | Report ACARS unit inference and quality filters; avoid equating labels with precise direct fuel-flow measurements. |
| Aircraft imbalance and temporal shift | Report per-group sample counts, errors, and coverage. Keep the later test period locked. |
| Research overlap | Complete the Phase 1 literature matrix; narrow the contribution or report a rigorous negative result. |
| Resource needs | Begin with train metadata and a stratified flight subset; record runtime/memory before full extraction. |
| Submission risk | Prioritize college rubric and guide deadlines; paper writing does not displace required report, diary, presentation, or demo. |

## Phase 0 completion and next gate

The charter, submission mapping, literature review and Phase 1 report content establish the proposal. Dataset usability and paper-level novelty are **documented hypotheses, not empirically verified in this repository**. Phase 2 is the data acceptance gate. Implementation starts only after the allowed feature list and split protocol are explicit.

**Inputs to confirm with student/guide:** current Phase 1 and final deadlines; solo versus group membership and responsibilities; guide-approved title; current final-report rules; available storage/RAM/GPU; whether additional data access or an aircraft-performance reference is available. The 2025 archive dates are historical, not current deadlines.
