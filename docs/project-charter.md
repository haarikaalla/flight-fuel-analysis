# Project charter — Phase 0

**Status:** provisional scope, updated 19 October 2026 after the Phase 2 metadata audit. Read [Phase 2 data acceptance](phase-2-data-acceptance.md) before using the earlier Phase 1 hypotheses. This is a planning document, not a model results report.

## Problem and purpose

Fuel consumption is not directly present in ordinary ADS-B aircraft states. The previous prototype generated fuel labels from its own physics formula, making its ML evaluation a test of formula reproduction. This project instead investigates fuel estimation against ACARS-derived interval labels and explicitly measures reliability when trajectory observations are incomplete.

**Research question:** How accurately can fuel consumed (kg) during a *completed* flight interval be estimated using ADS-B trajectory measurements and aircraft type/metadata available at the end of that interval? How do missing observations affect error and calibrated uncertainty?

**Provisional title:** Reliable Aircraft Fuel Burn Estimation from Flight Trajectories. Final title and scope are subject to guide review and the Phase 1 literature gap.

**Phase 2 scope revision:** A closely related 2026 paper already tests missingness and uncertainty. Our primary research gate is now the *observation and measurement boundary*: zero/sparse ADS-B intervals, suspected label quantization, actual interval durations, and whether coverage-aware ML fallback or abstention improves error at a stated acceptance rate. Prediction-interval calibration remains optional/secondary until the full-archive audit and novelty check. This revision supersedes the earlier objectives 3–4 where they assume missingness-plus-conformal is the central contribution.

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

**Verified in Phase 2:** all seven small metadata/label files against official MD5 hashes, schema/joins, split ID disjointness, aircraft counts and label/duration diagnostics. The README's reported five-minute minimum does not match 76,723 released intervals shorter than five minutes. A deterministic six-flight range probe confirms trajectory schema and finds one flight with no ADS-B rows inside ten target intervals. **Still outstanding:** full ZIP download/checksums, full-archive ADS-B coverage and unit audit, storage/RAM profile, repeated-aircraft assessment. See the Phase 2 acceptance report. Do not upload raw 4.9 GB data or generated models to GitHub.

## Evaluation protocol

1. Use train period (April–August 2025) for fitting, ranking period (September) for model choice/calibration, and final period (October) as a locked chronological test. Check flight ID and any duplicated segment/aircraft identifiers across partitions. Never randomly split intervals from one flight between train and validation.
2. Compare duration/type and physics-inspired references with a strong tabular model before attempting a sequence or deep-learning model. Compute overall MAE/RMSE in kg, normalized or relative metrics only where denominators are safe, and subgroup metrics. Evaluate by **flight** as well as by interval to avoid large flights dominating claims.
3. Learn preprocessing on training data only. First compare a metadata-only fallback with a trajectory model, including intervals with no ADS-B records. Prespecify any abstention threshold on ranking data and evaluate accepted-case error **and** fraction accepted on final data. Calibrate intervals on ranking data only if uncertainty remains in scope.
4. Characterize natural absence and gap severity on all released archives before designing synthetic corruption. Preserve every labeled interval in the initial benchmark, including sub-five-minute cases, and publish a separate duration-filter sensitivity. Never use the label-lattice diagnostic as an inference-time predictor.
5. Preserve experiment configuration, seeds, package versions, dataset citation and reproducible figures. Never tune against the locked final split.

## Research positioning

Published PRC 2025 challenge papers already report physics-guided boosting and weather-enhanced fuel estimation. A paper published in October 2026 also studies data quality and conformal uncertainty with a small observed subset of this benchmark alongside synthetic experiments. Phase 2 therefore changes the primary contribution hypothesis to **selective fuel estimation under measured observation and label limitations**, detailed in the [acceptance report](phase-2-data-acceptance.md). This is not a proven novelty claim. Dataset authors already identify unit ambiguity. Whether the refined study is sufficiently distinct for a Q1 journal remains open; no quartile, acceptance, or performance improvement is promised.

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
