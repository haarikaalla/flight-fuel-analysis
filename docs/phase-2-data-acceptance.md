# Phase 2 data acceptance and revised research gate

**Audit date:** 19 October 2026. **Dataset:** [Zenodo record 19184662](https://zenodo.org/records/19184662), version v1, CC BY 4.0. The associated [dataset paper](https://doi.org/10.59490/joas.2026.8750) explains ACARS fuel-on-board differencing and OpenAP-assisted unit inference. The numerical counts below come from downloaded, MD5-verified Parquet metadata and labels, not from the article's summary. Reproduce them using `python -m src.flight_fuel.data_audit --download`. The original six-flight probe used range requests; **all three full ZIPs were subsequently downloaded and checksum verified** for Phase 3, as recorded in the current gate below.

## What the released files actually support

| Split | Period | Flights | Fuel intervals | Intervals <5 min | Intervals <1 min | Integer-pound-lattice labels* |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train | Apr 12–Aug 31, 2025 | 11,037 | 131,530 | 48,856 (37.1%) | 3,358 | 70,659 (53.7%) |
| Rank | Sep 1–27, 2025 | 1,888 | 24,289 | 10,549 (43.4%) | 950 | 13,867 (57.1%) |
| Final | Oct 1–31, 2025 | 2,836 | 37,456 | 17,318 (46.2%) | 1,450 | 21,788 (58.2%) |

*An exact match to an integer multiple of 0.453592 kg within a numerical tolerance. This is a **diagnostic signature consistent with pound-based reporting**, not proof of any individual record's original unit or precision. Details and per-type counts are in [`reports/phase-2-metadata-audit.json`](../reports/phase-2-metadata-audit.json).

- All 193,275 target rows have positive fuel and duration, non-null target and interval fields, unique `idx` within their split, and a matching flight ID. All 15,761 flight IDs are distinct between chronological splits. This checks keys, **not** whether repeated physical aircraft occur in multiple splits; aircraft registration is not in these metadata.
- The released data **contradict the README statement that intervals shorter than five minutes are excluded**. Durations under five minutes are common. We must publish both all-valid-row and `duration >= 5 min` sensitivity results; do not silently discard 40% of the benchmark or pretend the filter was already applied.
- Aircraft types are imbalanced. Train contains 26 types; final contains an additional `B734`. Its frequency and treatment need a prespecified rare/unseen-type rule. A random interval split would violate the evaluation design.
- Label magnitudes range widely, with a train maximum above 32,000 kg per interval. Exact values such as 200 kg recur often. This deserves interval-duration/type-stratified inspection before interpreting extreme residuals as model failure.
- The dataset paper acknowledges imperfect ACARS timing, ambiguous original fuel units and a broad OpenAP acceptance margin. Therefore `fuel_kg` is a quality-controlled observed-derived benchmark label, not an error-free engine sensor value. The raw FOB values and original unit tags are **not** in the released fuel tables, so true per-record label precision cannot be recovered from the lattice pattern.

## Trajectory probe: schema and input availability

[`reports/phase-2-trajectory-probe.json`](../reports/phase-2-trajectory-probe.json) inspects the lexicographically first **two flights in each split** through range requests. This deterministic, six-flight probe establishes parsing and reveals possible failure modes; it is not a prevalence estimate.

- All six flight Parquet files include interleaved `source = adsb` and `source = acars`. The latter must be excluded from **predictors**, including ACARS-only `mach`, `TAS`, and `CAS`. Interval start/end and `fuel_kg` are a label/task definition, never input telemetry.
- One sampled train flight (`prc770831136`) has **zero ADS-B records in all ten labeled intervals**. Others have at least one ADS-B record within each interval in this small probe. Empty-input intervals must be retained and reported separately, with a metadata-only fallback; quietly dropping them would bias the evaluation toward well-observed flights.
- Raw sampled cruise altitudes near 35,000–39,000 and ground speeds around 430–500 match **feet and knots** as aviation values, while the dataset README and paper table describe altitude in meters and groundspeed in m/s. Raw vertical-rate magnitudes near 2,000 are likewise inconsistent with m/s. This is a **suspected documentation/unit mismatch** based on six sampled flights and the paper's own 35,000 ft / 480 knot example, not a verified conversion rule for every row. Do not derive physical features or use a phase threshold until a larger systematic audit resolves units and any mixed conversions.
- The sampled flights contain substantial nulls in ADS-B `groundspeed`, `track` and `vertical_rate`. Max gaps in the sampled ADS-B timestamps reach 13,382 seconds. A maximum gap alone does not tell us which labeled intervals are observed; compute interval-wise counts, edge gaps, longest within-interval gap and a time-coverage proxy on the full archives.

## Allowed feature contract (retrospective interval estimation)

| Item | Allowed use | Reason / caveat |
| --- | --- | --- |
| `flight_id`, `idx` | Joins and split audits only | High-cardinality IDs could memorize flight or interval; never predictors. |
| `start`, `end` | Define evaluation intervals; use duration | Boundaries come from ACARS messages. This is not deployable from public ADS-B alone without a separate interval-definition scheme. |
| `fuel_kg` | Training target and held-out scoring only | Never impute inputs from the target or use final labels for selection. |
| `aircraft_type` | Predictor | Independently known type; unseen-type fallback must be defined. |
| `takeoff`, `landed`, airports | Audit/context; optional feature only after availability review | EUROCONTROL metadata may not be available in a pure live ADS-B setting. Freeze an explicit feature budget before training. |
| `source == 'adsb'`: timestamp, position, altitude, groundspeed, track, vertical rate | Candidate predictors after a unit and missingness audit | Aggregate only points in `[start, end)`; preserve empty intervals. Groundspeed and vertical rate need validated units before derived physics. |
| `source == 'acars'`, Mach/TAS/CAS, raw FOB | Excluded from predictors | ACARS is the label channel; source leakage would invalidate an ADS-B claim. |

## Distinct research direction and gate

**Provisional revised question:** *How does the observation process limit aircraft fuel-interval estimation from public surveillance inputs?* We will separate (i) empty/sparse ADS-B intervals, (ii) interval duration and suspected label-resolution regimes, and (iii) performance by aircraft type and period. The **ML component** compares a metadata-only estimator with an ADS-B feature model and a coverage-aware model that can fall back or abstain when no useful trajectory is observed. Paired error and selective-risk/coverage curves must show whether the fallback or abstention rule actually helps. A duration-filter sensitivity and error-vs-label-lattice analysis test whether apparent gains are artifacts of telemetry granularity.

This question is **more specific** than adding missingness and conformal intervals, which [Stawiarska (2026)](https://doi.org/10.3390/en19204797) already studies partly on this benchmark. The dataset authors already discuss ACARS unit ambiguity, so recognizing it is **not** novel either. A paper would require a full-scale, well-controlled finding about selective prediction and measurement limitations, a deeper comparison with related work, and ideally independent validation. We cannot promise uniqueness or Q1 acceptance from the present probe. Group-conditional uncertainty is secondary until these first-order data issues are resolved.

**Current gate:** metadata acceptance **passed**; all three ZIPs were subsequently downloaded and matched the published MD5s. [Phase 3 feature extraction](phase-3-baselines.md) processed every train and rank flight and found **45,874 / 131,530** training intervals and **6,395 / 24,289** ranking intervals with no usable ADS-B position in `[start, end)`. These population counts supersede the six-flight probe for those periods. A standalone all-split coverage audit, systematic unit reconciliation and label-regime prevalence study remain pending. Do not interpret raw altitude/speed as SI physical inputs yet. October remains locked for model evaluation.

`src/flight_fuel/download_archives.py` supports resumable ZIP downloads, and `src/flight_fuel/full_coverage_audit.py` implements the optional all-split population audit. A synthetic interval-boundary test passes. The fixed train/rank feature extraction completed on verified full archives; see the Phase 3 report for results and the remaining measurement caveats.

For pipeline development, [`configs/dev-flights.json`](../configs/dev-flights.json) fixes two flights per aircraft type in train and rank using a salted hash of flight ID, without consulting labels. It is a plumbing sample, not an evaluation set; the final split remains untouched for model selection.
